import torch
import folder_paths
import nodes
import json
import sys
import os


## I need to finish up sage...
SAGE_AVAILABLE = False
try:
    import sageattention
    SAGE_AVAILABLE = True
except ImportError:
    pass

def get_node_class(class_name):
    if class_name in nodes.NODE_CLASS_MAPPINGS:
        return nodes.NODE_CLASS_MAPPINGS[class_name]
    return None

class JonModelOnlyLoader:
    @classmethod
    def INPUT_TYPES(s):
        sage_kernels = [
            "disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda",
            "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda",
            "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3"
        ]

        ckpt_list = ["None"] + folder_paths.get_filename_list("checkpoints")
        raw_gguf = folder_paths.get_filename_list("unet_gguf") if "unet_gguf" in folder_paths.folder_names_and_paths else []
        gguf_unet_list = ["None"] + raw_gguf

        return {
            "required": {
                "gguf_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "ON = Use GGUF UNet | OFF = Use Standard Checkpoint."
                }),
                "sage_kernel": (sage_kernels, {
                    "default": "disabled",
                    "tooltip": "apply sage attention"
                }),
                "enable_tf32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable TensorFloat-32 (TF32)."
                }),
                "lora_stack_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "display": "hidden"
                }),
            },

            "optional": {
                "ckpt_name": (ckpt_list, {
                    "default": "None",
                    "tooltip": "Standard Checkpoint (.safetensors)."
                }),
                "gguf_unet_name": (gguf_unet_list, {
                    "default": "None",
                    "tooltip": "GGUF UNet Model."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_and_process"
    CATEGORY = "josephs_odd_nodes/loaders"

    OUTPUT_TOOLTIPS = ("The loaded Model (UNet/Transformer) with LoRAs applied.",)

    DESCRIPTION = "Loads ONLY the Model (UNet) from a Checkpoint or GGUF file. Useful for complex workflows where CLIP/VAE are loaded separately."

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return str(kwargs)

    def load_and_process(self, gguf_model, sage_kernel, enable_tf32, lora_stack_json,
                         ckpt_name="None", gguf_unet_name="None", **kwargs):

        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
             torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        model_obj = None

        # Model
        if gguf_model:
            UnetLoaderGGUF = get_node_class("UnetLoaderGGUF")
            if not UnetLoaderGGUF:
                raise ImportError("[JonModelOnlyLoader] GGUF Mode enabled but 'ComfyUI-GGUF' nodes not found in registry! Is it installed?")

            if gguf_unet_name == "None" or not gguf_unet_name:
                raise ValueError("[JonModelOnlyLoader] GGUF Mode is ON, but 'gguf_unet_name' is missing or None!")

            print(f"[JonModelOnlyLoader] Loading GGUF Model: {gguf_unet_name}")
            model_obj = UnetLoaderGGUF().load_unet(gguf_unet_name)[0]
        else:
            if ckpt_name == "None" or not ckpt_name:
                raise ValueError("[JonModelOnlyLoader] Standard Mode is ON, but 'ckpt_name' is missing or None!")

            print(f"[JonModelOnlyLoader] Loading Checkpoint (Model Only): {ckpt_name}")
            ckpt_loader = nodes.CheckpointLoaderSimple()
            model_obj, _, _ = ckpt_loader.load_checkpoint(ckpt_name)

        # LoRA's
        current_model = model_obj

        try:
            ui_state = json.loads(lora_stack_json)
        except Exception:
            ui_state = {}

        if ui_state:
            lora_loader = nodes.LoraLoader()
            for slot_id, data in ui_state.items():
                if not data.get("enabled", True):
                    continue

                name = data.get("name")
                if not name or name == "Select LoRA..." or name == "null":
                    continue
                input_key = data.get("input_name")
                strength = 1.0
                if input_key and input_key in kwargs:
                    strength = float(kwargs[input_key])

                try:
                    # Pass None for CLIP since this is a Model-Only loader
                    (current_model, _) = lora_loader.load_lora(current_model, None, name, strength, strength)
                    print(f"[JonModelOnlyLoader] LoRA: {name} @ {strength}")
                except Exception as e:
                    print(f"[JonModelOnlyLoader] Failed LoRA {name}: {e}")

        # Sage
        if sage_kernel != "disabled":
            if SAGE_AVAILABLE:
                current_model = current_model.clone()
                if "sageattention" not in current_model.model_options:
                     current_model.model_options["sageattention"] = {}
                current_model.model_options["sageattention"]["kernel"] = sage_kernel
                current_model.model_options["sageattention"]["compile"] = False
                print(f"[JonModelOnlyLoader] SageAttention Applied: {sage_kernel}")

        return (current_model,)

NODE_CLASS_MAPPINGS = {"JonModelOnlyLoader": JonModelOnlyLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"JonModelOnlyLoader": "JonLoader(Model Only)"}
