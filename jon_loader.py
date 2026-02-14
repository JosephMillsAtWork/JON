import torch

import os
import sys
import json
import logging

import folder_paths
import nodes
import comfy.utils
import comfy.sd
import comfy.model_management as model_management

from .jon_utils import LIGHTRICKS_AVAILABLE, SAGE_AVAILABLE, get_node_class, get_sage_func

class JonLoader:
    @classmethod
    def INPUT_TYPES(s):
        sage_kernels = [
            "disabled",
            "auto",
            "sageattn_qk_int8_pv_fp16_cuda",
            "sageattn_qk_int8_pv_fp16_triton",
            "sageattn_qk_int8_pv_fp8_cuda",
            "sageattn_qk_int8_pv_fp8_cuda++",
            "sageattn3"
        ]
        ## https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py#L976
        clip_type = [
            "stable_diffusion",
            "stable_cascade",
            "sd3",
            "stable_audio",
            "mochi",
            "ltxv",
            "pixart",
            "cosmos",
            "lumina2",
            "wan",
            "hidream",
            "chroma",
            "ace",
            "omnigen2",
            "qwen_image",
            "hunyuan_image",
            "flux",
            "flux2",
            "ovis",
            "sdxl"
        ]

        # Models lists
        model_types = ["checkpoint", "gguf", "diffusion"]
        ckpt_list = ["None"] + folder_paths.get_filename_list("checkpoints")
        diffusion_list = ["None"] + folder_paths.get_filename_list("diffusion_models")

        raw_gguf = folder_paths.get_filename_list("unet_gguf") if "unet_gguf" in folder_paths.folder_names_and_paths else []
        gguf_unet_list = ["None"] + raw_gguf

        # Text Encoders lists
        clip_list_standard = ["None", "Embedded"] + folder_paths.get_filename_list("clip")
        raw_gguf_clip = folder_paths.get_filename_list("clip_gguf") if "clip_gguf" in folder_paths.folder_names_and_paths else []
        raw_clip_combined = sorted(list(set(folder_paths.get_filename_list("clip") + raw_gguf_clip)))
        clip_list_gguf = ["None", "Embedded"] + raw_clip_combined

        # VAE lists
        vae_list = ["None", "Embedded"] + folder_paths.get_filename_list("vae")

        return {

            "required": {
                "model_type": (model_types , {
                    "default": "gguf",
                    "tooltip": "The Type of Model to load"
                }),

                "clip_model_type": (["safetensor", "gguf"], {
                    "default" : "gguf"
                }),

                "dual_clip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load two separate CLIP models."
                }),

                "dual_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load Advanced VAE models"
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
                    "tooltip": "checkpoints models safetensor file"
                }),

                "unet_name": (diffusion_list, {
                    "default" : "None",
                    "tooltip": "diffusion models safetensor file"
                }),

                # "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],{}), << thinking about adding this

                "gguf_unet_name": (gguf_unet_list, {
                    "default": "None",
                    "tooltip": "GGUF file for the model"
                }),

                "clip_type": (clip_type, {
                    "default": "stable_diffusion",
                    "tooltip": "The type of clip.(upstream provider name)"
                }),
                "clip_name": (clip_list_standard, {
                    "default": "Embedded",
                    "tooltip": "primary text encoders file"
                }),
                "gguf_clip_name": (clip_list_gguf, {
                    "default": "Embedded",
                    "tooltip": "primary text encoders GGUF file"
                }),


                "secondary_clip_model_type": (["safetensor", "gguf"], {
                    "default" : "gguf",
                    "tooltip": "use this to change the secondary text encoder type(safetensor, gguf)"
                }),

                "secondary_clip_type": (clip_type, {
                    "default": "stable_diffusion",
                    "tooltip": "The type of clip.(upstream provider name)"
                }),

                "clip_name_2": (clip_list_standard, {
                    "default": "None",
                    "tooltip": "secondary text encoders file"
                }),
                "gguf_clip_name_2": (clip_list_gguf, {
                    "default": "None",
                    "tooltip": "secondary text encoders GGUF file"
                }),


                "vae_name": (vae_list, {
                    "default": "Embedded",
                    "tooltip": "primary VAE file"
                }),


                "vae_audio_name": (vae_list, {
                    "default": "Embedded",
                    "tooltip": "secondary VAE file"
                }),
                "vae_audio_device": (["default", "cpu"], {
                    "default": "default",
                    "tooltip": "secondary VAE host device file"
                }),
                "vae_audio_dtype": (["default", "fp32", "fp16", "bf16"], {
                    "default": "default",
                    "tooltip": "secondary VAE precision mode"
                }),

                "sage_kernel": (sage_kernels, {
                    "default": "disabled",
                    "tooltip": "apply sage attention"
                }),
                "enable_tf32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable TensorFloat-32 (TF32)."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("model", "clip", "vae", "vae_audio")
    FUNCTION = "load_and_process"
    CATEGORY = "josephs_odd_nodes/loaders"
    OUTPUT_TOOLTIPS = (
        "The loaded Model.",
        "The loaded CLIP.",
        "The loaded VAE.",
        "The Secondary loaded VAE.(if enabled).")

    DESCRIPTION = "Joseph's Odd Node Loader is a advanced loader that is meant to aid in the loading of models, text encoders, vae's and lora's. The node also has sage attention built in. This node depends on CompyUi core, CompyUi-GGUF and inspriation from kjnodes. When LoRA's are added, A input is added. These inputs are intentended to be used(connected) with JobFloatSlider"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return str(kwargs)

    def load_and_process(self,
                         model_type,
                         clip_model_type,
                         dual_clip,
                         dual_vae,
                         lora_stack_json,
                         ckpt_name="None", unet_name="None", gguf_unet_name="None",
                         clip_type="stable_diffusion", clip_name="Embedded", gguf_clip_name="Embedded",
                         secondary_clip_model_type="gguf", secondary_clip_type="stable_diffusion",  clip_name_2="None",  gguf_clip_name_2="None",
                         vae_name="Embedded",
                         vae_audio_name="Embedded", vae_audio_device="default", vae_audio_dtype="default",
                         sage_kernel="default", enable_tf32=False,
                         **kwargs):

        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
             torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        model_obj = None
        clip_obj = None

        vae_obj = None
        vae_audio_obj = None

        # MODEL "checkpoint", "gguf", "diffusion"]
        if model_type == "checkpoint":
            if ckpt_name in ["None", None, ""]:
                raise ValueError("Standard Mode ON but no file selected!")

            print(f"[JonLoader] Loading Checkpoint: {ckpt_name}")
            ckpt_loader = nodes.CheckpointLoaderSimple()
            model_obj, clip_obj, vae_obj = ckpt_loader.load_checkpoint(ckpt_name)

        elif model_type == "gguf":
            UnetLoaderGGUF = get_node_class("UnetLoaderGGUF")
            if not UnetLoaderGGUF:
                raise ImportError("GGUF nodes not found!")
            if gguf_unet_name in ["None", None, ""]:
                raise ValueError("GGUF Mode ON but no file selected!")

            print(f"[JonLoader] Loading GGUF: {gguf_unet_name}")
            model_obj = UnetLoaderGGUF().load_unet(gguf_unet_name)[0]

        elif model_type == "diffusion":
            if unet_name in ["None", None, ""]:
                raise ValueError("diffusion Mode ON but no file selected!")
            print(f"[JonLoader] Loading Diffusion Model: {unet_name}")
            diffusion_loader = nodes.UNETLoader()
            model_obj = diffusion_loader.load_unet(unet_name, "default")[0]


        # TEXT ENCODER/CLIP
        gguf_clip_1 = False
        gguf_clip_2 = False

        if clip_model_type == "gguf":
            gguf_clip_1 = True

        if secondary_clip_model_type == "gguf":
            gguf_clip_2 = True

        active_clip_1 = gguf_clip_name if gguf_clip_1 else clip_name
        active_clip_2 = gguf_clip_name_2 if gguf_clip_2 else clip_name_2
        should_load_clip = False

        if model_type == "gguf" or model_type == "diffusion":
            if active_clip_1 in ["Embedded", "None", None]:
                raise ValueError("GGUF Mode requires external CLIP.{model_type}")
            should_load_clip = True
        elif active_clip_1 not in ["Embedded", "None", None]:
            should_load_clip = True

        if should_load_clip:
            try:
                CLIPLoaderGGUF = get_node_class("CLIPLoaderGGUF")
                DualCLIPLoaderGGUF = get_node_class("DualCLIPLoaderGGUF")

                if dual_clip:
                     if active_clip_2 in ["None", None]:
                         raise ValueError("[JonLoader] Dual CLIP requires 2 files.")
                     if gguf_clip_2:
                         clip_obj = DualCLIPLoaderGGUF().load_clip(active_clip_1, active_clip_2, type=clip_type)[0]
                     else:
                         clip_obj = nodes.DualCLIPLoader().load_clip(active_clip_1, active_clip_2, type=clip_type)[0]
                else:
                     if gguf_clip_1:
                         clip_obj = CLIPLoaderGGUF().load_clip(active_clip_1, type=clip_type)[0]
                     else:
                        l = nodes.CLIPLoader()
                        try:
                            clip_obj = l.load_clip(active_clip_1, type=clip_type)[0]
                        except:
                            clip_obj = l.load_clip(active_clip_1)[0]
            except Exception as e:
                print(f"JonLoader CLIP Error: {e}")



        # VAE
        if (model_type != "checkpoint" and vae_name in ["Embedded", "None"]) or (vae_name not in ["Embedded", "None"]):
             try:
                 vae_obj = nodes.VAELoader().load_vae(vae_name)[0]
             except:
                 pass



        # LTX2 AUDIO VAE
        if dual_vae:
            if vae_audio_name not in ["Embedded", "None", None]:
                try:
                    vae_path = folder_paths.get_full_path_or_raise("vae", vae_audio_name)
                    sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)

                    target_device = model_management.get_torch_device()
                    if vae_audio_device == "cpu":
                        target_device = torch.device("cpu")
                    target_dtype = None

                    if vae_audio_dtype == "fp16":
                        target_dtype = torch.float16
                    elif vae_audio_dtype == "bf16":
                        target_dtype = torch.bfloat16
                    elif vae_audio_dtype == "fp32":
                        target_dtype = torch.float32

                    if "vocoder.conv_post.weight" in sd:

                        if LIGHTRICKS_AVAILABLE:
                            print(f"[JonLoader] LTX2 AudioVAE Detected: {vae_audio_name}")
                            from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
                            vae_audio_obj = AudioVAE(sd, metadata)
                            if hasattr(vae_audio_obj, "first_stage_model"):
                                if target_device:
                                    vae_audio_obj.first_stage_model.to(target_device)
                                if target_dtype:
                                    vae_audio_obj.first_stage_model.to(dtype=target_dtype)
                            else:
                                print(f"[JonLoader] ")
                    else:
                         print(f"[JonLoader] Standard VAE loaded in Audio Slot: {vae_audio_name}")
                         vae_audio_obj = comfy.sd.VAE(sd=sd, device=target_device, dtype=target_dtype, metadata=metadata)
                except Exception as e:
                    print(f"[JonLoader] Audio VAE Error: {e}")
            elif vae_obj: vae_audio_obj = vae_obj

        current_model = model_obj
        current_clip = clip_obj

        #
        ## LORA's'
        #
        try:
            ui_state = json.loads(lora_stack_json)
        except:
            ui_state = {}

        if ui_state:
            lora_loader = nodes.LoraLoader()
            for slot_id, data in ui_state.items():
                if not data.get("enabled", True):
                    continue

                name = data.get("name")
                if not name or name in ["Select LoRA...", "None", "null"]:
                    continue

                input_key = data.get("input_name")
                strength = float(kwargs.get(input_key, 1.0)) if input_key else 1.0

                try:
                    (current_model, current_clip) = lora_loader.load_lora(current_model, current_clip, name, strength, strength)
                except:
                    pass
        #
        ## Sage
        #
        if sage_kernel != "disabled":
            if SAGE_AVAILABLE:
                current_model = current_model.clone()
                sage_impl = get_sage_func(sage_kernel)

                if sage_impl:
                    def sage_override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
                        in_dtype = v.dtype
                        if q.dtype == torch.float32:
                            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

                        if skip_reshape:
                            b, _, _, dim_head = q.shape
                            tensor_layout = "HND"
                        else:
                            b, _, dim_head = q.shape
                            dim_head //= heads
                            q, k, v = map(lambda t: t.view(b, -1, heads, dim_head), (q, k, v))
                            tensor_layout = "NHD"

                        if mask is not None:
                             if mask.ndim == 2:
                                 mask = mask.unsqueeze(0)
                             if mask.ndim == 3:
                                 mask = mask.unsqueeze(1)

                        out = sage_impl(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)

                        if tensor_layout == "HND":
                            if not skip_output_reshape:
                                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                        else:
                            if skip_output_reshape:
                                out = out.transpose(1, 2)
                            else:
                                out = out.reshape(b, -1, heads * dim_head)
                        return out

                    if "transformer_options" not in current_model.model_options:
                        current_model.model_options["transformer_options"] = {}

                    current_model.model_options["transformer_options"]["optimized_attention_override"] = sage_override
                    print(f"[JonSage] SageAttention Patched: {sage_kernel}")

        return (current_model,
                current_clip,
                vae_obj, vae_audio_obj)

NODE_CLASS_MAPPINGS = {"JonLoader": JonLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"JonLoader": "JonLoader"}
