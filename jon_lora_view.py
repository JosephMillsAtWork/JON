import nodes
import sys
import json

class JonLoRAChain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The base diffusion model (UNet/Transformer) to apply the LoRA stack to."
                }),
                "clip": ("CLIP", {
                    "tooltip": "The CLIP text encoder to apply the LoRA stack to."
                }),
                "lora_stack_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "display": "hidden",
                    "tooltip": "Internal State: This JSON string holds the list of loaded LoRAs and their settings. It is managed automatically by the U(client/javascript)."
                }),
            },
        }


    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "process_lora_stack"
    CATEGORY = "josephs_odd_nodes/loaders"
    OUTPUT_TOOLTIPS = (
        "The modified Model with all enabled LoRAs applied.",
        "The modified CLIP with all enabled LoRAs applied."
    )
    DESCRIPTION = "Loads and applies multiple LoRAs in sequence. Dynamic Inputs: Add as many LoRAs as you need. Toggle: Enable or disable specific LoRAs without removing them. Unified Strength: Exposes a single strength input per LoRA (applied to both Model and CLIP). Auto-Defaults: If a strength input is not connected, it defaults to 1.0"

    @classmethod
    def IS_CHANGED(s, model, clip, lora_stack_json, **kwargs):
        return lora_stack_json

    def process_lora_stack(self, model, clip, lora_stack_json, **kwargs):
        try:
            ui_state = json.loads(lora_stack_json)
        except Exception:
            ui_state = {}

        current_model = model
        current_clip = clip
        loader = nodes.LoraLoader()

        for slot_id, data in ui_state.items():
            name = data.get("name")
            strength = float(data.get("strength", 1.0))
            if not data.get("enabled", True):
                print(f"[JonLoRAChain] Skipping disabled LoRA: {name}")
                continue

            #
            input_label = data.get("input_name", f"lora_strength_{slot_id}")

            # wire
            if input_label in kwargs:
                strength = float(kwargs[input_label])
            elif f"lora_strength_{slot_id}" in kwargs:
                strength = float(kwargs[f"lora_strength_{slot_id}"])

            if not name or name == "Select LoRA..." or name == "null":
                continue

            try:
                current_model, current_clip = loader.load_lora(
                    model=current_model,
                    clip=current_clip,
                    lora_name=name,
                    strength_model=strength,
                    strength_clip=strength
                )
            except Exception as e:
                print(f"[JonLoRAChain] Error: {e}", file=sys.stderr)

        return (current_model, current_clip)

NODE_CLASS_MAPPINGS = {
    "JonLoRAChain": JonLoRAChain
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JonLoRAChain": "JonLoRAChain"
}
