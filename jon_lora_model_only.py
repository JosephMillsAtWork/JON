import nodes
import sys
import json

class JonLoRAModelOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",{
                    "tooltip": "The base diffusion model (UNet/Transformer) to apply the LoRA stack to."
                }),
                "lora_stack_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "display": "hidden",
                    "tooltip": "Internal State: This JSON string holds the list of loaded LoRAs and their settings. It is managed automatically by the U(client/javascript)."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "process_lora_stack"
    CATEGORY = "josephs_odd_nodes/loaders"
    OUTPUT_TOOLTIPS = (
        "The modified Model with all enabled LoRAs applied."
    )
    DESCRIPTION = "Loads and applies multiple LoRAs (no CLIP) in sequence"

    @classmethod
    def IS_CHANGED(s, model, lora_stack_json, **kwargs):
        return lora_stack_json

    def process_lora_stack(self, model, lora_stack_json, **kwargs):
        # Parse JSON
        try:
            ui_state = json.loads(lora_stack_json)
        except Exception:
            ui_state = {}

        current_model = model
        loader = nodes.LoraLoaderModelOnly()

        # iterate
        for slot_id, data in ui_state.items():
            name = data.get("name")
            strength = float(data.get("strength", 1.0))

            # If enabled is missing (old workflows), default to True.
            if not data.get("enabled", True):
                continue

            # dynamic input
            input_label = data.get("input_name", f"lora_strength_{slot_id}")

            # wire
            if input_label in kwargs:
                strength = float(kwargs[input_label])
            elif f"lora_strength_{slot_id}" in kwargs:
                strength = float(kwargs[f"lora_strength_{slot_id}"])

            if not name or name == "Select LoRA..." or name == "null":
                continue

            try:
                (current_model,) = loader.load_lora_model_only(
                    model=current_model,
                    lora_name=name,
                    strength_model=strength
                )
            except Exception as e:
                print(f"[JonLoRAModelOnly] {e}", file=sys.stderr)

        return (current_model,)

NODE_CLASS_MAPPINGS = {
    "JonLoRAModelOnly": JonLoRAModelOnly
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JonLoRAModelOnly": "JonLoRAChain(Model Only)"
}
