import nodes

class JonChannelMixer:
    @classmethod
    def INPUT_TYPES(s):
        ch_inputs = {}
        for i in range(1, 9):
            ch_inputs[f"ch_{i}"] = ("FLOAT", {
                "default": 0.0,
                "step": 0.01,
                "min": -10000.0,
                "max": 10000.0,
                "display": "number",
                "tooltip": f"Hidden Control value for Channel {i}."
            })

        return {
            "required": {
                "channels": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "tooltip": "How many fader strips to display on the mixer board.(8 MAX)"
                }),
                "min_val": ("FLOAT", {
                    "default": 0.0,
                    "min": -1000.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "lower limit for all faders (e.g. 0.0 or -1.0)."
                }),
                "max_val": ("FLOAT", {
                    "default": 1.0,
                    "min": -1000.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "upper limit for all faders (e.g. 1.0 or 10.0)."
                }),
            },
            "optional": ch_inputs
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("ch_1", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6", "ch_7", "ch_8")
    OUTPUT_TOOLTIPS = (
        "Output value for Channel 1",
        "Output value for Channel 2",
        "Output value for Channel 3",
        "Output value for Channel 4",
        "Output value for Channel 5",
        "Output value for Channel 6",
        "Output value for Channel 7",
        "Output value for Channel 8",
    )
    FUNCTION = "get_values"
    CATEGORY = "josephs_odd_nodes/utils"
    DESCRIPTION = "A visual mixing console for controlling float values. Useful in gui for LoRA's"
    def get_values(self, channels=4, min_val=0.0, max_val=1.0, **kwargs):
        return (
            kwargs.get("ch_1", 0.0), kwargs.get("ch_2", 0.0),
            kwargs.get("ch_3", 0.0), kwargs.get("ch_4", 0.0),
            kwargs.get("ch_5", 0.0), kwargs.get("ch_6", 0.0),
            kwargs.get("ch_7", 0.0), kwargs.get("ch_8", 0.0)
        )

NODE_CLASS_MAPPINGS = {
    "JonChannelMixer": JonChannelMixer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonChannelMixer": "JonChannelMixer"
}
