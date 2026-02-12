class JonDebugNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "input_data": ("*",),
            },
            "optional": {
                "prefix": ("STRING", {"default": "DEBUG: "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "debug_print"
    OUTPUT_NODE = True
    CATEGORY = "josephs_odd_nodes/utils"

    def debug_print(self, input_data, prefix):
        if hasattr(input_data, "shape"):
            text_value = f"Tensor Shape: {list(input_data.shape)} | Type: {input_data.dtype}"
        elif isinstance(input_data, list):
            text_value = f"List with {len(input_data)} items: {input_data}"
        else:
            text_value = str(input_data)
        message = f"{prefix}{text_value}"
        print(f"\n{message}\n")
        return {"ui": {"text": (message,)}, "result": (message,)}

NODE_CLASS_MAPPINGS = {
    "JonDebugNode": JonDebugNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonDebugNode": "JonDebugger"
}
