import torch

class JonMathConversion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.01,
                    "tooltip": "The input float"
                }),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    FUNCTION = "convert"
    CATEGORY = "josephs_odd_nodes/utils"

    def convert(self, value):
        return (int(value), float(value), str(value))

class JonFrameRate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "get_fps"
    CATEGORY = "josephs_odd_nodes/utils"

    def get_fps(self, fps):
        return (float(fps), int(fps))

class JonSimpleMath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "b": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "op": (["Add", "Subtract", "Multiply", "Divide", "Power", "Modulus"],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "op"
    CATEGORY = "josephs_odd_nodes/utils"

    def op(self, a, b, op):
        res = 0.0
        if op == "Add": res = a + b
        elif op == "Subtract": res = a - b
        elif op == "Multiply": res = a * b
        elif op == "Divide": res = a / b if b != 0 else 0.0
        elif op == "Power": res = pow(a, b)
        elif op == "Modulus": res = a % b if b != 0 else 0.0
        return (res, int(res))

NODE_CLASS_MAPPINGS = {
    "JonMathConversion": JonMathConversion,
    "JonSimpleMath": JonSimpleMath,
    "JonFrameRate": JonFrameRate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonMathConversion": "Any -> Int/Float/Str",
    "JonSimpleMath": "Simple Math",
    "JonFrameRate": "Frame Rate"
}
