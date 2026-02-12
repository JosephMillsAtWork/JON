import torch
import torch.nn.functional as TorchFunc
import math
import os
import folder_paths
import nodes


# comfy.ldm.lightricks
LIGHTRICKS_AVAILABLE = False
try:
    from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
    LIGHTRICKS_AVAILABLE = True
except ImportError:
    pass


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


