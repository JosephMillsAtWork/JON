import torch
import torch.nn.functional as TorchFunc
import math
import os
import numpy as np
from PIL import Image, ImageOps
import folder_paths

# from.jon_utils import JONUtils as JUtils
# # import load_image_from_disk, process_image

class JonWorkflowSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        supported_models = [
            "custom "
            "z-image-turbo-txt2img", "z-image-turbo-img2img",
            "flux2-klein-9b", "flux2-klein-9b-img2img", "flux2-klein-9b-img2img"
            "qwen-2511-txt2img", "qwen-2511-img2img", "qwen-2511-imgs2img",
            "wan2.2-img2vid",
            "ltx2"
        ]

        seconds_list = list(range(1, 21))
        crop_rules = ["PreserveAspectCrop", "PreserveAspectFit", "Stretch", "Center (No Resize)"]

        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        image_files = sorted(files) if files else ["Put_Images_In_Input_Folder"]

        return {
            "required": {
                # IMAGE's Three
                "image1": (image_files, {
                    "tooltip": "Select the Start Image from the ComfyUI input folder."
                }),
                "image2": (image_files, {
                    "tooltip": "Select the Middle Image from the ComfyUI input folder."
                }),
                "image3": (image_files, {
                    "tooltip": "Select the Last Image from the ComfyUI input folder."
                }),

                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Global random seed for the workflow."
                }),

                # Geometry & Time
                "aspect_ratio": (["16:9", "4:3", "1:1", "21:9"], {
                    "default": "16:9",
                    "tooltip": "The target aspect ratio."
                }),
                "resolution": (["480p", "720p", "1080p", "1440p", "2160p"], {
                    "default": "1080p",
                    "tooltip": "The vertical resolution class. Combined with Aspect Ratio to calculate Width/Height."
                }),
                "orientation": (["Landscape", "Portrait"], {
                    "default": "Landscape",
                    "tooltip": "Swaps Width and Height if set to Portrait."
                }),

                # Prompts
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The main text prompt describing what to generate."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text describing what to exclude from the generation."
                }),


                # Image Crop Processings
                "resampling": (["bicubic", "bilinear", "nearest", "area"], {
                    "default": "bicubic",
                    "tooltip": "Algorithm used when resizing input images."
                }),
                "crop_rule": (crop_rules, {
                    "default": "PreserveAspectCrop",
                    "tooltip": "How to fit images: 'Crop' cuts edges, 'Fit' adds black bars, 'Stretch' distorts."
                }),
                "padding_color": (["black", "white"], {
                    "default": "black",
                    "tooltip": "Color of the bars added if using 'PreserveAspectFit'."
                }),


                ## VIDEO Settings
                "video_seconds": (seconds_list, {
                    "default": 9,
                    "tooltip": "Duration of the video generation in seconds."
                }),

                ## Upscale(LTX2 only ATM)
                "upscale_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flag passed to downstream nodes to enable/disable upscaling if available"
                }),
            },
            "optional": {
                "override_image1": ("IMAGE", {
                    "tooltip": "Direct image connection (Overrides the file selector for Start Image)."
                }),
                "override_image2": ("IMAGE", {
                    "tooltip": "Direct image connection (Overrides the file selector for Mid Image)."
                }),
                "override_image3": ("IMAGE", {
                    "tooltip": "Direct image connection (Overrides the file selector for End Image)."
                }),
            }
        }

    OUTPUT_TOOLTIPS = (
        "The active seed",
        "Calculated Width", "Calculated Height",
        "The Positive Prompt string", "The Negative Prompt string",
        "Total frames based on FPS", "Target FPS (fixed at 24)", "Upscale boolean flag",
        "Processed Start Image (or Error Trap if disabled)",
        "Processed Mid Image (or Error Trap if disabled)",
        "Processed End Image (or Error Trap if disabled)"
    )

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",
                    "INT", "INT", "INT",
                    "STRING", "STRING",
                    "INT", "FLOAT",
                    "BOOLEAN",)

    RETURN_NAMES = ("image1", "image2", "image3",
                    "seed", "width", "height",
                    "positive_text" , "negative_text",
                    "total_frames", "fps",
                    "upscale_enabled",)
    FUNCTION = "calculate"
    CATEGORY = "josephs_odd_nodes/utils"
    DESCRIPTION = "JonProductionNode The central command center for image and video workflows. Manages resolution, timing, prompts, and input images"

    def calculate(self,
                  seed,
                  aspect_ratio, resolution, orientation,
                  positive_prompt, negative_prompt,
                  video_seconds,
                  resampling, crop_rule, padding_color,
                  upscale_enabled,
                  image1, image2, image3,
                  override_image1=None, override_image2=None, override_image3=None):

        def load_image_from_disk(filename):
            if not filename or filename == "undefined" or filename == "Put_Images_In_Input_Folder":
                return None
            try:
                input_dir = folder_paths.get_input_directory()
                image_path = os.path.join(input_dir, filename)
                if not os.path.exists(image_path):
                    return None

                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                return image
            except Exception:
                return None

        def get_image(filename, wire):
            if wire is not None:
                return wire
            return load_image_from_disk(filename)



        # Images
        raw_img_1 = get_image(image1, override_image1)
        raw_img_2 = get_image(image2, override_image2)
        raw_img_3 = get_image(image3, override_image3)

        # Geometry
        base_height = int(resolution.split("p")[0])
        if aspect_ratio == "16:9":
            base_width = int(base_height * (16/9))
        elif aspect_ratio == "4:3":
            base_width = int(base_height * (4/3))
        elif aspect_ratio == "21:9":
            base_width = int(base_height * (21/9))
        else: base_width = base_height

        def make_divisible_by_32(val):
            return int(round(val / 32) * 32)

        final_width = make_divisible_by_32(base_width)
        final_height = make_divisible_by_32(base_height)

        if orientation == "Portrait":
            final_width, final_height = final_height, final_width

        fps_out = 24.0
        total_frames = (video_seconds * 24) + 1

        # Image Processing
        def get_blank_canvas(h, w, ref_img):
            if padding_color == "white":
                return torch.ones((1, h, w, 3), dtype=ref_img.dtype, device=ref_img.device)
            return torch.zeros((1, h, w, 3), dtype=ref_img.dtype, device=ref_img.device)

        def resize_tensor(img_tensor, width, height, mode):
            i = img_tensor.permute(0, 3, 1, 2)
            if mode in ["bicubic", "bilinear"]:
                i = TorchFunc.interpolate(i, size=(height, width), mode=mode, align_corners=False)
            else:
                i = TorchFunc.interpolate(i, size=(height, width), mode=mode)
            return i.permute(0, 2, 3, 1)

        def process_image(img):
            if img is None:
                return None
            _, cur_h, cur_w, _ = img.shape
            target_w, target_h = final_width, final_height

            if crop_rule == "Stretch":
                return resize_tensor(img, target_w, target_h, resampling)
            elif crop_rule == "PreserveAspectCrop":
                scale = max(target_w / cur_w, target_h / cur_h)
                new_w, new_h = int(cur_w * scale), int(cur_h * scale)
                img = resize_tensor(img, new_w, new_h, resampling)
                crop_x, crop_y = max(0, (new_w - target_w) // 2), max(0, (new_h - target_h) // 2)

                return img[:, crop_y:crop_y+target_h, crop_x:crop_x+target_w, :]
            elif crop_rule == "PreserveAspectFit":
                scale = min(target_w / cur_w, target_h / cur_h)
                new_w, new_h = int(cur_w * scale), int(cur_h * scale)
                img = resize_tensor(img, new_w, new_h, resampling)
                canvas = get_blank_canvas(target_h, target_w, img)
                paste_x, paste_y = (target_w - new_w) // 2, (target_h - new_h) // 2
                canvas[:, paste_y:paste_y+new_h, paste_x:paste_x+new_w, :] = img

                return canvas
            elif crop_rule == "Center (No Resize)":
                canvas = get_blank_canvas(target_h, target_w, img)
                paste_x, paste_y = (target_w - cur_w) // 2, (target_h - cur_h) // 2
                dst_x_s, dst_y_s = max(0, paste_x), max(0, paste_y)
                src_x_s, src_y_s = max(0, -paste_x), max(0, -paste_y)
                src_x_e = min(cur_w, target_w - paste_x)
                src_y_e = min(cur_h, target_h - paste_y)
                canvas[:, dst_y_s:dst_y_s+(src_y_e-src_y_s), dst_x_s:dst_x_s+(src_x_e-src_x_s), :] = \
                    img[:, src_y_s:src_y_e, src_x_s:src_x_e, :]

                return canvas
            return img

        # Process the images internally (returns None if disabled)
        out1 = process_image(raw_img_1)
        out2 = process_image(raw_img_2)
        out3 = process_image(raw_img_3)

        return (out1, out2, out3,
                seed,
                final_width, final_height,
                positive_prompt, negative_prompt,
                total_frames, fps_out,
                upscale_enabled)

NODE_CLASS_MAPPINGS = {"JonWorkflowSettings": JonWorkflowSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"JonWorkflowSettings": "JonWorkflowSettings"}
