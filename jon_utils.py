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


def get_sage_func(sage_mode):
    if sage_mode == "auto":
        from sageattention import sageattn as func
        return func
    elif sage_mode == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda as func
        return lambda q,k,v,**kw: func(q,k,v, pv_accum_dtype="fp32", **kw)
    elif sage_mode == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton as func
        return func
    elif sage_mode == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda as func
        return lambda q,k,v,**kw: func(q,k,v, pv_accum_dtype="fp32+fp32", **kw)
    elif sage_mode == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda as func
        return lambda q,k,v,**kw: func(q,k,v, pv_accum_dtype="fp32+fp16", **kw)
    elif "sageattn3" in sage_mode:
        try:
            from sageattn3 import sageattn3_blackwell
            return lambda q,k,v,**kw: sageattn3_blackwell(
                q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                per_block_mean=(sage_mode == "sageattn3_per_block_mean"), **kw
            ).transpose(1,2)
        except ImportError:
            return None
    return None




class JonUtils:
    def __init__(self):
        pass


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

    # Image Processing
    def get_blank_canvas(height, width, ref_img, padding_color="black"):
        if padding_color == "white":
            return torch.ones((1, height, width, 3), dtype=ref_img.dtype, device=ref_img.device)
        return torch.zeros((1, height, w, 3), dtype=ref_img.dtype, device=ref_img.device)

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
