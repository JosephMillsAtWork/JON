import nodes
import comfy.utils
import hashlib

from .jon_utils import get_node_class, send_status

class JonZImageSampler:
    def __init__(self):
        self.cache = {
            "text_encode": {
                "hash": None,
                "result": None # (p_prompt, n_prompt)
            }
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Model from JonLoader"
                }),
                "clip": ("CLIP", {
                    "tooltip": "The primary CLIP from JonLoader"
                }),
                "vae": ("VAE", {
                    "tooltip": "The primary VAE from JonLoader"
                }),
                # Simple types
                "img2img": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "use image to image and not text to image"
                }),

                "save_image": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the image after generation"
                }),

                "save_name": ("STRING", {
                    "default": "ZImage/Jon",
                    "tooltip": "The prefix for the file to save if enabled"
                }),
                "seed": ("INT", {
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "The Seed from JonWorkflowSettings"
                }),
                "width": ("INT", {
                    "tooltip": "The width from JonWorkflowSettings"
                }),
                "height": ("INT", {
                    "tooltip": "The height from JonWorkflowSettings"
                }),
                "positive": ("STRING", {
                    "multiline": True,
                    "tooltip": "The positive encoded text from JonWorkflowSettings"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "The input image"
                }),
                # Simple Types
                "denoise_1": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."
                }),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_TOOLTIPS = ()
    OUTPUT_NODE = True

    FUNCTION = "sample"
    CATEGORY = "josephs_odd_nodes/samplers"

    def sample(self,
               save_image, save_name,
               model, clip, vae,
               positive, seed, width, height,
               denoise=1.0,
               img2img=False, denoise_1=1.0, image=None):

        total_steps = 6
        if img2img:
            total_steps = total_steps + 4

        total_steps = total_steps + 5

        pbar = comfy.utils.ProgressBar(total_steps)
        ns = "JonZImageSampler"

        p_prompt = None
        n_prompt = None

        empty_lat = None

        samp_lat = None
        samp_lat1 = None

        image_result = None

        # do_text_encode()
        # Clip -> TextEncode
        current_text_hash = hashlib.sha256(f"{id(clip)}_{positive}".encode()).hexdigest()

        if clip is not None:
            if (self.cache["text_encode"]["hash"] == current_text_hash and self.cache["text_encode"]["result"] is not None):
                send_status(ns, "Using Cached Text Encoding")
                p_prompt, n_prompt = self.cache["text_encode"]["result"]
                pbar.update(2)
            else:
                try:
                    send_status(ns, "No Cache Text Encode ZImage")
                    p_prompt = nodes.CLIPTextEncode().encode(clip=clip, text=positive)[0]
                    pbar.update(1)
                    n_prompt = nodes.ConditioningZeroOut().zero_out(conditioning=p_prompt)[0]
                    pbar.update(2)
                    self.cache["text_encode"]["hash"] = current_text_hash
                    self.cache["text_encode"]["result"] = (p_prompt, n_prompt)
                except Exception as e:
                    print(f"[JonZImageSampler] CLIPTextEncoder Failed {e}")
                    raise ValueError("[JonZImageSampler] CLIPTextEncode Failed")

        if p_prompt is None or n_prompt is None:
            raise ValueError("[JonZImageSampler] Text Encode Failed")
        send_status(ns, "Text Encode ZImage Done")
        pbar.update(3)

        # do_latent()
        # Latent
        # current_latent_hash = hashlib.sha256(f"{id(image)}_{img2img}_{width}_{height}".encode()).hexdigest()
        if img2img and image is not None:
            send_status(ns, "img2img Detected")
            empty_lat = nodes.VAEEncode().encode(vae, image)[0]
        elif img2img and image is None:
            raise ValueError("Jon ZImage Sampler Image to Image is set but the image is not set or bad")
        else:
            empty_lat = nodes.EmptyLatentImage().generate(width, height, 1)[0]
        pbar.update(4)



        # do_sample()
        # KSampler
        mcfg = 1.0
        if img2img:
            mcfg = 5.0

        # current_sampler_hash = hashlib.sha256(f"{id(model)}_{seed}_{mcfg}_{denoise}".encode()).hexdigest()
        samp_lat_res = nodes.KSampler().sample(
            model=model,
            seed=seed,
            steps=6,
            cfg=mcfg,
            sampler_name="res_multistep",
            scheduler="simple",
            positive=p_prompt,
            negative=n_prompt,
            latent_image=empty_lat,
            denoise=denoise
        )
        samp_lat = samp_lat_res[0]
        pbar.update(10)

        if img2img:
            samp_lat_res_1 = nodes.KSampler().sample(
                model=model,
                seed=seed,
                steps=4,
                cfg=1.0,
                sampler_name="res_multistep",
                scheduler="simple",
                positive=p_prompt,
                negative=n_prompt,
                latent_image=samp_lat,
                denoise=denoise_1
            )
            samp_lat_1 = samp_lat_res_1[0]
            image_result = nodes.VAEDecode().decode(vae, samp_lat_1)
            pbar.update(14)
        else:
            image_result = nodes.VAEDecode().decode(vae, samp_lat)

        send_status(ns, "KSampler Done")

        # current_save_hash = hashlib.sha256(f"{id(model)}_{seed}_{mcfg}_{denoise}".encode()).hexdigest()
        # do_save()
        ui_results = {}
        if save_image:
            full_output = nodes.SaveImage().save_images(
                images=image_result[0],
                filename_prefix=save_name,
                prompt=positive,
                extra_pnginfo=None
            )
            ui_results = full_output.get("ui", {})
        else:
            preview_output = nodes.PreviewImage().save_images(
                images=image_result[0],
                filename_prefix="JonPreview",
            )
            ui_results = preview_output.get("ui", {})

        pbar.update(total_steps)
        return {"ui": ui_results}


NODE_CLASS_MAPPINGS = {
    "JonZImageSampler": JonZImageSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonZImageSampler": "JonZImageSampler"
}
