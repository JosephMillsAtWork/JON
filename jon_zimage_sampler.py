import nodes
from .jon_utils import get_node_class

class JonZImageSampler:
    def __init__(self):
        pass

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
                "img2img": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "use image to image and not text to image"
                }),
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("The Generated Image",)
    OUTPUT_NODE = True

    FUNCTION = "sample"
    CATEGORY = "josephs_odd_nodes/samplers"

    def sample(self,
               save_image, save_name,
               model, clip, vae,
               positive, seed, width, height,
               denoise=1.0,
               img2img=False, denoise_1=1.0, image=None):

        p_prompt = None
        n_prompt = None

        empty_lat = None

        samp_lat = None
        samp_lat1 = None

        image_result = None

        # Clip -> TextEncode
        if clip is not None:
            try:
                print(f"[JonZImageSampler] Text Encode ZImage Detected")
                p_prompt = nodes.CLIPTextEncode().encode(clip=clip, text=positive)[0]
                n_prompt = nodes.ConditioningZeroOut().zero_out(conditioning=p_prompt)[0]
            except Exception as e:
                print(f"[JonZImageSampler] CLIPTextEncoder Failed {e}")
                raise ValueError("[JonZImageSampler] CLIPTextEncode Failed")

        if p_prompt is None or n_prompt is None:
            raise ValueError("[JonZImageSampler] Text Encode Failed")
        print(f"[JonZImageSampler] Text Encode ZImage Done")

        # Latent
        if img2img and image is not None:
            print(f"[JonZImageSampler] img2img Detected")
            empty_lat = nodes.VAEEncode().encode(vae, image)[0]
        elif img2img and image is None:
            raise ValueError("Jon ZImage Sampler Image to Image is set but the image is not set or bad")
        else:
            empty_lat = nodes.EmptyLatentImage().generate(width, height, 1)[0]

        # KSampler
        mcfg = 1.0
        if img2img:
            mcfg = 5.0

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
        samp_lat = samp_lat_result[0]

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
        else:
            image_result = nodes.VAEDecode().decode(vae, samp_lat)

        print(f"[JonZImageSampler] KSampler Done")

        # Save Image
        if save_image:
            # def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
            nodes.SaveImage().save_images(
                images=image_result[0],
                filename_prefix=save_name,
                prompt=positive,
                extra_pnginfo=None
            )

        return image_result


NODE_CLASS_MAPPINGS = {
    "JonZImageSampler": JonZImageSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonZImageSampler": "JonZImageSampler"
}
