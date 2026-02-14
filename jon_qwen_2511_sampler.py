import nodes
from .jon_utils import get_node_class, send_status

class JonQwen2511Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "save_image": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the image after generation"
                }),

                "save_name": ("STRING", {
                    "default": "QWEN2511/Jon",
                    "tooltip": "The prefix for the file to save if enabled"
                }),

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
                    "tooltip": "the positive encoded text from JonWorkflowSettings"
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "tooltip": "negative encoded text from JonWorkflowSettings"
                }),
                "steps" : ("INT", {
                    "min": 0,
                    "max": 10000,
                    "default": 4,
                    "tooltip": "The number of steps for denoise"
                }),
            },

            # "hidden": {
            #     "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            # },

            "optional": {
                "image1": ("IMAGE", {
                    "tooltip": "The 1st Image"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "The 2nd Image"
                }),
                "image3": ("IMAGE", {
                    "tooltip": "The 3rd Image"
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
               positive, negative, seed, width, height, steps=4,
               image1=None, image2=None, image3=None):

        ns = "JonQwen2511Sampler"
        p_prompt = None
        n_prompt = None
        empty_lat = None
        samp_lat = None
        image_result = None

        if clip is not None:
            try:
                send_status(ns, "Text Encode Qwen Detected")

                TextEncodeQwenImageEdit = get_node_class("TextEncodeQwenImageEdit")
                TextEncodeQwenImageEditPlus = get_node_class("TextEncodeQwenImageEditPlus")

                if not TextEncodeQwenImageEdit and not TextEncodeQwenImageEditPlus:
                    raise ImportError("TextEncodeQwenImageEdit nodes not found!")

                if image3 is not None and image2 is not None and image1 is not None:
                    send_status(ns, "Text Encode Qwen 3 Image")
                    p_prompt = TextEncodeQwenImageEditPlus.execute(clip=clip,  prompt=positive, vae=vae, image1=image1, image2=image2, image3=image3)[0]
                    n_prompt = TextEncodeQwenImageEditPlus.execute(clip=clip,  prompt=negative, vae=vae, image1=image1, image2=image2, image3=image3)[0]
                elif image1 is not None:
                    send_status(ns, "Text Encode Qwen 1 Image")
                    p_prompt = TextEncodeQwenImageEdit().execute(clip=clip, prompt=positive, vae=vae, image=image1)[0]
                    n_prompt = TextEncodeQwenImageEdit().execute(clip=clip, prompt=negative, vae=vae, image=image1)[0]
                else:
                    send_status(ns, "Text Encode Qwen 0 Image")
                    p_prompt = TextEncodeQwenImageEdit().execute(clip=clip, prompt=positive, vae=None, image=None)[0]
                    n_prompt = TextEncodeQwenImageEdit().execute(clip=clip, prompt=negative, vae=None, image=None)[0]
            except Exception as e:
                print(f"[JonQwen2511Sampler] CLIPTextEncoder Failed {e}")
                raise ValueError("[JonQwen2511Sampler] CLIPTextEncode Failed")


        if p_prompt is None or n_prompt is None:
            raise ValueError("[JonQwen2511Sampler] Text Encode Failed")


        send_status(ns, "Text Encode Qwen Done")

        ## Empty LAT
        empty_lat_result = nodes.EmptyLatentImage().generate(width, height, 1)
        empty_lat = empty_lat_result[0]

        #KSampler
        ksamp = nodes.KSamplerAdvanced()
        samp_lat_result = ksamp.sample(
            model=model,
            add_noise="enable",
            noise_seed=seed,
            steps=steps,
            cfg=1.0,
            sampler_name="uni_pc",
            scheduler="normal",
            positive=p_prompt,
            negative=n_prompt,
            latent_image=empty_lat,
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise="disable",
            denoise=1.0
        )
        samp_lat = samp_lat_result[0]
        send_status(ns, "KSampler Qwen Done")

        image_result = nodes.VAEDecode().decode(vae, samp_lat)

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

        return {"ui": ui_results}

NODE_CLASS_MAPPINGS = {
    "JonQwen2511Sampler": JonQwen2511Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonQwen2511Sampler": "JonQwen2511Sampler"
}
