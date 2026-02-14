import nodes
from .jon_utils import get_node_class, send_status
# comfy_extras/nodes_flux.py
# comfy_extras/nodes_custom_sampler.py
# comfy_extras/nodes_edit_model.py

class JonFlux2Klein9bSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img2img": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "use image to image and not text to image"
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
                "save_image": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the image after generation"
                }),

                "save_name": ("STRING", {
                    "default": "FLUX2_9B/Jon",
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
            },
            "optional": {
                "image1": ("IMAGE", {
                    "tooltip": "The input image"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "The input image"
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
               img2img=False, image1=None, image2=None):

        ns = "JonFlux2Klein9bSampler"

        f_pos_prompt = None
        f_neg_prompt = None

        empty_lat = None

        samp_lat = None
        samp_lat1 = None

        image_result = None

        # Clip -> TextEncode
        if clip is not None:
            try:
                send_status(ns, "Text Encode Flux 2 K 9b Detected")
                p_prompt = nodes.CLIPTextEncode().encode(clip=clip, text=positive)[0]
                n_prompt = nodes.ConditioningZeroOut().zero_out(conditioning=p_prompt)[0]

                if img2img:
                    if image2 is None and image1 is None:
                        raise ValueError("[JonFlux2Klein9bSampler] image to image is enabled but there is not image connected. Please connect one")

                    ReferenceLatent = get_node_class("ReferenceLatent")
                    if not ReferenceLatent:
                        raise ImportError("[JonFlux2Klein9bSampler] ReferenceLatent Failed to import")

                    if image1 is not None:
                        send_status(ns, "using 1 images for image to image from image 1")
                        img1_lat = nodes.VAEEncode().encode(vae, image1)[0]
                        img1_pos_cond = ReferenceLatent().execute(conditioning=p_prompt, latent=img1_lat)[0]
                        img1_neg_cond = ReferenceLatent().execute(conditioning=n_prompt, latent=img1_lat)[0]
                        if image2 is None:
                            send_status(ns, "image to image only one image")
                            f_pos_prompt = img1_pos_cond
                            f_neg_prompt = img1_neg_cond
                    elif image2 is not None and image1 is not None:
                        send_status(ns, "using 2 images for img to img from image 1 and image 2")
                        img2_lat = nodes.VAEEncode().encode(vae, image2)[0]
                        img2_pos_cond = ReferenceLatent().execute(conditioning=p_prompt, latent=img1_lat)[0]
                        img2_neg_cond = ReferenceLatent().execute(conditioning=n_prompt, latent=img1_lat)[0]
                        f_pos_prompt = img2_pos_cond
                        f_neg_prompt = img2_neg_cond
                    elif image2 is not None and image1 is None:
                        send_status(ns, "using 1 images for image to image from image 2")
                        img2_lat = nodes.VAEEncode().encode(vae, image2)[0]
                        img2_pos_cond = ReferenceLatent().execute(conditioning=p_prompt, latent=img2_lat)[0]
                        img2_neg_cond = ReferenceLatent().execute(conditioning=n_prompt, latent=img2_lat)[0]
                        f_pos_prompt = img2_pos_cond
                        f_neg_prompt = img2_neg_cond
                else:
                    f_pos_prompt = p_prompt
                    f_neg_prompt = n_prompt

            except Exception as e:
                print(f"[JonFlux2Klein9bSampler] CLIPTextEncoder Failed {e}")
                raise ValueError("[JonFlux2Klein9bSampler] CLIPTextEncode Failed")
        if p_prompt is None or n_prompt is None:
            raise ValueError("[JonFlux2Klein9bSampler] Text Encode Failed")
        send_status(ns, "Text Encode Flux 2 K 9b Done")


        f_width = int(width)
        f_height = int(width)

        # Latent
        EmptyFlux2LatentImage = get_node_class("EmptyFlux2LatentImage")
        if not EmptyFlux2LatentImage:
            raise ImportError("[JonFlux2Klein9bSampler] EmptyFlux2LatentImage Failed to import")
        empty_lat = EmptyFlux2LatentImage().execute(width=f_width, height=f_height, batch_size=1)[0]

        ## Setup sampler
        Flux2Scheduler = get_node_class("Flux2Scheduler")
        if not Flux2Scheduler:
            raise ImportError("[JonFlux2Klein9bSampler] Flux2Scheduler Failed to import")
        sigmas = Flux2Scheduler().execute(4, f_width, f_height)[0]

        RandomNoise = get_node_class("RandomNoise")
        if not RandomNoise:
            raise ImportError("[JonFlux2Klein9bSampler] RandomNoise Failed to import")
        noise = RandomNoise().execute(noise_seed=seed)[0]

        CFGGuider = get_node_class("CFGGuider")
        if not CFGGuider:
            raise ImportError("[JonFlux2Klein9bSampler] CFGGuider Failed to import")
        cfg = CFGGuider().execute(model=model, positive=f_pos_prompt, negative=f_neg_prompt, cfg=1.0)[0]

        KSamplerSelect = get_node_class("KSamplerSelect")
        if not KSamplerSelect:
            raise ImportError("[JonFlux2Klein9bSampler] KSamplerSelect Failed to import")
        samp_algo = KSamplerSelect().execute(sampler_name="euler")[0]

        # KSample
        SamplerCustomAdvanced = get_node_class("SamplerCustomAdvanced")
        if not SamplerCustomAdvanced:
            raise ImportError("[JonFlux2Klein9bSampler] SamplerCustomAdvanced Failed to import")
        samp_lat = SamplerCustomAdvanced().execute(
            noise=noise,
            guider=cfg,
            sampler=samp_algo,
            sigmas=sigmas,
            latent_image=empty_lat)[0]

        send_status(ns, "KSampler Done")

        # VAE decode
        image_result = nodes.VAEDecode().decode(vae, samp_lat)

        # Save Image
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
    "JonFlux2Klein9bSampler": JonFlux2Klein9bSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonFlux2Klein9bSampler": "JonFlux2Klein9bSampler"
}
