import nodes
from .jon_utils import get_node_class

# comfy_extras/nodes_wan.py
# comfy_extras/nodes_model_advanced.py
# comfy_extras/nodes_images.py
# comfy_extras/nodes_video.py

class JonWan22Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "save_video": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the image after generation"
                }),

                "save_name": ("STRING", {
                    "default": "WAN_22/Jon",
                    "tooltip": "The prefix for the file to save if enabled"
                }),

                "save_last_img": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the image after generation"
                }),

                "model_high": ("MODEL", {
                    "tooltip": "Model from JonLoader"
                }),
                "model_low": ("MODEL", {
                    "tooltip": "Model from JonLoader"
                }),
                "clip": ("CLIP", {
                    "tooltip": "The primary CLIP from JonLoader"
                }),
                "vae": ("VAE", {
                    "tooltip": "The primary VAE from JonLoader"
                }),
                "start_image": ("IMAGE", {
                    "tooltip": "The 1st Image"
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
                "total_frames" : ("INT", {
                    "min": 1,
                    "max": nodes.MAX_RESOLUTION,
                    "tooltip": "The toal frames in the video"
                }),
                "fps" : ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "tooltip": "The frames per second for the video playback speed (default: 30.0)."
                }),


            },

            "optional": {
                "end_image": ("IMAGE", {
                    "tooltip": "The last Image"
                }),
                "codec": (["auto", "h264"], {
                    "default": "auto",
                    "tooltip": "codec to save to"
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
               save_video, save_name, save_last_img,
               model_high, model_low, clip, vae,
               positive, negative,
               seed,
               width, height,
               total_frames, fps,
               steps=4,
               start_image=None, end_image=None, codec="auto"):

        # Clip -> TextEncode
        p_prompt = None
        n_prompt = None
        empty_lat = None
        samp_lat = None
        image_result = None

        has_end_image = False
        if start_image is None:
            raise ValueError("[JonWan22Sampler] Starting image is not connected please fix")

        # Clip -> TextEncode
        if clip is not None:
            try:
                print(f"[JonWan22Sampler] Text Encode Wan 2.2 img2vid Detected")
                p_prompt = nodes.CLIPTextEncode().encode(clip=clip, text=positive)[0]
                n_prompt = nodes.ConditioningZeroOut().zero_out(conditioning=p_prompt)[0]
            except Exception as e:
                print(f"[JonWan22Sampler] CLIPTextEncoder Failed {e}")
                raise ValueError("[JonWan22Sampler] CLIPTextEncode Failed")

        if p_prompt is None or n_prompt is None:
            raise ValueError("[JonWan22Sampler] Text Encode Failed")
        print(f"[JonWan22Sampler] Text Encode Wan 2.2 img2vid Done")


        ## 1st or last frame
        if end_image is not None:
            has_end_image = True

        WanFirstLastFrameToVideo = get_node_class("WanFirstLastFrameToVideo")
        if not WanFirstLastFrameToVideo:
            raise ImportError("[JonWan22Sampler] WanFirstLastFrameToVideo Failed to import")

        print(f"[JonWan22Sampler] Wan 2.2 First Last Frame")
        last_frame_obj = WanFirstLastFrameToVideo.execute(
            positive=p_prompt,
            negative=n_prompt,
            vae=vae,
            width=width,
            height=height,
            length=total_frames,
            batch_size=1,
            start_image=start_image,
            end_image=end_image,
            clip_vision_start_image=None, clip_vision_end_image=None)

        # So this part I am confused about. I am
        # trying to get the outout of the positive and negitive conditioning along with the latent that WanFirstLastFrameToVideo.execute) returns
        f_pos_cond = last_frame_obj[0]
        f_neg_cond = last_frame_obj[1]
        f_lat = last_frame_obj[2]

        ModelSamplingSD3Low = get_node_class("ModelSamplingSD3")
        ModelSamplingSD3High = get_node_class("ModelSamplingSD3")
        if not ModelSamplingSD3Low:
            raise ImportError("[JonWan22Sampler] ModelSamplingSD3 Failed to import")

        low_sd3_model = ModelSamplingSD3Low().patch(model=model_low, shift=8.00, multiplier=1000)[0]
        high_sd3_model = ModelSamplingSD3High().patch(model=model_high, shift=8.00, multiplier=1000)[0]


        ksamp = nodes.KSamplerAdvanced()

        high_samp_lat_result = ksamp.sample(
            model=high_sd3_model,
            add_noise="enable",
            noise_seed=seed,
            steps=4,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=f_pos_cond,
            negative=f_neg_cond,
            latent_image=f_lat,
            start_at_step=0,
            end_at_step=2,
            return_with_leftover_noise="enable",
            denoise=1.0
        )
        hight_samp_lat = high_samp_lat_result[0]

        low_samp_lat_result = ksamp.sample(
            model=low_sd3_model,
            add_noise="disable",
            noise_seed=0,
            steps=4,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=f_pos_cond,
            negative=f_neg_cond,
            latent_image=hight_samp_lat,
            start_at_step=2,
            end_at_step=1000,
            return_with_leftover_noise="disable",
            denoise=1.0
        )
        low_samp_lat = low_samp_lat_result[0]

        print(f"[JonWan22Sampler] Wan 2.2 Sampler Done")

        # decode
        image_result = nodes.VAEDecode().decode(vae, low_samp_lat)[0]

        # make the video
        CreateVideo = get_node_class("CreateVideo")
        if not CreateVideo:
            raise ImportError("[JonWan22Sampler] CreateVideo Failed to import")
        final_video = CreateVideo.execute(images=image_result, fps=fps, audio=None)[0]

        print(f"[JonWan22Sampler] Wan 2.2 Create Video Done")
        if save_video:
            SaveVideo = get_node_class("SaveVideo")
            if not SaveVideo:
                raise ImportError("[JonWan22Sampler] SaveVideo Failed to import")

            SaveVideo.hidden = type('obj', (object,), {'prompt': positive, 'extra_pnginfo': None})
            SaveVideo.execute(video=final_video, filename_prefix=save_name, format="auto", codec=codec) # we fail here
            print(f"[JonWan22Sampler] Wan 2.2 Save Video Done")

        # check if the save of the last image if so save it
        if save_last_img:
            print(f"[JonWan22Sampler] Wan 2.2 Gathering last image")
            ibatch = nodes.ImageBatch().batch(image1=image_result, image2=image_result)[0]
            ImageFromBatch = get_node_class("ImageFromBatch")
            if not ImageFromBatch:
                raise ImportError("[JonWan22Sampler] ImageFromBatch Failed to import")

            last_img_frame = ImageFromBatch().execute(image=ibatch, batch_index=999, length=total_frames)
            nodes.SaveImage().save_images(
                images=last_img_frame[0],
                filename_prefix=save_name+"/last_frame",
                prompt=positive,
                extra_pnginfo=None
            )
            print(f"[JonWan22Sampler] Wan 2.2 last image Done")

        return ()


NODE_CLASS_MAPPINGS = {
    "JonWan22Sampler": JonWan22Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonWan22Sampler": "JonWan22Sampler"
}
