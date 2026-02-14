import nodes
import folder_paths
from .jon_utils import get_node_class, send_status

# comfy_extras/nodes_lt.py
# comfy_extras/nodes_dataset.py
# comfy_extras/nodes_lt_audio.py
# comfy_extras/nodes_lt_upsampler.py:

# comfy_extras.nodes_dataset -> ResizeImagesByLongerEdgeNode

# comfy_extras/nodes_custom_sampler.py
# comfy_extras/nodes_hunyuan.py

# comfy_extras/nodes_images.py
# comfy_extras/nodes_video.py


class JonLTX2Sampler:
    def __init__(self):
        pass


    @classmethod
    def INPUT_TYPES(s):
        upscale_list = folder_paths.get_filename_list("latent_upscale_models")
        return {
            "required": {
                "img2vid": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "generate a image to video. False generates a Text to Video"
                }),

                "save_video": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the image after generation"
                }),
                "save_name": ("STRING", {
                    "default": "LTX2/Jon",
                    "tooltip": "The prefix for the file to save if enabled"
                }),
                "save_last_img": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the image after generation"
                }),

                "codec": (["auto", "h264"], {
                    "default": "auto",
                    "tooltip": "codec to save to"
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
                "audio_vae": ("VAE", {
                    "tooltip": "The audio VAE from JonLoader"
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
                "upscale_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enabling this required the upscale model and also runs two samplers"
                }),

            },

            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "The 1st Image"
                }),

                "upscale_model": (upscale_list, {
                    "tooltip" : "LTX2 Upscale model"
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
               img2vid,
               save_video, save_name, save_last_img,
               model, clip, vae, audio_vae,
               seed,
               width, height,
               positive, negative,
               total_frames, fps,
               upscale_enabled, upscale_model,
               image=None, codec="auto"):

        ns = "JonLTX2Sampler"
        p_prompt = None
        pos_prompt = None

        n_prompt = None
        neg_prompt = None

        empty_lat = None
        samp_lat = None
        image_result = None
        first_sigmas = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
        second_sigmas = "0.909375, 0.725, 0.421875, 0.0"

        real_image = None
        if not img2vid or img2vid and image is None:
            real_image = nodes.EmptyImage().generate(width=width, height=height, batch_size=1, color=0)[0]
        else:
            real_image = image

        # Clip -> TextEncode
        if clip is not None:
            try:
                send_status(ns, "Text Encode LTX2 Detected")
                pos_prompt = nodes.CLIPTextEncode().encode(clip=clip, text=positive)[0]
                neg_prompt = nodes.CLIPTextEncode().encode(clip=clip, text=negative)[0]
            except Exception as e:
                print(f"[JonLTX2Sampler] CLIPTextEncoder Failed {e}")
                raise ValueError("[JonLTX2Sampler] CLIPTextEncode Failed")

        LTXVConditioning = get_node_class("LTXVConditioning")
        if not LTXVConditioning:
            raise ImportError("[JonLTX2Sampler] LTXVConditioning Failed to import")
        prompt_obj = LTXVConditioning().execute(positive=pos_prompt, negative=neg_prompt, frame_rate=fps)
        p_prompt = prompt_obj[0]
        n_prompt = prompt_obj[1]
        if p_prompt is None or n_prompt is None:
            raise ValueError("[JonLTX2Sampler] Text Encode Failed")

        send_status(ns, "CLIPTextEncoder LTX2 Done")

        # process imcoming image.
        ResizeImagesByLongerEdgeNode = get_node_class("ResizeImagesByLongerEdge")
        if not ResizeImagesByLongerEdgeNode:
            raise ImportError("[JonLTX2Sampler] ResizeImagesByLongerEdgeNode Failed to import")
        res_img_obj = ResizeImagesByLongerEdgeNode()._process(image=real_image, longer_edge=1536)
        res_img = res_img_obj[0]

        LTXVPreprocess = get_node_class("LTXVPreprocess")
        if not LTXVPreprocess:
            raise ImportError("[JonLTX2Sampler] LTXVPreprocess Failed to import")
        pre_img_obj = LTXVPreprocess().execute(image=res_img_obj, img_compression=33)
        pre_img = pre_img_obj[0]
        pre_img_height = pre_img.shape[1]
        pre_img_width = pre_img.shape[2]

        send_status(ns, "pre_img widht{pre_img_width} height {pre_img_height}")

        new_pre_img = nodes.EmptyImage().generate(width=pre_img_width, height=pre_img_height, batch_size=1, color=0)[0]
        upscale_pre_img = nodes.ImageScaleBy().upscale(image=new_pre_img, upscale_method="lanczos", scale_by=0.50)[0]
        upscale_pre_img_height = upscale_pre_img.shape[1]
        upscale_pre_img_width = upscale_pre_img.shape[2]

        EmptyLTXVLatentVideo = get_node_class("EmptyLTXVLatentVideo")
        if not EmptyLTXVLatentVideo:
            raise ImportError("[JonLTX2Sampler]  EmptyLTXVLatentVideo to import")
        empty_vid_lat = EmptyLTXVLatentVideo.execute(width=upscale_pre_img_width, height=upscale_pre_img_height, length=total_frames, batch_size=1)[0]

        LTXVEmptyLatentAudio =  get_node_class("LTXVEmptyLatentAudio")
        if not LTXVEmptyLatentAudio:
            raise ImportError("[JonLTX2Sampler]  LTXVEmptyLatentAudio to import")
        empty_audio_lat = LTXVEmptyLatentAudio.execute(
            frames_number=int(total_frames),
            frame_rate=int(fps),
            batch_size=1,
            audio_vae=audio_vae,
        )[0]

        LTXVImgToVideoInplace = get_node_class("LTXVImgToVideoInplace")
        if not LTXVImgToVideoInplace:
            raise ImportError("[JonLTX2Sampler]  LTXVImgToVideoInplace to import")
        vid_lat = LTXVImgToVideoInplace.execute(vae=vae, image=pre_img, latent=empty_vid_lat, strength=1.0, bypass=False)[0]


        def jon_ltx2_sampler(seed, pos_prompt, neg_prompt, vid_lat, audio_lat, sigmas, algo):
            RandomNoise = get_node_class("RandomNoise")
            if not RandomNoise:
                raise ImportError("[JonLTX2Sampler] RandomNoise Failed to import")
            noise_seed = RandomNoise().execute(noise_seed=seed)[0]

            CFGGuider = get_node_class("CFGGuider")
            if not CFGGuider:
                raise ImportError("[JonLTX2Sampler] CFGGuider Failed to import")
            cfg = CFGGuider().execute(model=model, positive=pos_prompt, negative=neg_prompt, cfg=1.0)[0]

            KSamplerSelect = get_node_class("KSamplerSelect")
            if not KSamplerSelect:
                raise ImportError("[JonLTX2Sampler] KSamplerSelect Failed to import")
            samp_algo = KSamplerSelect().execute(sampler_name=algo)[0]

            ManualSigmas = get_node_class("ManualSigmas")
            if not ManualSigmas:
                raise ImportError("[JonLTX2Sampler] ManualSigmas Failed to import")
            sigma = ManualSigmas().execute(sigmas=sigmas)[0]

            LTXVConcatAVLatent = get_node_class("LTXVConcatAVLatent")
            if not LTXVConcatAVLatent:
                raise ImportError("[JonLTX2Sampler] LTXVConcatAVLatent Failed to import")
            concat_lat = LTXVConcatAVLatent().execute(video_latent=vid_lat, audio_latent=audio_lat)[0]

            # KSample
            SamplerCustomAdvanced = get_node_class("SamplerCustomAdvanced")
            if not SamplerCustomAdvanced:
                raise ImportError("[JonLTX2Sampler] SamplerCustomAdvanced Failed to import")
            samp_lat = SamplerCustomAdvanced().execute(noise=noise_seed, guider=cfg, sampler=samp_algo, sigmas=sigma, latent_image=concat_lat)[1]
            return samp_lat


        denoise_pass_one = jon_ltx2_sampler(seed, p_prompt, n_prompt, vid_lat, empty_audio_lat, first_sigmas, "euler")

        LTXVSeparateAVLatent = get_node_class("LTXVSeparateAVLatent")
        if not LTXVSeparateAVLatent:
            raise ImportError("[JonLTX2Sampler] LTXVSeparateAVLatent Failed to import")
        # vidio_av_latent = av_latent_obj[0]
        # audio_av_latent = av_latent_obj[1]

        if upscale_enabled:
            LatentUpscaleModelLoader = get_node_class("LatentUpscaleModelLoader")
            if not LatentUpscaleModelLoader:
                raise ImportError("[JonLTX2Sampler] LatentUpscaleModelLoader Failed to import")
            latent_upscale_model = LatentUpscaleModelLoader().execute(model_name=upscale_model)[0]

            av_latent_obj = LTXVSeparateAVLatent().execute(av_latent=denoise_pass_one)
            vidio_av_latent = av_latent_obj[0]
            audio_av_latent = av_latent_obj[1]


            LTXVCropGuides = get_node_class("LTXVCropGuides")
            if not LTXVCropGuides:
                raise ImportError("[JonLTX2Sampler] LTXVCropGuides Failed to import")
            crop_guide_obj = LTXVCropGuides().execute(positive=p_prompt, negative=n_prompt, latent=vidio_av_latent)
            crop_guide_pos = crop_guide_obj[0]
            crop_guide_neg = crop_guide_obj[1]
            crop_guide_lat = crop_guide_obj[2]

            LTXVLatentUpsampler = get_node_class("LTXVLatentUpsampler")
            if not LTXVLatentUpsampler:
                raise ImportError("[JonLTX2Sampler] LTXVLatentUpsampler Failed to import")
            spatial_lat = LTXVLatentUpsampler().upsample_latent(samples=crop_guide_lat, upscale_model=latent_upscale_model, vae=vae)[0]

            LTXVImgToVideoInplace =  get_node_class("LTXVImgToVideoInplace")
            if not LTXVImgToVideoInplace:
                raise ImportError("[JonLTX2Sampler] LTXVImgToVideoInplace Failed to import")
            spatial_upscale_lat = LTXVImgToVideoInplace().execute(vae=vae, image=pre_img, latent=spatial_lat, strength=1.0, bypass=False)[0]

            denoise_pass_two = jon_ltx2_sampler(0, crop_guide_pos, crop_guide_neg, spatial_upscale_lat, audio_av_latent, second_sigmas, "gradient_estimation")
            av_latent_obj = LTXVSeparateAVLatent().execute(av_latent=denoise_pass_two)
            vidio_av_latent = av_latent_obj[0]
            audio_av_latent = av_latent_obj[1]
        else:
            av_latent_obj = LTXVSeparateAVLatent().execute(av_latent=denoise_pass_one)
            vidio_av_latent = av_latent_obj[0]
            audio_av_latent = av_latent_obj[1]



        video_dec = nodes.VAEDecode().decode(vae, vidio_av_latent)[0]
        LTXVAudioVAEDecode =  get_node_class("LTXVAudioVAEDecode")
        if not LTXVAudioVAEDecode:
            raise ImportError("[JonLTX2Sampler] LTXVAudioVAEDecode Failed to import")
        audio_dec = LTXVAudioVAEDecode.execute(samples=audio_av_latent, audio_vae=audio_vae)[0]

        # make the video
        CreateVideo = get_node_class("CreateVideo")
        if not CreateVideo:
            raise ImportError("[JonLTX2Sampler] CreateVideo Failed to import")
        final_video = CreateVideo.execute(images=video_dec, fps=fps, audio=audio_dec)[0]

        send_status(ns, "LTX2 Create Video Done")

        ui_results = {}
        if save_video:
            SaveVideo = get_node_class("SaveVideo")
            if not SaveVideo:
                raise ImportError("[JonLTX2Sampler] SaveVideo Failed to import")
            SaveVideo.hidden = type('obj', (object,), {'prompt': positive, 'extra_pnginfo': None})

            video_out = SaveVideo.execute(
                video=final_video,
                filename_prefix=save_name,
                format="auto", codec=codec)

            ui_results.update(video_out.ui.as_dict())
            send_status(ns, "LTX2 Save Video Done")

        # check if the save of the last image if so save it
        if save_last_img:
            send_status(ns, "LTX2 Gathering last image")
            last_frame = video_dec[-1:]
            img_out = nodes.SaveImage().save_images(
                images=last_frame,
                filename_prefix=save_name+"/last_frame",
                prompt=positive,
                extra_pnginfo=None
            )
            if not ui_results:
                ui_results.update(img_out.get("ui", {}))
            send_status(ns, "LTX2 last image Done")

        if not ui_results:
            first_frame = video_dec[0:1]
            preview_out = nodes.PreviewImage().save_images(
                images=first_frame,
                filename_prefix="JonPreview"
            )
            ui_results.update(preview_out.get("ui", {}))

        return {"ui": ui_results, "result": ()}

NODE_CLASS_MAPPINGS = {
    "JonLTX2Sampler": JonLTX2Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JonLTX2Sampler": "JonLTX2Sampler"
}
