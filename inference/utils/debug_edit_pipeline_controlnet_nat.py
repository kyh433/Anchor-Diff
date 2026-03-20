import pdb, sys

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "inference/utils")
from base_pipeline import BasePipeline
from diffusers import StableDiffusionControlNetPipeline
from diffusers.models import ControlNetModel
from cross_attention import prep_unet
from diffusers.utils.torch_utils import is_compiled_module
import PIL
from PIL import Image
import cv2
from diffusers.utils import load_image
# ------------
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, MultiControlNetModel, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

import copy


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
def auto_canny(image, sigma=0.33):
    # 计算图像中像素灰度的中位数
    v = np.median(image)

    # 根据中值自动计算阈值
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(image, lower, upper)

class EditingPipeline_control:
    def __init__(self, model1: BasePipeline, model2: StableDiffusionControlNetPipeline):
        
        self.model1 = model1
        self.model2 = model2


        # 自动从 model2 中提取子模块（已加载的组件）
        vae = model2.vae
        text_encoder = model2.text_encoder
        tokenizer = model2.tokenizer
        unet = model2.unet
        controlnet = model2.controlnet
        scheduler = model2.scheduler
        safety_checker = model2.safety_checker
        feature_extractor = model2.feature_extractor
        image_encoder = getattr(model2, "image_encoder", None)

        self.model2.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.model2.vae_scale_factor = 2 ** (len(self.model2.vae.config.block_out_channels) - 1) if getattr(self.model2, "vae", None) else 8
        self.model2.image_processor = VaeImageProcessor(vae_scale_factor=self.model2.vae_scale_factor, do_convert_rgb=True)
        self.model2.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.model2.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
    def _preprocess_control_image(self, image, height, width):
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height))
            image = torch.tensor(np.array(image)).float() / 255.0  # normalize
            image = image.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # to (1, 3, H, W)
        return image.to(dtype=self.model2.unet.dtype, device=self.model2.device)
    
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.model2.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

        # pix2pix parameters
        guidance_amount=0.1,
        edit_dir=None,
        x_in=None,
        only_sample=False, # only perform sampling, and no editing

    ):
        model1 = copy.copy(self.model1)

        # 0. modify the unet to be useful :D
        model1.unet = prep_unet(model1.unet)
        self.model2.unet = prep_unet(self.model2.unet)
        
        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        
        # 2. Default height and width to unet
        height = height or model1.unet.config.sample_size * model1.vae_scale_factor
        width = width or model1.unet.config.sample_size * model1.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = model1._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        x_in = x_in.to(dtype=model1.unet.dtype, device=device)
        # 3. Encode input prompt = 2x77x1024
        prompt_embeds,negative_prompt_embeds = model1.encode_prompt( 
            prompt, 
            device, 
            num_images_per_prompt,
            do_classifier_free_guidance, 
            negative_prompt, 
            prompt_embeds=None, 
            negative_prompt_embeds=None)
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     model1.scheduler, num_inference_steps, device, model1.scheduler.timesteps)

        model1.scheduler.set_timesteps(num_inference_steps, device=device)
        self.model2.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = model1.scheduler.timesteps
        timesteps2 = self.model2.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = model1.unet.config.in_channels
        
        # randomly sample a latent code if not provided
        # generator=torch.manual_seed(0)

        latents = model1.prepare_latents(batch_size * num_images_per_prompt,
                                        num_channels_latents,
                                        height,
                                        width,
                                        prompt_embeds.dtype,
                                        device,
                                        generator,
                                        latents,
        )
        
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = model1.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * model1.scheduler.order
        with torch.no_grad():
            with model1.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = model1.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = model1.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                    # add the cross attention map to the dictionary
                    d_ref_t2attn[t.item()] = {}
                    for name, module in model1.unet.named_modules():
                        module_name = type(module).__name__
                        # if module_name == "CrossAttention" and 'attn2' in name:
                        if module_name == "Attention" and 'attn2' in name:
                            attn_mask = module.attn_probs # size is num_channel,s*s,77
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = model1.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model1.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        with torch.no_grad():
            image_rec = model1.numpy_to_pil(model1.decode_latents(latents))

        if only_sample:
            return image_rec

        # del model1.unet
        # del model1.vae  # 如果用到了vae
        # del model1.text_encoder
        # del model1.tokenizer
        # del model1.scheduler
        del model1
        torch.cuda.empty_cache()

        # 把model2移到GPU
        self.model2.unet.to(device)
        self.model2.vae.to(device)
        
        # prompt_embeds_edit = prompt_embeds.clone()
        #add the edit only to the second prompt, idx 0 is the negative prompt
        # prompt_embeds_edit[1:2] += edit_dir
        # prompt_edit='The image appears to be a satellite image of a city or town, with various buildings, roads, and other structures visible'
        # prompt_edit="The image appears to be a satellite image of a mountainous region, likely captured by a high-resolution satellite such as Landsat or Sentinel-2"
        prompt_edit=prompt
        # prompt_embeds_edit = self.model1._encode_prompt( prompt_edit, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=None, negative_prompt_embeds=negative_prompt_embeds,)

        prompt_embeds, negative_prompt_embeds = self.model2.encode_prompt(
            prompt_edit,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds_edit = torch.cat([negative_prompt_embeds, prompt_embeds])

        # generator=torch.manual_seed(0)
        # latents = latents_init.half()
        latents = self.model2.prepare_latents(batch_size * num_images_per_prompt,
                                        num_channels_latents,
                                        height,
                                        width,
                                        prompt_embeds.dtype,
                                        device,
                                        generator,
        )
        # 使用 Canny 算法提取边缘
        # edges_np = auto_canny(np.array(image_rec[0].convert("L")), sigma=0.33)
        # cv2.imwrite("./rec_line.jpg", edges_np)
        # condition_image = load_image("/mnt/data/project_kyh/MultimodalityGeneration25/alpha_output/nat2ir_controlnet_batch/natural_5215_reconstruction.png") 
        # condition_image = load_image("/mnt/data/project_kyh/MultimodalityGeneration25/alpha_output/debug_nat2ir/all_infrared_11624_reconstruction.png") 
        condition_image=image_rec

        # 4. Prepare image
        if isinstance(self.model2.controlnet, ControlNetModel):
            control_image = self.prepare_image(
                image=condition_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.model2.controlnet.dtype,
                do_classifier_free_guidance=True,
                guess_mode=False,
            )
#####################
        # from torchvision.transforms.functional import to_pil_image
        # import os

        # # 取第一张图
        # img = control_image[0].detach().cpu()
        # if img.min() < 0:
        #     img = (img + 1) / 2
        # img = img.clamp(0, 1)  # 确保范围在[0,1]

        # # 转成 PIL 图像
        # pil_img = to_pil_image(img)

        # # 保存
        # save_path = os.path.join("./alpha_output/debug_nat2ir/control_image_0.png")
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # pil_img.save(save_path)
################
        controlnet_conditioning_scale=1.0
        control_guidance_start=0.0
        control_guidance_end=1.0

        controlnet = self.model2.controlnet._orig_mod if is_compiled_module(self.model2.controlnet) else self.model2.controlnet
        controlnet_keep = []
        for i in range(len(timesteps2)):
            keeps = [
                1.0 - float(i / len(timesteps2) < s or (i + 1) / len(timesteps2) > e)
                for s, e in zip([control_guidance_start], [control_guidance_end])
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        
        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            controlnet_cond_scale = controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep[i]

        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.model2.scheduler.order
        with self.model2.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps2):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model2.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                # predict the noise residual

                # from torch.amp import autocast 
                # with autocast("cuda"):
                
                # 7.1 ControlNet 前向计算
                controlnet_output = self.model2.controlnet(
                    x_in,
                    t,
                    encoder_hidden_states=prompt_embeds_edit.detach(),
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    return_dict=True,)
                # 7.2 UNet 前向计算（加上 ControlNet 残差）
                noise_pred = self.model2.unet(
                    x_in,
                    t,
                    encoder_hidden_states=prompt_embeds_edit.detach(),
                    down_block_additional_residuals=controlnet_output.down_block_res_samples,
                    mid_block_additional_residual=controlnet_output.mid_block_res_sample,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=True)[0]

                # noise_pred = self.model2.unet(x_in,t,encoder_hidden_states=prompt_embeds_edit.detach(),cross_attention_kwargs=cross_attention_kwargs,).sample

                # loss = 0.0
                loss = torch.tensor(0.0, device=device) 
                for name, module in self.model2.unet.named_modules():
                    module_name = type(module).__name__
                    # if module_name == "CrossAttention" and 'attn2' in name:
                    if module_name == "Attention" and 'attn2' in name:
                        curr = module.attn_probs # size is num_channel,s*s,77
                        ref = d_ref_t2attn[t.item()][name].detach().to(device)
                        loss += ((curr-ref)**2).sum((1,2)).mean(0)
                loss.backward(retain_graph=False)
                opt.step()

                # recompute the noise
                with torch.no_grad():
                    noise_pred = self.model2.unet(
                        x_in.detach(),
                        t,
                        encoder_hidden_states=prompt_embeds_edit,
                        down_block_additional_residuals=controlnet_output.down_block_res_samples,
                        mid_block_additional_residual=controlnet_output.mid_block_res_sample,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=True)[0]         
                    # noise_pred = self.model2.unet(x_in.detach(),t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,).sample
                          
                latents = x_in.detach().chunk(2)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.model2.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = self.model2.scheduler.step(noise_pred, t, latents ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model2.scheduler.order == 0):
                    progress_bar.update()


        # 8. Post-processing
        image = self.model2.decode_latents(latents.detach())

        # 9. Run safety checker
        # image, has_nsfw_concept = self.model2.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image_edit = self.model1.numpy_to_pil(image)


        return image_rec, image_edit,condition_image
