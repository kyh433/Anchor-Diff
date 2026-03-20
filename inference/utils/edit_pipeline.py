import sys
import copy
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

sys.path.insert(0, "inference/utils")
from base_pipeline import BasePipeline
from cross_attention import prep_unet

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class EditingPipeline_control:
    def __init__(self, model1: BasePipeline, model2):
        self.model1 = model1
        self.model2 = model2

        vae = model2.vae
        text_encoder = model2.text_encoder
        tokenizer = model2.tokenizer
        unet = model2.unet
        scheduler = model2.scheduler
        safety_checker = model2.safety_checker
        feature_extractor = model2.feature_extractor
        image_encoder = getattr(model2, "image_encoder", None)

        self.model2.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.model2.vae_scale_factor = 2 ** (len(self.model2.vae.config.block_out_channels) - 1) if getattr(self.model2, "vae", None) else 8
        self.model2.image_processor = VaeImageProcessor(vae_scale_factor=self.model2.vae_scale_factor, do_convert_rgb=True)

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
        guidance_amount=0.1,
        edit_dir=None,
        x_in=None,
        only_sample=False,
    ):
        model1 = copy.copy(self.model1)

        model1.unet = prep_unet(model1.unet)
        self.model2.unet = prep_unet(self.model2.unet)

        d_ref_t2attn = {}

        height = height or model1.unet.config.sample_size * model1.vae_scale_factor
        width = width or model1.unet.config.sample_size * model1.vae_scale_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = model1._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        x_in = x_in.to(dtype=model1.unet.dtype, device=device)

        prompt_embeds, negative_prompt_embeds = model1.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        model1.scheduler.set_timesteps(num_inference_steps, device=device)
        self.model2.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = model1.scheduler.timesteps
        timesteps2 = self.model2.scheduler.timesteps

        num_channels_latents = model1.unet.config.in_channels
        latents = model1.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = model1.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * model1.scheduler.order
        with torch.no_grad():
            with model1.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = model1.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = model1.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    d_ref_t2attn[t.item()] = {}
                    for name, module in model1.unet.named_modules():
                        module_name = type(module).__name__
                        if module_name == "Attention" and 'attn2' in name:
                            attn_mask = module.attn_probs
                            d_ref_t2attn[t.item()][name] = attn_mask.detach().cpu()

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = model1.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model1.scheduler.order == 0):
                        progress_bar.update()

        with torch.no_grad():
            image_rec = model1.numpy_to_pil(model1.decode_latents(latents))

        if only_sample:
            return image_rec

        del model1
        torch.cuda.empty_cache()

        self.model2.unet.to(device)
        self.model2.vae.to(device)

        prompt_edit = prompt
        prompt_embeds, negative_prompt_embeds = self.model2.encode_prompt(
            prompt_edit,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        prompt_embeds_edit = torch.cat([negative_prompt_embeds, prompt_embeds]) if do_classifier_free_guidance else prompt_embeds

        latents = self.model2.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        condition_image = image_rec

        num_warmup_steps = len(timesteps2) - num_inference_steps * self.model2.scheduler.order
        with self.model2.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps2):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.model2.scheduler.scale_model_input(latent_model_input, t)

                x_in = latent_model_input.detach().clone()
                x_in.requires_grad = True
                opt = torch.optim.SGD([x_in], lr=guidance_amount)

                noise_pred = self.model2.unet(
                    x_in,
                    t,
                    encoder_hidden_states=prompt_embeds_edit.detach(),
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=True,
                )[0]

                loss = torch.tensor(0.0, device=device)
                for name, module in self.model2.unet.named_modules():
                    module_name = type(module).__name__
                    if module_name == "Attention" and 'attn2' in name:
                        curr = module.attn_probs
                        ref = d_ref_t2attn[t.item()][name].detach().to(device)
                        loss += ((curr - ref) ** 2).sum((1, 2)).mean(0)
                loss.backward(retain_graph=False)
                opt.step()

                with torch.no_grad():
                    noise_pred = self.model2.unet(
                        x_in.detach(),
                        t,
                        encoder_hidden_states=prompt_embeds_edit,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=True,
                    )[0]

                latents = x_in.detach().chunk(2)[0] if do_classifier_free_guidance else x_in.detach()

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.model2.scheduler.step(noise_pred, t, latents).prev_sample

                if i == len(timesteps2) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.model2.scheduler.order == 0):
                    progress_bar.update()

        with torch.no_grad():
            image = self.model2.decode_latents(latents.detach())
        image_edit = self.model1.numpy_to_pil(image)

        return image_rec, image_edit, condition_image
