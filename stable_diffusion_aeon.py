
from typing import Callable, List, Optional, Union, Any, Dict
import copy
import os
import numpy as np
import PIL
from statistics import mean
from tqdm import tqdm
import itertools
import time
import torch.nn as nn
from torch import inference_mode
import torch
from torch.cuda.amp import GradScaler, autocast
from diffusers import StableDiffusionPipeline
from diffusers.utils import BaseOutput
import logging
from optim_utils import *

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logging.basicConfig(
    level=logging.INFO,  # seg logger level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # set logger format
    handlers=[
        logging.StreamHandler(),  # output to terminal
        logging.FileHandler('logs/output.log', mode='a', encoding='utf-8')  # output to file
    ]
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def get_pred_ori(x_t, alpha_t, epx_xt):
    ''' from xt to x0'''
    beta_t = 1 - alpha_t
    return (
        (x_t - beta_t ** (0.5)
         * epx_xt) / alpha_t ** (0.5)
    )

class DeepHashNet(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.flatten_dim = 4 * 64 * 64            # =16 384
        self.hash_layer  = nn.Linear(self.flatten_dim, self.flatten_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_flat = x.view(B, -1)                    # 1. Flatten

        z = torch.tanh(self.hash_layer(x_flat))   # 2. Forward pass → Z

        # 3. Straight‑through estimator  H ← sign(Z) + (Z – stop‑grad(Z))
        hash_code = (z.sign() - z).detach() + z           # forward = sign(z), grad = 1
        
        reconstructed = hash_code.view(B, 4, 64, 64)  # Reshape to (B, 4, 64, 64)

        return reconstructed
    



class AEONStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]
    latents: Optional[torch.FloatTensor]
    inner_latents: Optional[List[torch.FloatTensor]]

class AEONStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super(AEONStableDiffusionPipeline, self).__init__(vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                requires_safety_checker)
                    
        self.hash_net = DeepHashNet()
        self.hash_net.load_state_dict(torch.load("/<?Path to checkpoint>////ckpts_adaptive_reconstruction_hashing/optimized_wm5-30_embedding-step-2000.pt")["hashing_model"])
        self.hash_net.to(device=vae.device, dtype=vae.dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        watermarking_mask: Optional[torch.BoolTensor] = None,
        watermarking_steps: int = None,
        args = None,
        gt_patch = None,
        lguidance = None,
        opt_acond = None,
        has_net=None
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # print('got new version')
        inner_latents = []
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        init_latents = copy.deepcopy(latents)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        inner_latents.append(init_latents)

        # 7. Denoising loop
        max_train_steps=1  #100
        latents_wm = None
        text_embeddings_opt = None
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        start_time = time.time()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if (watermarking_steps is not None) and (i >= watermarking_steps):
                    mask = watermarking_mask  # mask from outside
                    if i == watermarking_steps:
                        self.hash_net = self.hash_net.to(latents.device)
                        gt_patch = self.hash_net(latents)
                        latents_wm = inject_watermark(latents, mask,gt_patch, args)  # inject latent watermark
                        inner_latents[-1] = latents_wm  
                        if opt_acond is not None:
                            uncond, cond = text_embeddings.chunk(2)
                            opt_acond = opt_acond.to(cond.dtype)
                            text_embeddings_opt = torch.cat([uncond, opt_acond, cond])  # opt as another cond
                        else:
                            text_embeddings_opt = text_embeddings.clone()
                        if lguidance is not None:
                            guidance_scale = lguidance  

                    latents_wm, _ = self.xn1_latents_3(latents_wm,do_classifier_free_guidance,t
                                                            ,text_embeddings_opt,guidance_scale,**extra_step_kwargs)

                if (watermarking_steps is None) or (watermarking_steps is not None and i < watermarking_steps):
                    latents, _ = self.xn1_latents(latents,do_classifier_free_guidance,t
                                                            ,text_embeddings,guidance_scale,**extra_step_kwargs)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                if (watermarking_steps is not None and i < watermarking_steps) or (watermarking_steps is None):
                    inner_latents.append(latents)   # save for memory
                else: 
                    inner_latents.append(latents_wm)

                if watermarking_steps is not None and watermarking_steps == 50:
                    latents_wm = inject_watermark(latents, watermarking_mask,gt_patch, args)  # inject latent watermark
                    inner_latents[-1] = latents_wm  

        end_time = time.time()
        execution_time = end_time - start_time
        # 8. Post-processing
        if latents_wm is not None:
            image = self.decode_latents(latents_wm)
        else:
            image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)
        if text_embeddings_opt is not None:
            return AEONStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents, latents=latents, inner_latents=inner_latents,gt_patch=gt_patch,opt_acond=text_embeddings_opt[0],time=execution_time)
        else:
            return AEONStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents, latents=latents, inner_latents=inner_latents,gt_patch=gt_patch,time=execution_time)
        

    def optimizer_wm_prompt(self, dataloader,hyperparameters, mask,opt_wm,save_path,args,
                            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                            eta: float = 0.0,     ):
        train_batch_size = hyperparameters["train_batch_size"]
        gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
        learning_rate = hyperparameters["learning_rate"]
        max_train_steps = hyperparameters["max_train_steps"]
        output_dir = hyperparameters["output_dir"]
        gradient_checkpointing = hyperparameters["gradient_checkpointing"]

        text_encoder = self.text_encoder
        unet = self.unet
        vae = self.vae
        scheduler = self.scheduler

        freeze_params(vae.parameters())
        freeze_params(unet.parameters())
        freeze_params(text_encoder.parameters())

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=hyperparameters["mixed_precision"]
        )

        if gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        if hyperparameters["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        tester_prompt = '' # assume at the detection time, the original prompt is unknown
        text_embeddings = self.get_text_embedding(tester_prompt)  # text_embedding.dtype = torch.float16

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        unet, text_encoder, dataloader,text_embeddings = accelerator.prepare(
            unet, text_encoder, dataloader, text_embeddings
        ) 

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)

        # Keep vae in eval mode as we don't train it
        vae.eval()
        # Keep unet in train mode to enable gradient checkpointing
        unet.train()

        
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataloader)}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        scaler = GradScaler()
        # self.scheduler.set_timesteps(1000)  # need for compute the next state
        
        opt_wm_embedding = self.get_text_embedding('')
        null_embedding = opt_wm_embedding.clone()
        total_time = 0
        with autocast():
            for epoch in range(num_train_epochs):
                for step, batch in enumerate(dataloader):
                    with accelerator.accumulate(unet):
                        # Convert images to latent space
                        gt_tensor = batch["pixel_values"]
                        image = 2.0 * gt_tensor - 1.0
                        latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample().detach()
                        latents = latents * 0.18215
                       
                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(200, 300, (bsz,), device=latents.device).long()  # 35～40steps

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                        opt_wm = opt_wm.to(noisy_latents.device).to(torch.complex64)  # add wm to latents
                        
                        ### detailed the inject_watermark function for fft.grad
                        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(noisy_latents), dim=(-1, -2))
                        init_latents_w_fft[mask] = opt_wm[mask].clone()
                        init_latents_w_fft.requires_grad = True
                        noisy_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

                        ### Get the text embedding for conditioning CFG 
                        prompt = batch["prompt"]
                        # # print(f'prompt: {prompt}')
                        cond_embedding = self.get_text_embedding(prompt)
                        text_embeddings = torch.cat([opt_wm_embedding, cond_embedding, null_embedding]) 
                        text_embeddings.requires_grad = True

                        ### Predict the noise residual with CFG 
                        latent_model_input = torch.cat([noisy_latents] * 3)
                        latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)
                        noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample
                        noise_pred_wm, noise_pred_text, noise_pred_null = noise_pred.chunk(3)
                        noise_pred = noise_pred_null + 3.5 * (noise_pred_text - noise_pred_null) + 3.5 * (noise_pred_wm - noise_pred_null)   # different guidance scale
                        
                        ### get the predicted x0 tensor
                        x0_latents = scheduler.convert_model_output(noise_pred,timesteps.item(),noisy_latents)  #predict x0 in one-step
                        x0_tensor = self.decode_latents_wgrad(x0_latents)

                        loss_noise = F.mse_loss(x0_tensor.float(), gt_tensor.float(), reduction="mean")  # pixel alignment
                        loss_wm = torch.mean(torch.abs(opt_wm[mask].real))
                        loss_constrain = F.mse_loss(noise_pred_wm.float(), noise_pred_null.float(), reduction="mean")  # prompt constraint

                        ### optimize wm pattern and uncond prompt alternately
                        if (global_step // 500) % 2 == 0:
                            loss = 10 * loss_noise + loss_constrain - 0.00001 * loss_wm  # opt wm pattern
                            accelerator.backward(loss)
                            with torch.no_grad():  
                                grads = init_latents_w_fft.grad
                                init_latents_w_fft = init_latents_w_fft - 1.0 * grads  # update wm pattern
                                init_latents_w_fft = to_ring(init_latents_w_fft, args)
                                opt_wm = init_latents_w_fft.detach()
                        else:
                            loss = 10 * loss_noise + loss_constrain  # opt prompt
                            accelerator.backward(loss)
                            with torch.no_grad():  
                                grads = text_embeddings.grad
                                text_embeddings = text_embeddings - 5e-04 * grads  
                                opt_wm_embedding = text_embeddings[0].unsqueeze(0).detach()  # update acond embedding


                        print(f'global_step: {global_step}, loss_mse: {loss_noise}, loss_wm: {loss_wm}, loss_cons: {loss_constrain},loss: {loss}')

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        if global_step % hyperparameters["save_steps"] == 0:
                            path = os.path.join(save_path, f"optimized_wm5-30_embedding-step-{global_step}.pt")
                            torch.save({'opt_acond': opt_wm_embedding, 'opt_wm': opt_wm.cpu()}, path)

                    logs = {"loss": loss.detach().item()}
                    progress_bar.set_postfix(**logs)

                    if global_step >= max_train_steps:
                        break

                accelerator.wait_for_everyone()

        return opt_wm, opt_wm_embedding


    def xn1_latents(self,latents,do_classifier_free_guidance,t
                        ,text_embeddings,guidance_scale,**extra_step_kwargs):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents, noise_pred
    
    def xn1_latents_3(self,latents,do_classifier_free_guidance,t
                        ,text_embeddings,guidance_scale,**extra_step_kwargs):
        latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text1, noise_pred_text2 = noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + 3.5 * (noise_pred_text1 - noise_pred_uncond) + 3.5 * (noise_pred_text2 - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents, noise_pred
    
    def decode_latents_wgrad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def get_noise(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            watermarking_mask: Optional[torch.BoolTensor] = None,
            watermarking_steps: int = None,
            args = None,
            gt_patch = None,
        ):
        
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(prompt, height, width, callback_steps)

            # 2. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            text_embeddings = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # add noise logger
            inner_noise = []
            uncond_noise = []
            textcond_noise = []
            guidance_noise = []
            freq_noise = []
            noise_sim = []
            uncond_sim = []
            textcond_sim = []
            guidance_sim = []
            freq_sim = []
            latents_sim = []
            noise_sim_wm = []
            uncond_sim_wm = []
            textcond_sim_wm = []
            guidance_sim_wm = []
            freq_sim_wm = []
            latents_sim_wm = []
            mask = watermarking_mask
            latents_wm = None

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with torch.no_grad(): 
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        
                        # compute watermark noise
                        if (watermarking_steps is not None) and (i == watermarking_steps):
                            latents_wm = inject_watermark(latents, mask,gt_patch, args)

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute inner_noise as mean_value
                        inner_noise.append(torch.abs(noise_pred).mean().item())
                        uncond_noise.append(torch.abs(noise_pred_uncond).mean().item())
                        textcond_noise.append(torch.abs(noise_pred_text).mean().item())
                        guidance_noise.append(torch.abs(noise_pred - noise_pred_uncond).mean().item())
                        freq_noise.append(torch.abs(torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2)).real).mean().item())

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        # compute inner_noise as cos_sim
                        if i != 0:
                            noise_sim.append(cosine_distance(prev_noise, noise_pred).mean().item())
                            uncond_sim.append(cosine_distance(prev_uncond, noise_pred_uncond).mean().item())
                            textcond_sim.append(cosine_distance(prev_cond, noise_pred_text).mean().item())
                            guidance_sim.append(cosine_distance(prev_guidance, noise_pred - noise_pred_uncond).mean().item())
                            freq_sim.append(fcosine_distance(prev_latents, latents).mean().item())
                            latents_sim.append(cosine_distance(prev_latents, latents).mean().item())

                        # compute the predicted original sample x_t -> x_0
                        # alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        # latents_ori = get_pred_ori(latents, alpha_prod_t, noise_pred)
                        # res_noise.append(torch.abs(latents-latents_ori).mean().item())

                        # print("inner_noise: ", inner_noise, "res_noise: ", res_noise)

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                callback(i, t, latents)

                        

                        if latents_wm is not None:
                            ### watermark both conditional and unconditional
                            latent_model_input_wm = torch.cat([latents_wm] * 2) if do_classifier_free_guidance else latents_wm

                            ### watermark unconditional only
                            # if i == watermarking_steps:
                            #     latent_model_input_wm = torch.cat([latents_wm, prev_latents]) if do_classifier_free_guidance else latents_wm   # only watermark unconditional part :(
                            # else:
                            #     latent_model_input_wm = torch.cat([latents_wm] * 2) if do_classifier_free_guidance else latents_wm
                            latent_model_input_wm = self.scheduler.scale_model_input(latent_model_input_wm, t)

                            # predict the noise residual
                            noise_pred_wm = self.unet(latent_model_input_wm, t, encoder_hidden_states=text_embeddings).sample

                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond_wm, noise_pred_text_wm = noise_pred_wm.chunk(2)
                                noise_pred_wm = noise_pred_uncond_wm + guidance_scale * (noise_pred_text_wm - noise_pred_uncond_wm)  
                                # noise_pred_wm = noise_pred_uncond_wm + 0.0 * (noise_pred_text_wm - noise_pred_uncond_wm)  # decrease the guidance_scale

                            # compute cos_sim between ori and ori_wm
                            noise_sim_wm.append(cosine_distance(noise_pred_wm, noise_pred).mean().item())
                            uncond_sim_wm.append(cosine_distance(noise_pred_uncond_wm, noise_pred_uncond).mean().item())
                            textcond_sim_wm.append(cosine_distance(noise_pred_text_wm, noise_pred_text).mean().item())
                            guidance_sim_wm.append(cosine_distance(noise_pred_wm-noise_pred_uncond_wm, noise_pred-noise_pred_uncond).mean().item())
                            latents_sim_wm.append(cosine_distance(latents_wm, latents).mean().item())
                            freq_sim_wm.append(fcosine_distance(latents_wm, latents).mean().item())
       
                            # compute the previous noisy sample x_t -> x_t-1
                            latents_wm = self.scheduler.step(noise_pred_wm, t, latents_wm, **extra_step_kwargs).prev_sample

                        prev_noise = noise_pred
                        prev_uncond = noise_pred_uncond
                        prev_cond = noise_pred_text
                        prev_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                        prev_latents = latents

            if latents_wm is None:
                return inner_noise,uncond_noise,textcond_noise,guidance_noise,freq_noise,noise_sim,uncond_sim,textcond_sim,guidance_sim,freq_sim# res_noise, wm_noise
            else:
                return inner_noise,uncond_noise,textcond_noise,guidance_noise,freq_noise,noise_sim,uncond_sim,textcond_sim,guidance_sim,latents_sim, freq_sim, noise_sim_wm, uncond_sim_wm, textcond_sim_wm, guidance_sim_wm, latents_sim_wm, freq_sim_wm

    @torch.no_grad()
    def get_watermark_persistence(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            watermarking_mask: Optional[torch.BoolTensor] = None,
            watermarking_steps: int = None,
            args = None,
            gt_patch = None,
        ):
        
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(prompt, height, width, callback_steps)

            # 2. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            text_embeddings = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # add noise logger
            wm_nmse = []
            mask = watermarking_mask

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with torch.no_grad(): 
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        
                        # compute watermark noise
                        if (watermarking_steps is not None) and (i == watermarking_steps):
                            latents = inject_watermark(latents, mask,gt_patch, args)

                        # evaluate watermark persistence
                        if (watermarking_steps is not None)and (i >= watermarking_steps):
                            freq_latents = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2)).real 
                            wm_nmse.append(error_nmse(gt_patch[mask].real, freq_latents[mask]))

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                callback(i, t, latents)

                    
            ### only for compute nmse for clean latents
            freq_latents = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2)).real 
            wm_nmse.append(error_nmse(gt_patch[mask].real, freq_latents[mask]))
            return wm_nmse

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        # image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents