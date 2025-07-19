#!/usr/bin/env python
"""
Complete script integrating an adaptive watermark generator with the InversableStableDiffusionPipeline.
This script uses a dummy training loop that injects an adaptive watermark (conditioned on image and prompt)
into the latent space and minimizes a dummy L1 loss between watermarked and original latents.
The adaptive watermark is now produced with shape (batch, 4, 64, 64) to match later code.
"""

import argparse
import os
import time
from functools import partial
from typing import Optional, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from optim_utils import *
from io_utils import *
# ---------------------------
# InversableStableDiffusionPipeline and helper functions
# ---------------------------
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler

# Note: //StableDiffusionPipeline is assumed to be defined in stable_diffusion_//.
# For this example, we assume it exists.
try:
    from stable_diffusion_// import //StableDiffusionPipeline
except ImportError:
    # Dummy placeholder in case the module is unavailable.
    class //StableDiffusionPipeline(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            dummy = cls()
            dummy.vae = AutoencoderKL()
            dummy.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            dummy.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            dummy.unet = UNet2DConditionModel()
            dummy.scheduler = LMSDiscreteScheduler()
            dummy.safety_checker = StableDiffusionSafetyChecker()
            dummy.feature_extractor = CLIPFeatureExtractor()
            dummy._execution_device = "cuda" if torch.cuda.is_available() else "cpu"
            dummy.unet.in_channels = 4
            dummy.unet.config = type("dummy", (), {"sample_size": 64})
            dummy.vae_scale_factor = 0.18215
            return dummy

class InversableStableDiffusionPipeline(//StableDiffusionPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super(InversableStableDiffusionPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    def get_random_latents(self, latents=None, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        batch_size = 1
        device = self.device
        num_channels_latents = self.unet.in_channels
        latents = torch.randn(batch_size, num_channels_latents, 64, 64, device=device)
        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        encoding = encoding_dist.sample(generator=rng_generator) if sample else encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: bool = False,
        latents_b=None,
        **kwargs,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        prompt_to_prompt = (old_text_embeddings is not None and new_text_embeddings is not None)
        noise_b = []
        start_time = time.time()
        for i, t in enumerate(timesteps_tensor if not reverse_process else reversed(timesteps_tensor)):
            if prompt_to_prompt:
                text_embeddings = old_text_embeddings if i < use_old_emb_i else new_text_embeddings

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_b.insert(0, noise_pred_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (self.scheduler.alphas_cumprod[prev_timestep]
                                 if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod)
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
            if latents_b is not None:
                latents_b.insert(0, latents)
        if latents_b is not None:
            return latents, latents_b, noise_b
        return latents

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))]
        return torch.cat(image, dim=0)

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.cpu().permute(0, 2, 3, 1).numpy()

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    return (
        alpha_tm1**0.5 *
        ((alpha_t**-0.5 - alpha_tm1**-0.5) * x_t +
         ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt)
        + x_t
    )

# ---------------------------
# Adaptive Watermark Modules
# ---------------------------
class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, latent_dim)
        
    def forward(self, x):
        b = x.size(0)
        features = self.conv(x)
        features = features.view(b, -1)
        return self.fc(features)

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
    
    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        _, hidden = self.rnn(emb)
        return hidden.squeeze(0)

class AdaptiveWatermarkGenerator(nn.Module):
    def __init__(self, latent_dim=256, wm_shape=(4, 64, 64)):
        """
        wm_shape: Desired watermark pattern shape (channels, H, W). Now set to (4, 64, 64).
        """
        super().__init__()
        self.wm_shape = wm_shape
        wm_size = wm_shape[0] * wm_shape[1] * wm_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, wm_size),
            nn.Tanh(),
        )
        
    def forward(self, img_feat, text_feat):
        fused = torch.cat([img_feat, text_feat], dim=-1)
        wm_flat = self.fc(fused)
        wm_pattern = wm_flat.view(-1, *self.wm_shape)
        return wm_pattern

class AdaptiveWatermarkModule(nn.Module):
    def __init__(self, img_latent_dim=256, text_embed_dim=256, wm_shape=(4, 64, 64)):
        super().__init__()
        self.img_encoder = ImageEncoder(latent_dim=img_latent_dim)
        self.text_encoder = TextEncoder(embed_dim=text_embed_dim, vocab_size=256)
        self.wm_generator = AdaptiveWatermarkGenerator(latent_dim=img_latent_dim, wm_shape=wm_shape)
    
    def forward(self, image, prompt_tokens):
        img_feat = self.img_encoder(image)
        text_feat = self.text_encoder(prompt_tokens)
        wm_pattern = self.wm_generator(img_feat, text_feat)
        return wm_pattern

# ---------------------------
# Dummy Tokenizer for Prompts
# ---------------------------
def tokenize_prompt(prompt, max_length=16):
    tokens = [ord(c) for c in prompt][:max_length]
    tokens = tokens + [0] * (max_length - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

# ---------------------------
# Dataset Definition
# ---------------------------
def get_img_tensor(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image)

class OptimizedDataset(Dataset):
    def __init__(self, data_root, size=512, repeats=10, interpolation="bicubic", set_name="train", center_crop=False):
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop
        file_list = os.listdir(self.data_root)
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in file_list]
        self.dataset = [{"prompt": f"Prompt for image {i}"} for i in range(len(self.image_paths))]
        self.prompt_key = "prompt"
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats if set_name == "train" else self.num_images
        self.interpolation = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])
        if image.mode != "RGB":
            image = image.convert("RGB")
        text = self.dataset[i % self.num_images][self.prompt_key]
        example["prompt"] = text
        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        example["pixel_values"] = get_img_tensor(image)
        return example

def create_dataloader(dataset, batch_size=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# Training Loop for Adaptive Watermark
# ---------------------------
def train_watermark(pipe, watermark_module, dataloader, optimizer, device, hyperparameters):
    pipe.to(device)
    watermark_module.to(device)
    num_steps = hyperparameters['max_train_steps']
    step = 0
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break
            images = batch["pixel_values"].to(device)
            prompts = batch["prompt"]
            prompt_tokens = torch.stack([tokenize_prompt(p) for p in prompts]).to(device)
            wm_pattern = watermark_module(images, prompt_tokens)
            latents = pipe.get_random_latents(height=64, width=64)
            if wm_pattern.shape[1] != latents.shape[1]:
                wm_pattern = wm_pattern.repeat(1, latents.shape[1], 1, 1)
            mask = torch.ones_like(wm_pattern)
            watermarked_latents = latents + mask * wm_pattern
            loss = F.l1_loss(watermarked_latents, latents)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"Step {step}: Loss {loss.item():.6f}")
            step += 1
    torch.save(watermark_module.state_dict(), os.path.join(hyperparameters["output_dir"], "adaptive_watermark.pt"))
    print("Watermark training complete and checkpoint saved.")

# ---------------------------
# Main Function
# ---------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    scheduler = LMSDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
    )
    pipe.to(device)
    train_dataset = OptimizedDataset(
        data_root=args.data_root,
        size=args.image_length,
        repeats=10,
        center_crop=False,
        set_name="train"
    )
        
    hyperparameters = {
        "learning_rate": 5e-04,
        "scale_lr": True,
        "max_train_steps": 2000, 
        "save_steps": 500,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "seed": 42,
        "output_dir": "sd-concept-output"
    }
    
    save_path = 'ckpts_adaptive'

    train_dataloader = create_dataloader(train_dataset, batch_size=args.train_batch_size)
    init_latents_w = pipe.get_random_latents()
    opt_watermark = get_watermarking_pattern(pipe, args, device)  # random generate an initial watermark
    mask = get_watermarking_mask(init_latents_w, args, device).detach().cpu()
    pipe.optimizer_wm_prompt(train_dataloader,hyperparameters, mask,opt_watermark,save_path,args)

    
# ---------------------------
# Argument Parser and Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Watermark Diffusion Training")
    parser.add_argument("--data_root", default="data/images", help="Directory containing training images")
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--max_train_steps", default=6000, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--output_dir", default="sd-concept-output")
    args = parser.parse_args()
    main(args)
