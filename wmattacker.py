from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import io
from skimage.util import random_noise
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bm3d import bm3d_rgb
from compressai.zoo import (
    bmshj2018_factorized, bmshj2018_hyperprior,
    mbt2018_mean, mbt2018, cheng2020_anchor
)

# Base attacker interface.
class WMAttacker:
    def attack(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

# VAE-based attack: resizes the input to 512x512, encodes/decodes with a VAE model.
class VAEWMAttacker(WMAttacker):
    def __init__(self, model_name, quality=1, metric='mse', device='cpu'):
        if model_name == 'bmshj2018-factorized':
            self.model = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'bmshj2018-hyperprior':
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018-mean':
            self.model = mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018':
            self.model = mbt2018(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'cheng2020-anchor':
            self.model = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)
        else:
            raise ValueError('model name not supported')
        self.device = device

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(image_tensor, list):
            # Convert each image in the list to a tensor (if it's not already)
            image_tensor = [img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img) for img in image_tensor]

            # Check if all tensors have the same shape before stacking
            shapes = [img.shape for img in image_tensor]
            if len(set(shapes)) > 1:  # If different shapes exist, resize them first
                image_tensor = [F.interpolate(img.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
                                for img in image_tensor]

            image_tensor = torch.stack(image_tensor)  # Now we can safely stack

        # Ensure tensor is batched
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        # Resize to 512x512 using bilinear interpolation.
        image_tensor = F.interpolate(image_tensor, size=(512, 512), mode='bilinear', align_corners=False)
        image_tensor = image_tensor.to(self.device)
        out = self.model(image_tensor)
        out['x_hat'].clamp_(0, 1)
        return out['x_hat'].squeeze(0).cpu()

# Gaussian blur attack: applies cv2.GaussianBlur.
class GaussianBlurAttacker(WMAttacker):
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Convert tensor (C,H,W) [0,1] to NumPy array (H,W,C) in 0-255 (uint8)
        image_np = (image_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
        blurred_np = cv2.GaussianBlur(image_np, (self.kernel_size, self.kernel_size), self.sigma)
        # Convert back to tensor with values in [0,1]
        blurred_tensor = torch.from_numpy(blurred_np).permute(2, 0, 1).float() / 255.0
        return blurred_tensor

# Gaussian noise attack: adds random Gaussian noise using skimage.
class GaussianNoiseAttacker(WMAttacker):
    def __init__(self, std=0.05):
        self.std = std

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Convert tensor (C,H,W) to NumPy (H,W,C)
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        noisy_np = random_noise(image_np, mode='gaussian', var=self.std ** 2)
        noisy_np = np.clip(noisy_np, 0, 1)
        noisy_tensor = torch.tensor(noisy_np).permute(2, 0, 1).float()
        return noisy_tensor

# BM3D attack: denoises the image using BM3D.
class BM3DAttacker(WMAttacker):
    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Convert tensor to NumPy array (H,W,C)
        img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        y_est = bm3d_rgb(img_np, 0.1)  # 0.1 is the noise standard deviation used in BM3D.
        y_est = np.clip(y_est, 0, 1)
        return torch.tensor(y_est).permute(2, 0, 1).float()

# JPEG attack: compresses the image via JPEG compression in memory.
class JPEGAttacker(WMAttacker):
    def __init__(self, quality=80):
        self.quality = quality

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Convert tensor to PIL image.
        img_pil = transforms.ToPILImage()(image_tensor.cpu())
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        img_jpeg = Image.open(buf).convert('RGB')
        return transforms.ToTensor()(img_jpeg)

# Brightness attack: adjusts image brightness.
class BrightnessAttacker(WMAttacker):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_pil = transforms.ToPILImage()(image_tensor.cpu())
        enhancer = ImageEnhance.Brightness(img_pil)
        bright_img = enhancer.enhance(self.brightness)
        return transforms.ToTensor()(bright_img)

# Contrast attack: adjusts image contrast.
class ContrastAttacker(WMAttacker):
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_pil = transforms.ToPILImage()(image_tensor.cpu())
        enhancer = ImageEnhance.Contrast(img_pil)
        contr_img = enhancer.enhance(self.contrast)
        return transforms.ToTensor()(contr_img)

# Rotate attack: rotates the image by a given degree.
class RotateAttacker(WMAttacker):
    def __init__(self, degree=30, expand=1):
        self.degree = degree
        self.expand = expand

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_pil = transforms.ToPILImage()(image_tensor.cpu())
        rotated = img_pil.rotate(self.degree, expand=self.expand)
        rotated = rotated.resize((512, 512))
        return transforms.ToTensor()(rotated)

# Scale attack: scales the image by a given factor.
class ScaleAttacker(WMAttacker):
    def __init__(self, scale=0.5):
        self.scale = scale

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_pil = transforms.ToPILImage()(image_tensor.cpu())
        w, h = img_pil.size
        new_size = (int(w * self.scale), int(h * self.scale))
        scaled = img_pil.resize(new_size)
        return transforms.ToTensor()(scaled)

# Crop attack: crops the image by removing a fraction of its width/height.
class CropAttacker(WMAttacker):
    def __init__(self, crop_size=0.5):
        self.crop_size = crop_size

    def attack(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img_pil = transforms.ToPILImage()(image_tensor.cpu())
        w, h = img_pil.size
        crop_box = (int(w * self.crop_size), int(h * self.crop_size), w, h)
        cropped = img_pil.crop(crop_box)
        return transforms.ToTensor()(cropped)

# Diffusion-based attack: applies noise in the latent space and then denoises using a diffusion pipeline.
class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, noise_step=60, device='cpu'):
        self.pipe = pipe
        self.device = device
        self.noise_step = noise_step

    def attack(self, image_tensor: torch.Tensor, prompt: str = "") -> torch.Tensor:
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            # Ensure image is batched and convert from [0,1] to [-1,1]
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            img = (image_tensor - 0.5) * 2
            img = img.to(self.device).type(torch.float16)
            latents_dist = self.pipe.vae.encode(img).latent_dist
            sf = self.pipe.vae.config.get('scaling_factor', 0.18215)
            latents = latents_dist.sample(generator) * sf
            noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            latents_noised = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
            head_start_step = 50 - max(self.noise_step // 20, 1)
            images = self.pipe(
                prompt,
                head_start_latents=latents_noised,
                head_start_step=head_start_step,
                guidance_scale=7.5,
                generator=generator,
            )
            out_img = images[0]
            if isinstance(out_img, Image.Image):
                out_img = transforms.ToTensor()(out_img)
            return out_img
