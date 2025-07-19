import torch
import numpy as np
from diffusers import DDIMInverseScheduler
from typing import Union
from PIL import Image
from PIL import Image
import torch

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

from inverse_stable_diffusion import InversableStableDiffusionPipeline
# Re-use your existing imports or local utilities
from tree_ring_watermark._detect import _transform_img, _circle_mask, load_keys
from tree_ring_watermark.utils import get_org
from huggingface_hub import snapshot_download
from optim_utils import *
from io_utils import *
class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def extract_watermark(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    pipe,
    model_hash: str,
    threshold: float = 77.0
):
    """
    Attempt to extract or recover the watermark from a watermarked image using
    the stored 'keys' in the HF dataset for the given `model_hash`.

    1) Invert image -> latents (via DDIMInverseScheduler).
    2) Convert latents to frequency domain.
    3) For each known watermark key, measure the distance (like `_detect.detect()`).
       - If below `threshold`, we assume that key + mask is correct.
    4) Extract the watermark pattern from the matched frequency region and return it.
    
    Returns
    -------
    extracted_pattern_spatial : torch.Tensor or None
        The extracted watermark in the *spatial (latent) domain*, or None if not found.
    w_key_used : torch.Tensor or None
        The matching watermark key that triggered success detection, or None otherwise.
    """

    # 1) Download the relevant watermark keys from HF
    # org = get_org()  # from `utils.py`
    # repo_id = f"{org}/{model_hash}"
    # cache_dir = snapshot_download(repo_id, repo_type="dataset")
    # keys = load_keys(cache_dir)  # e.g. {filename: np.array(...)}
    path = "/////ckpts/optimized_wm5-30_embedding-step-2000.pt"
    keys  = torch.load(path)['opt_wm'].to(device)
    # 2) Swap scheduler to DDIMInverse to invert the image
    original_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    # w_mask = get_watermarking_mask(inverted_latents, args, device) 
    # 3) Convert `image` to latent
    img = _transform_img(image).unsqueeze(0).to(pipe.device, dtype=pipe.unet.dtype)  # from _detect.py
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215

    # 4) Run the "inversion" pass
    #    50 is typical, but use the same # steps you used in detection
    num_inference_steps = 50
    
    init_latents_no_w = pipe.get_random_latents()
    with torch.no_grad():
        inverted_output = pipe(
            prompt="",
            latents=image_latents,
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            watermarking_mask=w_mask
        )
    inverted_latents = torch.from_numpy(inverted_output.images).permute(0,3,1,2)  # shape: (1,4,64,64)

    print(inverted_latents.shape)

    # 5) Convert to frequency domain
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))

    # 6) Try each known key & mask
    shape = inverted_latents.shape
    best_dist = float('inf')
    best_key = None
    best_mask = None

    # for filename, w_key_np in keys.items():
    print(keys.shape)
    if keys.shape[0]==1:
        # # parse "somehash_w_channel_w_radius_wpattern.npy"
        # # e.g. "abcdef12345_0_10_rand.npy"
        # # parts = filename.replace(".npy", "").split("_")
        # # if len(parts) < 3:
        # #     continue
        # w_channel = int(keys[1])
        # w_radius = int(keys[2])

        # # Reconstruct the mask
        # np_mask = _circle_mask(shape[-1], r=w_radius)
        # torch_mask = torch.tensor(np_mask)
        # w_mask = torch.zeros(shape, dtype=torch.bool)
        # w_mask[:, w_channel] = torch_mask

        # Convert the stored key from np to torch
        
        args = Args(w_mask_shape="circle", w_channel=3, w_up_radius=30, w_low_radius=5)
        w_key = keys.to(inverted_latents_fft.device, dtype=inverted_latents.dtype)

        # measure the distance in the masked region
        w_key = w_key.to(device)
        inverted_latents_fft = inverted_latents_fft.to(device)
        dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

        if dist < best_dist:
            best_dist = dist
            best_key = w_key
            best_mask = w_mask

    # 7) If best distance is above threshold => No watermark found
    if best_dist > threshold or best_key is None or best_mask is None:
        pipe.scheduler = original_scheduler
        return None, None

    # Otherwise, we found a matching key. “Extract” the actual watermark content.
    # 
    # The user can define "extraction" differently, but a common approach:
    #  - isolate the difference in the freq domain
    #  - or directly isolate the freq domain region that was replaced by best_key
    #  - transform that slice back to spatial domain if you want to see the pattern
    #
    # Example #1: The "raw" region in freq domain that was inserted.
    extracted_fft = torch.zeros_like(inverted_latents_fft)
    extracted_fft[best_mask] = inverted_latents_fft[best_mask].clone()

    # Example #2: If you want to see how it differs from the known key in freq domain:
    # extracted_fft[best_mask] = (inverted_latents_fft[best_mask] - best_key[best_mask])

    # 8) Convert freq-domain slice -> spatial domain
    extracted_pattern_spatial_cplx = torch.fft.ifft2(torch.fft.ifftshift(extracted_fft, dim=(-1, -2)))
    extracted_pattern_spatial = extracted_pattern_spatial_cplx.real

    # 9) Restore original scheduler
    pipe.scheduler = original_scheduler

    return extracted_pattern_spatial, best_key

# Suppose you have a watermarked image at 'watermarked.png'
image = Image.open('//profile.jpg').convert('RGB')

# Load your standard pipe (just like target_api.py does):
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = 'stabilityai/stable-diffusion-2-1-base'
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
pipe = InversableStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to(device)


# The 'model_hash' must match whatever was used in `_get_noise.get_noise(...)`
extracted_pattern, matched_key = extract_watermark(image, pipe, model_hash="stable-diffusion-2-1-base")

if extracted_pattern is None:
    print("[INFO] No watermark detected / extracted.")
else:
    print("[INFO] Watermark recovered!")
    print(f"Extracted pattern shape: {extracted_pattern.shape}")
    # Example: Save the extracted watermark in latents domain
    torch.save(extracted_pattern, "extracted_watermark_latents.pt")
