import argparse
import copy
import torch
from PIL import Image

from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline

# Make sure you have the needed functions in optim_utils.py
# (e.g., get_watermarking_mask, inject_watermark, etc.)
from optim_utils import get_watermarking_mask, inject_watermark


def load_pipe(model_id: str, device: str = 'cuda'):
    """
    Load the stable diffusion pipeline with DPMSolverMultistepScheduler.
    """
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_id, 
        subfolder='scheduler'
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    pipe = pipe.to(device)
    return pipe


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = load_pipe(args.model_id, device=device)

    # Prepare your text prompt
    prompt = args.prompt

    # Get text embedding for the (optionally unknown) prompt if needed
    # If you're not reversing images or needing text embeddings, you may skip this.
    text_embedding = pipe.get_text_embedding(prompt)

    # Load the watermark pattern
    # Expecting something like: 
    #   torch.save({'opt_wm': <YOUR_WATERMARK_LATENT>, 'opt_acond': <YOUR_PROMPT_ACOND>}, 'wm_path')
    ckpt = torch.load(args.wm_path, map_location=device)
    opt_wm   = ckpt.get('opt_wm', None)       # watermark latent (complex)
    opt_acond = ckpt.get('opt_acond', None)   # watermark conditioning, if used

    # Generate an initial random latent
    init_latents = pipe.get_random_latents()

    # Create a watermark mask of the appropriate shape (e.g., circle, random, etc.)
    watermarking_mask = get_watermarking_mask(init_latents, args, device)

    # 1) Generate image WITHOUT watermark
    # -----------------------------------------------------------
    # We simply call the pipeline with normal parameters
    init_latents_no_w = copy.deepcopy(init_latents)
    output_no_w = pipe(
        prompt,
        num_images_per_prompt=1,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.image_size,
        width=args.image_size,
        latents=init_latents_no_w,
    )
    image_no_w = output_no_w.images[0]
    image_no_w.save('inference_no_watermark.png')
    print("[INFO] Saved generation without watermark -> inference_no_watermark.png")

    # 2) Generate image WITH watermark
    # -----------------------------------------------------------
    # The pipeline accepts additional arguments to inject the watermark
    # you must ensure your `InversableStableDiffusionPipeline.__call__` is
    # modified to handle `watermarking_mask`, `gt_patch` (your watermark),
    # and any additional watermarking parameters.
    init_latents_w = copy.deepcopy(init_latents)
    output_w = pipe(
        prompt,
        num_images_per_prompt=1,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.image_size,
        width=args.image_size,
        latents=init_latents_w,
        # Additional watermarking arguments (depends on your pipeline):
        watermarking_mask=watermarking_mask,
        watermarking_steps=args.watermarking_steps,
        args=args,             # you might pass the entire args
        gt_patch=opt_wm,       # the watermark pattern
        lguidance=args.lguidance,
        opt_acond=opt_acond    # if your pipeline uses an optimized prompt conditioning
    )
    image_w = output_w.images[0]
    image_w.save('inference_with_watermark.png')
    print("[INFO] Saved generation with watermark -> inference_with_watermark.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base',
                        help='HuggingFace model id or local path to stable diffusion checkpoint.')
    parser.add_argument('--prompt', type=str, default='a beautiful painting of a sunset over the sea',
                        help='Prompt for text-to-image generation.')
    parser.add_argument('--w_channel', default=3, type=int)
    parser.add_argument('--wm_path', type=str, default='ckpts/watermark.pt',
                        help='Path to your optimized watermark checkpoint.')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Width and height of generated image.')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale.')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps for diffusion.')
    parser.add_argument('--watermarking_steps', type=int, default=35,
                        help='At which timestep(s) you inject the watermark.')
    parser.add_argument('--lguidance', type=float, default=7.5,
                        help='Potentially a separate guidance scale for watermark injection.')
    # Watermarking mask parameters
    parser.add_argument('--w_seed', type=int, default=999999,
                        help='Seed for generating watermark pattern.')
    parser.add_argument('--w_up_radius', type=int, default=30,
                        help='Upper radius used for watermark mask generation.')
    parser.add_argument('--w_low_radius', type=int, default=5,
                        help='Lower radius used for watermark mask generation.')
    parser.add_argument('--w_pattern', type=str, default='rand',
                        help='Type of watermark pattern (rand, circle, etc.).')
    parser.add_argument('--w_mask_shape', type=str, default='circle',
                        help='Shape of watermark region (circle, square, etc.).')
    # More can be added depending on your watermark logic

    args = parser.parse_args()
    main(args)
