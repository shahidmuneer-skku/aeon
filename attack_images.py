import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image, ImageFilter, ImageFile
from torchvision.transforms.functional import pil_to_tensor
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
from scipy.spatial.distance import cdist
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype,to_pil_image
from PIL import Image
import math
from pytorch_msssim import ssim, ms_ssim


from WatermarkAttacker.wmattacker import VAEWMAttackerSingle
from Imprints.src.models.bvmr import bvmr
from Imprints.src.models.slbr import slbr
from wmattacker import *
from attndiffusion import ReSDPipeline

device="cuda"
att_pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
att_pipe.set_progress_bar_config(disable=True)
att_pipe.to(device)

ImageFile.LOAD_TRUNCATED_IMAGES = True
CALC_SIMILARITY = False
attack_model = VAEWMAttackerSingle(model_name="cheng2020-anchor") 

attack_model_bvm = bvmr(model_path="/media/NAS/USERS/shahid////Imprints/watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco/net_baseline_200.pth",
                        device=device)
# attack_model_slbr = slbr(model_path="/////Imprints/watermark_removal_works/SLBR/pretrained_model/model_best.pth.tar",
#                         device=device)

attackers = {
    'diff_attacker_60': DiffWMAttacker(att_pipe, noise_step=5,device=device),
    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
    'jpeg_attacker_50': JPEGAttacker(quality=50),
    'rotate_90': RotateAttacker(degree=90),
    'brightness_0.5': BrightnessAttacker(brightness=0.5),
    'contrast_0.5': ContrastAttacker(contrast=0.5),
    'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
    'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
    'bm3d': BM3DAttacker(),
}

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def to_ring(latent_fft, args):
    # Calculate mean value for each ring
    num_rings = args.w_up_radius - args.w_low_radius
    r_max = args.w_up_radius
    for i in range(num_rings):
        # ring_mask = mask[..., (radii[i * 2] <= distances) & (distances < radii[i * 2 + 1])]
        ring_mask = circle_mask(latent_fft.shape[-1], r_max=r_max, r_min=r_max-1)
        ring_mean = latent_fft[:, args.w_channel,ring_mask].real.mean().item()
        # print(f'ring mean: {ring_mean}')
        latent_fft[:, args.w_channel,ring_mask] = ring_mean
        r_max = r_max - 1

    return latent_fft

def to_numpy(img):
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, Image.Image):
        return np.array(img)
    elif isinstance(img, torch.Tensor):
        return img.cpu().detach().numpy()
    else:
        raise ValueError("Unsupported image type. Expected NumPy array, PIL Image, or Torch Tensor.")

def get_img_tensor(img):
    img_tensor = pil_to_tensor(img.convert("RGB"))/255
    return img_tensor#.unsqueeze(0)

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x

def image_distortion_multi(img1, img2, seed, attack):
    if attack == '1':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))

    if attack == '2':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")

    if attack == '3':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img2)
        
    if attack == '4':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img2)
        img1 = transforms.ColorJitter(brightness=6)(img1)
        img2 = transforms.ColorJitter(brightness=6)(img2)

    if attack == '5':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img2)
        img1 = transforms.ColorJitter(brightness=6)(img1)
        img2 = transforms.ColorJitter(brightness=6)(img2)
        img1 = transforms.RandomRotation((75, 75))(img1)
        img2 = transforms.RandomRotation((75, 75))(img2)

    if attack == '6':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img2)
        img1 = transforms.ColorJitter(brightness=6)(img1)
        img2 = transforms.ColorJitter(brightness=6)(img2)
        img1 = transforms.RandomRotation((75, 75))(img1)
        img2 = transforms.RandomRotation((75, 75))(img2)
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, 0.1, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    return img1, img2

def image_distortion(img1, img2, seed, attack, index=None):
    if attack == 'rotation':
        img1 = transforms.RandomRotation((75, 75))(img1)
        img2 = transforms.RandomRotation((75, 75))(img2)
        
        # img2.save(f"generated_imgs/rotation_attack/reconstructed_{index}.png")

    if attack == 'jpeg':
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")
        
        # img2.save(f"generated_imgs/jpeg_attack/reconstructed_{index}.png")

    if attack == 'cropping':
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img2)
        
        # img2.save(f"generated_imgs/crop_attack/reconstructed_{index}.png")
        
    if attack == 'blurring':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))

    if attack == 'noise':
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, 0.1, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))
        
        # img2.save(f"generated_imgs/noise_attack/reconstructed_{index}.png")

    if attack == 'color_jitter':
        img1 = transforms.ColorJitter(brightness=6)(img1)
        img2 = transforms.ColorJitter(brightness=6)(img2)
    if attack == 'reconstruction':
        img2 = attack_model.attack(img2)
    if attack == 'reconstructionBVMR':
        # img2 = Image.open("//profile.jpg")
        # img2 = img2.resize((512,512))
        # img2.save('generated_imgs/.png') 
        # img2 = attack_model.attack(img2)
        img2 = pil_to_tensor(img2)  # Converts to torch.uint8
        # img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")  # Converts to torch.float32 and scales values
        # img2 = attack_model_bvm.endtoend_func(img2.unsqueeze(0))
        # img2 = img2.squeeze(dim=0)

        for attacker_name, attacker in attackers.items():
            img2 = attackers[attacker_name].attack(img2)
            
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/reconstruction_attack/reconstructed_{index}.png")

        
    if attack == 'diff_attacker_60':
        img2 = pil_to_tensor(img2)  
        img2 = attackers["diff_attacker_60"].attack(img2)
        # print(img2)
        img2 = img2[0]
        # img2 = to_pil_image(img2)
        # print(img2)
        # img2.save(f"generated_imgs/diff_attacker_60/reconstructed_{index}.png")
    
    if attack == 'cheng2020-anchor_3':
        img2 = pil_to_tensor(img2)  
        # print(img2)ssss
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["cheng2020-anchor_3"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/cheng2020-anchor_3/reconstructed_{index}.png")
    
    if attack == 'bmshj2018-factorized_3':
        img2 = pil_to_tensor(img2)  
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["bmshj2018-factorized_3"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/bmshj2018-factorized_3/reconstructed_{index}.png")
    
    if attack == 'jpeg_attacker_50':
        img2 = pil_to_tensor(img2)  
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["jpeg_attacker_50"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/jpeg_attacker_50/reconstructed_{index}.png")
    
    if attack == 'rotate_90':
        img2 = pil_to_tensor(img2)  
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["rotate_90"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/rotate_90/reconstructed_{index}.png")
    
    if attack == 'brightness_0.5':
        img2 = pil_to_tensor(img2)  
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["brightness_0.5"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/brightness_0.5/reconstructed_{index}.png")
    
    if attack == 'Gaussian_noise':
        img2 = pil_to_tensor(img2)  
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["Gaussian_noise"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/Gaussian_noise/reconstructed_{index}.png")
    
    if attack == 'Gaussian_blur':
        img2 = pil_to_tensor(img2)  
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["Gaussian_blur"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/Gaussian_blur/reconstructed_{index}.png")
    
    if attack == 'bm3d':
        img2 = pil_to_tensor(img2)  
        img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")
        img2 = attackers["bm3d"].attack(img2)
        img2 = to_pil_image(img2)
        # img2.save(f"generated_imgs/bm3d/reconstructed_{index}.png")
    
    # if attack == 'reconstructionSLBR':
    #     img2 = pil_to_tensor(img2)  # Converts to torch.uint8
    #     img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")  # Converts to torch.float32 and scales values
    #     # img2 = attack_model_slbr.endtoend_func(img2.unsqueeze(0),mask)
    #     img2 = to_pil_image(img2.squeeze(dim=0))
        

    return img1, img2

# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)  # text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)

def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        # dataset = load_dataset(args.dataset)['test']
        dataset = load_dataset("parquet", data_files={'train': '/media/NAS/DATASET/Stable-Diffusion-Prompts/data/train.parquet', 'test': '/media/NAS/DATASET/Stable-Diffusion-Prompts/data/eval.parquet'})['test']
        prompt_key = 'Prompt'
    # Ensure the dataset is shuffled before splitting
    # dataset = dataset.shuffle(seed=42)
    # total_samples = len(dataset)
    # subset_size = int(total_samples * 0.2)
    # subset_dataset = dataset.select(range(subset_size))

    # print(f"Total samples in original dataset: {total_samples}")
    # print(f"Samples in 30% subset: {len(subset_dataset)}")


    return dataset, prompt_key


def circle_mask(size=64, r_max=10, r_min=0, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return (((x - x0)**2 + (y-y0)**2)<= r_max**2) & (((x - x0)**2 + (y-y0)**2) > r_min**2)

def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r_max=args.w_up_radius, r_min=args.w_low_radius)

        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask

# def get_watermarking_pattern(pipe, args, device, shape=None):
#     set_random_seed(args.w_seed)
#     # set_random_seed(10)  # test weak high freq watermark
#     if shape is not None:
#         gt_init = torch.randn(*shape, device=device)#.type(torch.complex32)
#     else:
#         gt_init = pipe.get_random_latents()

#     if 'seed_ring' in args.w_pattern:  # spacial
#         gt_patch = gt_init

#         gt_patch_tmp = copy.deepcopy(gt_patch)
#         for i in range(args.w_up_radius, args.w_low_radius, -1):
#             tmp_mask = circle_mask(gt_init.shape[-1], r_max=args.w_up_radius, r_min=args.w_low_radius)
#             tmp_mask = torch.tensor(tmp_mask).to(device)
            
#             for j in range(gt_patch.shape[1]):
#                 gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
#     elif 'seed_zeros' in args.w_pattern:
#         gt_patch = gt_init * 0
#     elif 'seed_rand' in args.w_pattern:
#         gt_patch = gt_init
#     elif 'rand' in args.w_pattern:
#         gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
#         gt_patch[:] = gt_patch[0]
#     elif 'zeros' in args.w_pattern:
#         gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
#     elif 'const' in args.w_pattern:
#         gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
#         gt_patch += args.w_pattern_const
#     elif 'ring' in args.w_pattern:
#         gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

#         gt_patch_tmp = copy.deepcopy(gt_patch)
#         for i in range(args.w_up_radius, args.w_low_radius, -1):  
#             tmp_mask = circle_mask(gt_init.shape[-1],r_max=i,r_min=args.w_low_radius)
#             tmp_mask = torch.tensor(tmp_mask).to(device)
            
#             for j in range(gt_patch.shape[1]):
#                 gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        
#     return gt_patch

def simple_gaussian_blur(x, kernel_size=7, sigma=3):
    """
    A simple Gaussian blur for a 4D tensor image (B,C,H,W).
    """
    # Create a 2D Gaussian kernel
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.to(x.device)

    # Expand the kernel to apply to each channel
    channels = x.shape[1]
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    # Pad the input so that the output size matches the input
    padding = kernel_size // 2
    x_blurred = F.conv2d(x, kernel, padding=padding, groups=channels)
    return x_blurred


def get_watermarking_pattern(pipe, args, device,image=None, shape=None,latent=None,dtype=None,alpha=0.6):
    set_random_seed(args.w_seed)
    # set_random_seed(10)  # test weak high freq watermark
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)#.type(torch.complex32)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:  # spacial
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_up_radius, args.w_low_radius, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r_max=args.w_up_radius, r_min=args.w_low_radius)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const

    # elif 'ring' in args.w_pattern:
    #     gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
    #     gt_patch_tmp = copy.deepcopy(gt_patch)
    #     for i in range(args.w_radius, 0, -1):
    #         tmp_mask = circle_mask(gt_init.shape[-1],r_max=i,r_min=args.w_low_radius)
    #         tmp_mask = torch.tensor(tmp_mask).to(device)
            
    #         for j in range(gt_patch.shape[1]):
    #             gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_up_radius, args.w_low_radius, -1):  
            tmp_mask = circle_mask(gt_init.shape[-1],r_max=i,r_min=args.w_low_radius)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'adaptive' in args.w_pattern:
            # Generate a dynamic watermark base in the frequency domain:
            base_init = latent
            base_fft = torch.fft.fftshift(torch.fft.fft2(base_init), dim=(-1, -2))
            # kernel_size = 3
            # sigma = 0.75
            # base_fft = gaussian_blur(base_fft, kernel_size, sigma)
            # Create three rings as before (for frequency control) if desired:
            size = base_fft.shape[-1]
            inner_radius = 10
            middle_radius = 20
            outer_radius = 30

            # ring1_mask = torch.tensor(circle_mask(size, r_max=inner_radius, r_min=0), device=device, dtype=torch.bool)
            # ring2_mask = torch.tensor(circle_mask(size, r_max=middle_radius, r_min=inner_radius), device=device, dtype=torch.bool)
            # ring3_mask = torch.tensor(circle_mask(size, r_max=outer_radius, r_min=middle_radius), device=device, dtype=torch.bool)
            # # base_fft = base_fft.to(dtype=)
            # # Set complex values for each ring (you might adjust these as needed):
            # for j in range(base_fft.shape[1]):
            #     base_fft[:, j, ring1_mask] = torch.tensor(
            #         2.0 + 2.0j, device=device, dtype=base_fft.dtype  # ComplexHalf
            #     )
            #     base_fft[:, j, ring2_mask] = torch.tensor(
            #         1.0 + 1.0j, device=device, dtype=base_fft.dtype  # ComplexHalf
            #     )
            #     base_fft[:, j, ring3_mask] = torch.tensor(
            #         0.5 + 0.5j, device=device, dtype=base_fft.dtype  # ComplexHalf
            #     )
            
            # Incorporate the diffusion information:
            diffusion_fft = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            blend_factor = 0.4
            combined_fft = blend_factor * diffusion_fft + (1 - blend_factor) * base_fft

            # Convert back to spatial domain
            gt_patch = torch.real(torch.fft.ifft2(torch.fft.ifftshift(base_fft, dim=(-1, -2))))
            
            # *** Inject a Visible Watermark Object ***
            # Assume watermark_img is a tensor representing the visible watermark (e.g., a logo)
            # and that it has been preprocessed to the appropriate size.
            image = image.resize((64,64))

            # Convert to NumPy array
            np_image = np.array(image)  # Shape: [H, W, C] (uint8)
            
            # Convert to Tensor
            tensor_image = torch.from_numpy(np_image).permute(2, 0, 1).float() / 255.0  # Convert to [C, H, W] and normalize
            # print(tensor_image.shape)
            watermark_img = tensor_image.to(device)  # Shape: [1, channels, H_w, W_w]
            if watermark_img.shape[0] == 3 and gt_patch.shape[1] == 4:
                # Create an extra channel of ones (you could also use zeros if needed)
                extra_channel = torch.ones(1, watermark_img.shape[1], watermark_img.shape[2], device=watermark_img.device)
                # Concatenate along the channel dimension
                watermark_img = torch.cat([watermark_img, extra_channel], dim=0)
                # print("Expanded watermark_img shape:", watermark_img.shape)  # Should now be [4, H_w, W_w]
            # print(gt_patch.shape,tensor_image.shape)
            # Determine the location (for example, bottom-right corner)
            # print(gt_patch.shape, watermark_img.shape)
            _, _, H, W = gt_patch.shape
            # print(watermark_img.shape)
            _, H_w, W_w = watermark_img.shape
            margin = 10
            x_start = max(0, W - W_w - margin)  # Ensure non-negative
            y_start = max(0, H - H_w - margin)  # Ensure non-negative

            # Optionally, you can also ensure the watermark fits into the target:
            if x_start + W_w > W or y_start + H_w > H:
                raise ValueError("Watermark does not fit in the target image with the given margin.")

            alpha = alpha # Blending factor

            # Perform blending in the selected region
            gt_patch[..., y_start:y_start+H_w, x_start:x_start+W_w] = (
                alpha * watermark_img + (1 - alpha) * gt_patch[..., y_start:y_start+H_w, x_start:x_start+W_w]
            )

    return gt_patch

# def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
#     init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
#     gt_patch = gt_patch.to(init_latents_w_fft.dtype)
#     if args.w_injection == 'complex':
#         init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()  # complexhalf = complexfloat
#     elif args.w_injection == 'seed':
#         init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
#         return init_latents_w
#     else:
#         NotImplementedError(f'w_injection: {args.w_injection}')

#     init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

#     return init_latents_w

def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    """
    Injects a watermark into the latent representation by blending the watermark patch 
    with the original latent values rather than fully replacing them. The blend factor is
    determined by args.watermark_alpha (0 = no watermark, 1 = full watermark).

    Parameters:
      init_latents_w   : Input latent tensor.
      watermarking_mask: Boolean mask indicating where to inject the watermark.
      gt_patch         : Watermark patch tensor.
      args             : Arguments that include:
                         - w_injection: injection mode ('complex' or 'seed')
                         - watermark_alpha: blending factor (e.g., 0.5 for 50% opacity)
    
    Returns:
      Updated latent tensor with watermark injected at lower opacity.
    """
    # Default watermark opacity if not specified.
    alpha = getattr(args, 'watermark_alpha', 0.8)
    
    # Compute the FFT and shift the zero-frequency component to the center.
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    gt_patch = gt_patch.to(init_latents_w_fft.dtype)
    
    if args.w_injection == 'complex':
        # Blend the watermark patch with the original FFT coefficients.
        init_latents_w_fft[watermarking_mask] = (
            (1 - alpha) * init_latents_w_fft[watermarking_mask] +
            alpha * gt_patch[watermarking_mask].clone()
        )
    elif args.w_injection == 'seed':
        # For the seed injection, blend directly in the spatial domain.
        init_latents_w[watermarking_mask] = (
            (1 - alpha) * init_latents_w[watermarking_mask] +
            alpha * gt_patch[watermarking_mask].clone()
        )
        return init_latents_w
    else:
        raise NotImplementedError(f'w_injection: {args.w_injection}')

    # Transform back to spatial domain.
    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args,mask_fake=None):
    if mask_fake is None:
        mask_fake = watermarking_mask
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[mask_fake]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[mask_fake]).mean().item()
    elif 'cosim' in args.w_measurement:
        no_w_metric = F.cosine_similarity(reversed_latents_no_w_fft[watermarking_mask].real, target_patch[watermarking_mask].real, dim=0).mean().item()
        w_metric = F.cosine_similarity(reversed_latents_w_fft[watermarking_mask].real, target_patch[watermarking_mask].real, dim=0).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_ssim(a, b):
    return ssim(a, b, data_range=1.).item()

def eval_psnr_ssim_msssim(ori_img, new_img):
    if ori_img.size != new_img.size:
        new_img = new_img.resize(ori_img.size)
    ori_x = transforms.ToTensor()(ori_img).unsqueeze(0)
    new_x = transforms.ToTensor()(new_img).unsqueeze(0)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)

def error_nmse(orig,recon):
    orig = to_numpy(orig).astype(np.float32)
    recon = to_numpy(recon).astype(np.float32)
    error = (np.linalg.norm((orig - recon)) / np.linalg.norm(orig))**2
    return error

def error_map(orig, recon, scale=1):
    orig = to_numpy(orig).astype(np.float32)
    recon = to_numpy(recon).astype(np.float32)
    error = orig - recon    
    error_map = ((error * scale + 127).clip(0,255).astype(np.uint8))
    return error_map

def cosine_distance(image1_embeds, image2_embeds):
    return F.cosine_similarity(image1_embeds, image2_embeds)

def fcosine_distance(image1_embeds, image2_embeds, get_err = False, watermarking_mask = None):
    freq_image1 = torch.fft.fftshift(torch.fft.fft2(image1_embeds), dim=(-1, -2)).real  # change complex to real part
    freq_image2 = torch.fft.fftshift(torch.fft.fft2(image2_embeds), dim=(-1, -2)).real
    if get_err:
        return F.cosine_similarity(freq_image1, freq_image2), torch.abs(freq_image1-freq_image2).mean().item(), torch.abs(freq_image1[watermarking_mask]-freq_image2[watermarking_mask]).mean().item()
        # return F.cosine_similarity(freq_image1, freq_image2), error_nmse(freq_image1, freq_image2), error_nmse(freq_image1[watermarking_mask], freq_image2[watermarking_mask])
    return F.cosine_similarity(freq_image1, freq_image2)

def norm_to_img(latents):
    latents = ((latents + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    latents = latents.permute(0, 2, 3, 1)
    latents = latents.contiguous()
    latents = latents[0].detach().cpu().numpy()
    img = Image.fromarray(latents, 'RGB')

    return img




def attack_image(img1, seed, attack, index=None):
    if attack == 'rotation':
        img1 = transforms.RandomRotation((75, 75))(img1)
        

    if attack == 'jpeg':
        img1.save(f"tmp_{25}_{attack}.jpg", quality=75)
        

    if attack == 'cropping':
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        
        
    if attack == 'blurring':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))

    if attack == 'noise':
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, 0.1, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        

    if attack == 'color_jitter':
        img1 = transforms.ColorJitter(brightness=6)(img1)


    if attack == 'reconstruction':
        img1 = attack_model.attack(img1)


    if attack == 'reconstructionBVMR':
        img1 = pil_to_tensor(img1)  # Converts to torch.uint8
        img1 = attack_model_bvm.endtoend_func(img1.unsqueeze(0))


        
    if attack == 'diff_attacker_60':
        img1 = pil_to_tensor(img1)  
        img1 = attackers["diff_attacker_60"].attack(img1, prompt="smiling, crying")[0]


    if attack == 'cheng2020-anchor_3':
        # img1 = img1[0]
        img1 = pil_to_tensor(img1)  
        # print(img1)
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["cheng2020-anchor_3"].attack(img1)
        img1 = to_pil_image(img1)
    
    if attack == 'bmshj2018-factorized_3':
        img1 = pil_to_tensor(img1)  
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["bmshj2018-factorized_3"].attack(img1)
        img1 = to_pil_image(img1)
    
    if attack == 'jpeg_attacker_50':
        img1 = pil_to_tensor(img1)  
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["jpeg_attacker_50"].attack(img1)
        img1 = to_pil_image(img1)
    
    if attack == 'rotate_90':
        img1 = pil_to_tensor(img1)  
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["rotate_90"].attack(img1)
        img1 = to_pil_image(img1)
    
    if attack == 'brightness_0.5':
        img1 = pil_to_tensor(img1)  
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["brightness_0.5"].attack(img1)
        img1 = to_pil_image(img1)
    
    if attack == 'Gaussian_noise':
        img1 = pil_to_tensor(img1)  
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["Gaussian_noise"].attack(img1)
        img1 = to_pil_image(img1)
    
    if attack == 'Gaussian_blur':
        img1 = pil_to_tensor(img1)  
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["Gaussian_blur"].attack(img1)
        img1 = to_pil_image(img1)
    
    if attack == 'bm3d':
        img1 = pil_to_tensor(img1)  
        img1 = convert_image_dtype(img1, dtype=torch.float32).to("cuda")
        img1 = attackers["bm3d"].attack(img1)
        img1 = to_pil_image(img1)
    
    # if attack == 'reconstructionSLBR':
    #     img2 = pil_to_tensor(img2)  # Converts to torch.uint8
    #     img2 = convert_image_dtype(img2, dtype=torch.float32).to("cuda")  # Converts to torch.float32 and scales values
    #     # img2 = attack_model_slbr.endtoend_func(img2.unsqueeze(0),mask)
    #     img2 = to_pil_image(img2.squeeze(dim=0))
        

    return img1

# 3) Resize to 512x512 (using a high-quality resample, e.g., LANCZOS or BICUBIC)


attacks = [
    # "Gaussian_noise",
    # "diff_attacker_60",
    # "cheng2020-anchor_3",
    # "bmshj2018-factorized_3",
    'bm3d'
    # "reconstructionBVMR"
]
# ackers = {
#     'diff_attacker_60': DiffWMAttacker(att_pipe, noise_step=5,device=device),
#     'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
#     'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
#     'jpeg_attacker_50': JPEGAttacker(quality=50),
#     'rotate_90': RotateAttacker(degree=90),
#     'brightness_0.5': BrightnessAttacker(brightness=0.5),
#     'contrast_0.5': ContrastAttacker(contrast=0.5),
#     'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
#     'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
#     'bm3d': BM3DAttacker(),
# }
seed = 111
folder  = "/media/NAS/DATASET/AADD-2025_Challenge/AADD-2025/Dataset/test/fake"
output_path =  "/media/NAS/DATASET/AADD-2025_Challenge/AADD-2025/Attacked_images/DiffusionAttack"
for folder_path in os.listdir(folder):
    for file_path in os.listdir(os.path.join(folder,folder_path)):
        file_name = os.path.basename(file_path)
        output = os.path.join(output_path, folder_path)
        os.makedirs(output_path,exist_ok=True)
        img1 = Image.open(os.path.join(folder,folder_path,file_path))
        img1 = img1.convert("RGB")
        img1 = img1.resize((512, 512), resample=Image.LANCZOS)
        for attack in attacks: 
            img1 = attack_image(img1, seed, attack, index=None)

        img1.save(os.path.join(output,file_name))