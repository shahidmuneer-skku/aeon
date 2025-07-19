import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image, ImageFilter, ImageFile
from torchvision.transforms.functional import pil_to_tensor
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
from scipy.spatial.distance import cdist

import math
from pytorch_msssim import ssim, ms_ssim

ImageFile.LOAD_TRUNCATED_IMAGES = True
CALC_SIMILARITY = False

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

def image_distortion(img1, img2, seed, attack):
    if attack == 'rotation':
        img1 = transforms.RandomRotation((75, 75))(img1)
        img2 = transforms.RandomRotation((75, 75))(img2)

    if attack == 'jpeg':
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")

    if attack == 'cropping':
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img2)
        
    if attack == 'blurring':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))

    if attack == 'noise':
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, 0.1, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if attack == 'color_jitter':
        img1 = transforms.ColorJitter(brightness=6)(img1)
        img2 = transforms.ColorJitter(brightness=6)(img2)

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

def get_watermarking_pattern(pipe, args, device, shape=None):
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
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_up_radius, args.w_low_radius, -1):  
            tmp_mask = circle_mask(gt_init.shape[-1],r_max=i,r_min=args.w_low_radius)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        
    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    gt_patch = gt_patch.to(init_latents_w_fft.dtype)
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()  # complexhalf = complexfloat
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

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