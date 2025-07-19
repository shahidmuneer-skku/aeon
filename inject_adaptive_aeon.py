

import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
from collections import defaultdict
import pickle
import numpy as np
from sklearn import svm
from WatermarkAttacker.wmattacker import VAEWMAttackerSingle
import torch

from inverse_stable_diffusion_aeon import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
import torch.nn as nn
from Imprints.src.models.bvmr import bvmr

import time
# class DeepHashNet(nn.Module):
#     def __init__(self, hash_dim=128):
#         super().__init__()
#         self.flatten_dim = 4 * 64 * 64  # Flattened input size

#         # Hash projection layer (input → hash representation)
#         self.hash_layer = nn.Linear(self.flatten_dim, self.flatten_dim)

    
#     def forward(self, x):
#         batch_size = x.shape[0]

#         # Flatten input from (B, 4, 64, 64) → (B, 4*64*64)
#         x_flattened = x.view(batch_size, -1)

#         # Compute deep hash representation
#         hash_code = self.hash_layer(x_flattened)  # Shape: [B, hash_dim]
#         # hash_code = torch.sigmoid(hash_code)  # Normalize to [0,1]
#         hash_code = torch.tanh(hash_code) 
#         reconstructed = hash_code.view(batch_size, 4, 64, 64)  # Reshape to (B, 4, 64, 64)
        
#         return reconstructed

        
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

def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=["blending_factor",'gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)
    # attack_model = VAEWMAttackerSingle(model_name="cheng2020-anchor")
    # attack_model_bvm = bvmr(model_path="/////Imprints/watermark_removal_works/BVMR/demo_coco/checkpoints/demo_coco/net_baseline_200.pth",
    #                     device=device)
    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = "" # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)  # text_embedding.dtype = torch.float16
    # opt_acond=init_latents_w = pipe.get_random_latents()
    # opt_acond = None
    # opt_acond = get_watermarking_pattern(pipe, args, device) # watermark latentc  dtype = torch.complex32（half）
    # print(opt_acond,opt_acond.shape,text_embeddings.shape)
    
    # load the optimized image watermark and prompt embedding
    hash_net = DeepHashNet()
    hash_net.load_state_dict(torch.load(args.wm_path)["hashing_model"])
    hash_net.to(device=text_embeddings.device, dtype=text_embeddings.dtype)
    opt_wm = torch.load(args.wm_path)['opt_wm'].to(device)
    opt_acond = torch.load(args.wm_path, map_location='cuda:0')['opt_acond'].to(text_embeddings.dtype)

    init_latents_w = pipe.get_random_latents()
    watermarking_mask = get_watermarking_mask(init_latents_w, args, device) 
# 35
    blending_factors = [0.6]#,0.4,0.6,0.8,0.9]
    for blending_factor in blending_factors:
        for steps in [45]:  # injection steps
            lguidance = 7.5  
            results = []
            clip_scores = []
            clip_scores_w = []
            no_w_metrics = defaultdict(list)
            w_metrics = defaultdict(list)
            psnr_metrics = []
            ssim_metrics = []
            msssim_metrics = []

            for i in tqdm(range(args.start, args.end)):
                
                seed = i + args.gen_seed
                
                # seed = np.random.randint(0, 2**32 - 1)
                current_prompt = dataset[i][prompt_key]
                        
                tester_prompt = current_prompt # assume at the detection time, the original prompt is unknown
                text_embeddings = pipe.get_text_embedding(tester_prompt)  # text_embedding.dtype = torch.float16

                set_random_seed(seed)
                init_latents_no_w = pipe.get_random_latents()

                outputs_no_w = pipe(
                                current_prompt,
                                num_images_per_prompt=args.num_images,
                                guidance_scale=args.guidance_scale,
                                num_inference_steps=5,
                                height=args.image_length,
                                width=args.image_length,
                                latents=init_latents_no_w,
                                )
                
                orig_image_no_w = outputs_no_w.images[0]
                
                img_no_w = transform_img(orig_image_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
                latents = pipe.get_image_latents(img_no_w)
                # random wm pattern
                init_latents_no_w = hash_net(init_latents_no_w)
                opt_wm = get_watermarking_pattern(pipe, args, device,image=orig_image_no_w,shape=None,latent=latents,alpha=blending_factor) # watermark latentc  dtype = torch.complex32（half）
                print(f"Adaptive watermark `patch: "
                    f"real_min={opt_wm.real.min().item():.4f}, real_max={opt_wm.real.max().item():.4f}")

                wm_tensor = opt_wm.detach().squeeze(0).cpu()
                
                for ch in range(wm_tensor.shape[0]):
                    channel_data = wm_tensor[ch]
                    # Normalize channel to [0,1]
                    c_min, c_max = channel_data.real.min(), channel_data.real.max()
                    channel_img = (channel_data.real - c_min) / (c_max - c_min + 1e-7)
                    channel_img = (channel_img.numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(channel_img, mode='L')
                    pil_img.save(f"./generated_imgs/watermark_images/wm_{i:04d}_channel{ch}.png")

                
                ### generation
                # generation without watermarking
                set_random_seed(seed)
                init_latents_no_w = pipe.get_random_latents()
                outputs_no_w = pipe(
                    current_prompt,
                    num_images_per_prompt=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    latents=init_latents_no_w,
                    # hash_net=hash_net
                    )
                orig_image_no_w = outputs_no_w.images[0]
                orig_image_no_w.save(f'generated_imgs/clean/ori-lg7.5-{i}-{steps}-test.jpg')
                
                # generation with watermarking
                if init_latents_no_w is None:
                    set_random_seed(seed)
                    init_latents_w = pipe.get_random_latents()
                else:
                    init_latents_w = copy.deepcopy(init_latents_no_w)

                # inject watermark for tree-ring
                # init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

                outputs_w = pipe(
                    current_prompt,
                    num_images_per_prompt=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    latents=init_latents_w,
                    watermarking_mask=watermarking_mask,
                    watermarking_steps=steps,
                    args = args,
                    gt_patch = opt_wm,
                    lguidance = lguidance,
                    opt_acond = opt_acond,
                    )
                
                orig_image_w = outputs_w.images[0]
                orig_image_w.save(f'generated_imgs/aeon5-30-pixel-lessabs-500c-2kn-circle/g7.5-{i}-{steps}.jpg')

                ### evaluate watermark
                psnr, ssim, msssim = eval_psnr_ssim_msssim(orig_image_no_w, orig_image_w)
                
                psnr_metrics.append(psnr)
                ssim_metrics.append(ssim)
                msssim_metrics.append(msssim)
                
                if args.reference_model is not None:
                    sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
                    w_no_sim = sims[0].item()
                    
                    w_sim = sims[1].item()
                else:
                    w_no_sim = 0
                    w_sim = 0

                clip_scores.append(w_no_sim)  # clip score
                clip_scores_w.append(w_sim)

                for attack in ['none',"forgery"]:
                # 'rotation', 'jpeg', 'cropping', 'blurring', 'noise', 'color_jitter',
                #                 "reconstruction","reconstructionBVMR","diff_attacker_60","cheng2020-anchor_3",
                #                 "bmshj2018-factorized_3","jpeg_attacker_50","rotate_90","brightness_0.5","contrast_0.5",
                #                 "Gaussian_noise","Gaussian_blur","bm3d"]:
                    latents_b_no_w = []
                    latents_b_w = []

                    # # distortion
                    # if "BVMR" in attack:
                    #     attacking_model = attack_model_bvm
                    # else:
                        
                    #     attacking_model = attack_model

                    orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, attack)
                    # reverse img without watermarking
                    img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
                    image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

                    latents_b_no_w.insert(0,image_latents_no_w)

                    reversed_latents_no_w, latents_b_no_w, noise_b_no_w = pipe.forward_diffusion(
                        latents=image_latents_no_w,
                        text_embeddings=text_embeddings,
                        guidance_scale=1.0,
                        num_inference_steps=args.test_num_inference_steps,
                        latents_b = latents_b_no_w,
                    )


                    if attack == "forgery":
                        base_dir = f"/media/NAS/USERS/shahid/watermark/semantic-forgery/WIND_out/{i}"

                        # Recursively search for the first image file (e.g., .png, .jpg)
                        image_path = None
                        for root, dirs, files in os.walk(base_dir):
                            for file in files:
                                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                                    if("attack_instance_step=20" in file):
                                        image_path = os.path.join(root, file)
                                        break
                            if image_path:
                                break
                        if image_path is None:
                            raise FileNotFoundError(f"No image found in {base_dir}")
                        orig_image_w_auged = Image.open(image_path)
                        

                    # reverse img with watermarking
                    img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
                    image_latents_w = pipe.get_image_latents(img_w, sample=False)

                    latents_b_w.insert(0,image_latents_w)

                    reversed_latents_w, latents_b_w, noise_b_w = pipe.forward_diffusion(
                        latents=image_latents_w,
                        text_embeddings=text_embeddings,
                        guidance_scale=1.0,
                        num_inference_steps=args.test_num_inference_steps,
                        latents_b = latents_b_w,
                    )

                    start_time = time.time()
                    # eval
                    no_w_metric, w_metric = eval_watermark(latents_b_no_w[steps], latents_b_w[steps], watermarking_mask, opt_wm, args)  # for X_Optimal latents

                    end_time = time.time()
                    time_taken = end_time - start_time

                    print(f"Time required for watermark evaluation: {time_taken:.6f} seconds")
                    # exit()
                    ### apply attacks
                    no_w_metrics[attack].append(no_w_metric)
                    w_metrics[attack].append(w_metric)

                    if args.with_tracking:
                        if (args.reference_model is not None) and (i < args.max_num_log_image):
                            # log images when we use reference_model
                            table.add_data(blending_factor,wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, current_prompt, no_w_metric, w_metric)
                        else:
                            table.add_data(blending_factor,None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric)

            ### for validation
            print(f'steps: {steps}, radius: {args.w_low_radius}-{args.w_up_radius}, wm_seed: {args.w_seed}, opt wi&opt wt')
            
            for attack in ['none',"forgery"]:
            # 'rotation', 'jpeg', 'cropping', 'blurring', 'noise', 'color_jitter',
            #                 "reconstruction","reconstructionBVMR","diff_attacker_60","cheng2020-anchor_3",
            #                 "bmshj2018-factorized_3","jpeg_attacker_50","rotate_90","brightness_0.5","contrast_0.5",
            #                 "Gaussian_noise","Gaussian_blur","bm3d"]:
                print(f'attack: {attack}')
                preds = no_w_metrics[attack] +  w_metrics[attack]
                t_labels = [1] * len(no_w_metrics[attack]) + [0] * len(w_metrics[attack])

                fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                acc = np.max(1 - (fpr + (1 - tpr))/2)
                low = tpr[np.where(fpr<.01)[0][-1]]

                if args.with_tracking:
                    wandb.log({'Table': table})
                    # wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                    #         'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                    #         'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
                    wandb.log({
                        'Table': table,
                        'clip_score_mean': mean(clip_scores),
                        'clip_score_std': stdev(clip_scores),
                        'w_clip_score_mean': mean(clip_scores_w),
                        'w_clip_score_std': stdev(clip_scores_w),
                        f'attack_metrics/{attack}_auc': auc,
                        f'attack_metrics/{attack}_acc': acc,
                        f'attack_metrics/{attack}_TPR@1%FPR': low,
                    })
                
                print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
                print(f'mse_mean: {mean(no_w_metrics[attack])}, w_mse_mean: {mean(w_metrics[attack])}')
            print(f'clip_score_mean: {mean(clip_scores)}')
            print(f'w_clip_score_mean: {mean(clip_scores_w)}')
            print(f'psnr: {mean(psnr_metrics)}, ssim: {mean(ssim_metrics)}, msssim: {mean(msssim_metrics)}')
            if args.with_tracking:
                    wandb.log({
                        "blending_factor":blending_factor,
                        "PSNR":{mean(psnr_metrics)},
                        "SSIM":{mean(ssim_metrics)},
                        "MSSIM":{mean(msssim_metrics)}
                    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--wm_path', default='ckpts_adaptive')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)  # 999999
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_up_radius', default=30, type=int)  # 10
    parser.add_argument('--w_low_radius', default=5, type=int)  # 10
    parser.add_argument('--w_radius', default=15, type=int)  # 10
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=1, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)