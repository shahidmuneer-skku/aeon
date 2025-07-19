# [WACV2026] AEON: Adaptive Embedding Optimized Noise for Robust Watermarking in Diffusion Models

This repository contains the official PyTorch implementation of our **AEON** paper:  
# 🌊 **"AEON: Adaptive Embedding Optimized Noise for Robust Watermarking in Diffusion Models"**  

---

> **Abstract:**  
*The widespread use of synthetic image generation models and the challenges associated with authenticity preservation have fueled the demand for robust watermarking methods to safeguard authenticity and protect the copyright of synthetic images. Existing watermarking methods embed. Invisible signatures in synthetic images often compromise image quality and remain susceptible to multiple watermark removal attacks, including reconstruction and forgery methods. To overcome this issue, we propose a novel watermarking approach,~\SystemName, which seamlessly integrates the watermark into the latent diffusion process and ensures the watermark aligns with scene semantics in the final image. Unlike existing invisible in-diffusion watermarking and traditional hash-based methods, our approach adapts the neural synthesized hash-based watermark to the semantics of the generated image during the intermediate diffusion process instead of embedding traditional hashes with the initial noise. This facilitates visual coherence in the generated image while enhancing adversarial robustness and resilience against single or multiple adversarial and traditional watermark removal attacks. Our proposed approach a) modulates the noise sampling in each diffusion denoising iteration through a learnable watermark embedding, b) optimizes consistency, reconstruction, and similarity loss, enforcing local and global alignment between the watermark structure and the underlying image content, and c) generates a strong watermark by allowing late embedding of the watermark in the diffusion process. Empirical results demonstrate the effectiveness of the proposed approach in retaining quality and its robustness against cumulative adversarial attacks.*

---

## ✨ Key Features
- **Semantic, adaptive watermarking.** AEON injects a *learnable* hash‑based noise into the latent at run‑time so the watermark aligns with the scene’s semantics while remaining invisible.  
- **Hash‑network‑driven noise generation.** A dedicated network produces frequency‑domain watermark noise that stays close to the latent structure, minimising quality loss.  
- **Late‑stage injection without artifacts.** Because the watermark is adaptive, it can be embedded as late as step 40, yielding PSNR ≈ 25.9 dB with < 1 % AUC drop after cumulative attacks.  
- **State‑of‑the‑art robustness.** Across blur, noise, JPEG, crop, rotation and reconstruction attacks, AEON achieves an average AUC of 0.994 on Stable Diffusion—outperforming Tree‑Ring, ROBIN, WIND and others.  
- **Straight‑through training end‑to‑end.** The binary hash layer is trained with a straight‑through estimator, so gradients flow through the discretisation step (details in Section 3 of the paper).  
- **Easy verification.** A reverse DDIM pass to *t*<sub>inject</sub> plus an FFT on selected bins delivers a one‑shot match/no‑match decision.

---

## 🛠️ Architecture at a Glance

---



Stage 1  (Training)   : Hash network learns semantic noise in f ‑space.
Stage 2  (Generation) : Adaptive hash blended into latent via FFT/IFFT.
Stage 3  (Verification): Reverse DDIM + FFT recovers watermark bits.

AEON first trains a hashing network on intermediate noise, then blends its output into the latent via FFT/IFFT, and finally verifies by partially reversing the diffusion.

<p align="center">
  <img src="assets/aeon_pipeline.png" width="650" alt="AEON three‑stage pipeline">
</p>

---

## 📁 Dataset & Experimental Setup

| Purpose              | Details                                                                             |
|----------------------|-------------------------------------------------------------------------------------|
| Hash‑net training    | **MS‑COCO** images + 1 000 captions from Gustavosta’s *Stable‑Diffusion‑Prompts*    |
| Diffusion backbone   | Stable Diffusion v2.1 (50 DDIM steps, guidance 7.5)                                 |
| Hardware             | 1× NVIDIA A100 40 GB                                                                |
| Hyper‑parameters     | LR 5e‑4 · batch 1 · 2 000 steps · α 0.9 · γ 0.4                                     |

---

## 🚀 Training / Inference Pipeline

### 1 . Train the Hashing Network
```bash
bash train_aeon.sh


## 🚀 Injection Pipeline
bash inject_adaptive.sh


### 3. Inference
```bash
bash inference.sh

```
## 📊 Headline Results (Stable Diffusion)

| Attack | Clean | Blur | Noise | JPEG | Rotation | Crop | **Avg** |
|--------|------:|-----:|------:|-----:|---------:|-----:|--------:|
| **AUC** | 1.000 | 1.000 | 0.986 | 1.000 | 0.985 | 0.998 | **0.994** |

| Model      | PSNR ↑ | SSIM ↑ | MSSIM ↑ | FID ↓ |
|------------|-------:|-------:|--------:|------:|
| Tree‑Ring  | 15.37  | 0.568  | 0.626   | 25.93 |
| ROBIN      | 24.03  | 0.768  | 0.881   | 26.86 |
| **AEON**   | **25.93** | **0.812** | **0.918** | **26.61** |

---

## 🔬 Ablation — Effect of *t*<sub>inject</sub>

| Step | PSNR ↑ | AUC (post‑attack) ↑ |
|------|-------:|--------------------:|
| 10   | 19.63  | 0.665 |
| 20   | 20.96  | 0.762 |
| 35   | 24.35  | 0.981 |
| **40** | **25.93** | **0.991** |
