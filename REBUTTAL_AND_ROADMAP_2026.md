# HRHumanTex: Expert Rebuttal & 2026 Improvement Roadmap

**Author:** Expert Review Panel (Visual Computing: Graphics + Vision + Multi-modal GenAI)
**Date:** 2026-05-30
**Target:** Yunfan Liu's MSc Thesis — *High-Resolution Human Texture Estimation from a Single Image*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Remains Valuable & Relevant (2026)](#2-what-remains-valuable--relevant-2026)
3. [What Has Become Outdated](#3-what-has-become-outdated)
4. [Scientific Rebuttal: Strengths & Weaknesses](#4-scientific-rebuttal-strengths--weaknesses)
5. [The 2026 Landscape: What Changed Since 2025](#5-the-2026-landscape-what-changed-since-2025)
6. [Improvement Roadmap: v1 → v2 Architecture](#6-improvement-roadmap-v1--v2-architecture)
7. [Code-Level Action Items](#7-code-level-action-items)
8. [Open-Sourcing Strategy](#8-open-sourcing-strategy)
9. [References](#9-references)

---

## 1. Executive Summary

Yunfan Liu's MSc thesis presents **HRHumanTex**, a framework that extends SMPLitex to generate 1024×1024 UV textures from a single RGB image via a pipeline combining DensePose-based partial texture extraction, super-resolution data augmentation (7 SR models benchmarked), DreamBooth fine-tuning of Stable Diffusion v1.5, and a comprehensive multi-level evaluation suite including rendering-based metrics.

**Bottom-line verdict:** The core **idea** — scaling up UV texture resolution via SR-augmented data and diffusion inpainting — is **sound and remains relevant**. The **systematic benchmarking** of 7 SR models on UV texture data is a genuine contribution. However, several **technology choices** are now dated (SD1.5, full DreamBooth, AUTOMATIC1111 WebUI inference, CLIP encoder, PyTorch3D-only rendering). The thesis would benefit significantly from a modernization pass before open-sourcing. **The work is still worth open-sourcing**, but I recommend the "push original → then modernize through commits" strategy, with clear documentation of what's legacy vs. what's updated.

---

## 2. What Remains Valuable & Relevant (2026)

### 2.1 Core Pipeline Architecture (⭐⭐⭐⭐⭐)

The three-stage architecture is well-designed and timeless:
1. **Partial texture extraction** (DensePose + SGHM)
2. **SR-enhanced data augmentation** (512→1024)
3. **Diffusion-based texture inpainting** (DreamBooth fine-tuned)

This pipeline composition remains the dominant paradigm in 2026. The recent works like CHROME (ICCV 2025), HumanRef-GS (2025), and MVD-HuGaS (2026) all follow similar multi-stage designs. **Your pipeline structure is not outdated.**

### 2.2 Systematic SR Benchmarking on UV Textures (⭐⭐⭐⭐⭐)

The comparative evaluation of 7 SR models (RCAN, RealESRGAN, BSRGAN, SwinIR, SRFormer, StableSR, DiffBIR) specifically on **UV texture data** is a genuine contribution. Your finding that:

- **RCAN** (CNN-based) wins at 256→512 on UV textures
- **SRFormer** (Transformer) wins at 512→1024
- **DiffBIR** (Diffusion) scores worst in UV-space but best in rendering

...is an **important and non-obvious result**. In 2026, this insight about the **mismatch between UV-space metrics and rendering quality** is even more relevant as diffusion-based SR models (SUPIR, FoundIR-v2) gain popularity. You were early to observe this phenomenon.

### 2.3 Rendering-Based Evaluation Suite (⭐⭐⭐⭐)

Your multi-level evaluation (UV-space metrics + rendering-based SSIM/LPIPS via PyTorch3D) anticipates what has now become standard practice. The 2025-2026 literature (CHROME, HumanRef-GS, SyncHuman) all emphasize rendering-based evaluation over pure UV-space metrics. **You were ahead of the curve here.**

### 2.4 Differentiable Rendering Texture Optimization (⭐⭐⭐⭐)

The exploratory experiment in Section 4.7 (using PyTorch3D's differentiable renderer to optimize UV textures against reference images) is a **strong idea** that has gained massive traction in 2025-2026. The Score Distillation Sampling (SDS) paradigm and its successors are doing exactly this at scale. Your early experimentation with direct UV texture optimization via differentiable rendering was prescient.

### 2.5 Ablation Study Methodology (⭐⭐⭐⭐)

The two ablation studies:
- **Ablation A:** Dataset scale (20→50→100→266 samples)
- **Ablation B:** DreamBooth hyperparameters (SDv1.4 vs v1.5, steps, resolution, gradient accumulation)

...demonstrate solid experimental methodology. The finding that ~100 samples is the critical threshold for coherent full-body 1024px completion is a **practical insight** with lasting value.

### 2.6 Data-Scarcity Focus (⭐⭐⭐⭐)

Your focus on making things work with limited data (266 textures vs. TexDreamer's 50K) is **more relevant than ever**. The 2025-2026 trend is toward data-efficient methods (LoRA instead of full fine-tuning, few-shot personalization). Your work on achieving quality results with small datasets is a strength, not a weakness.

---

## 3. What Has Become Outdated

### 3.1 Stable Diffusion v1.5 (⭐⭐⭐⭐⭐ Critical Update Needed)

**Status in 2026:** SD1.5 is end-of-life. The community has moved on.

- **SDXL (2023-2024):** Native 1024×1024 generation, better text understanding, still widely used.
- **SD3/3.5 (2024-2025):** MMDiT architecture, improved typography and composition, better multi-subject handling.
- **Flux.1 (2024-2025):** From Black Forest Labs (original SD authors). Flow-matching, SOTA image quality at 1024+, the new default for fine-tuning.
- **Flux.1-Fill (2025):** The dedicated inpainting variant — this is the **direct modern replacement** for your SD1.5 inpainting model.

**Impact:** Your model cannot compete visually with fine-tuned Flux or SD3.5 models in 2026. The community will immediately ask "why SD1.5?"

**Recommendation:** Keep the SD1.5 version as "HRHumanTex-v1 (Legacy)" and add a Flux.1-Fill version as "HRHumanTex-v2."

### 3.2 Full DreamBooth Fine-Tuning (⭐⭐⭐⭐ Critical Update Needed)

**Status in 2026:** Full DreamBooth is obsolete for most use cases. LoRA is the standard.

- **LoRA (Low-Rank Adaptation)** trains <1% of parameters, produces MB-sized weights instead of GB-sized models
- **DiffuseKronA (WACV 2025)** achieves 99.947% parameter reduction vs. full DreamBooth with comparable quality
- **DoRA (2025)** further improves LoRA with weight-decomposed updates

**Impact:** Your models are ~4GB each, requiring full model distribution. LoRA adapters would be ~10-50MB each, making sharing trivial.

**Recommendation:** Train LoRA adapters for each SR variant. The LoRA weights can sit alongside the original DreamBooth checkpoints.

### 3.3 AUTOMATIC1111 WebUI as Inference Backend (⭐⭐⭐⭐ Critical Update Needed)

**Status in 2026:** A1111 is still popular but declining. The API-based inference with a separate running server is unnecessarily complex.

Modern alternatives:
- **ComfyUI:** The dominant node-based workflow engine in 2026. Better for reproducibility.
- **diffusers library (HuggingFace):** Direct Python API, no server needed. The `StableDiffusionInpaintPipeline` and `FluxInpaintPipeline` are production-ready.
- **Forge/reForge:** Faster inference with lower VRAM.

**Impact:** Requiring users to install WebUI just for inference is a major barrier to adoption. The `webui.sh` setup with separate user accounts is fragile and user-hostile.

**Recommendation:** Rewrite inference using `diffusers` library directly. No WebUI dependency.

### 3.4 CLIP-Based Text Encoder (⭐⭐⭐ Important Update)

**Status in 2026:** CLIP is legacy. SigLIP2 and DINOv3 are superior.

- **SigLIP2** is the default vision encoder for all major VLMs in 2026 (Qwen3-VL, Gemma 3)
- **DINOv3** provides superior dense features for spatial understanding
- **C-RADIOv4 (NVIDIA, 2025)** distills SigLIP2 + DINOv3 + SAM3 into one backbone

**Impact on your work:** This affects both the DreamBooth text conditioning quality and any LPIPS/perceptual metric computation. LPIPS computed with AlexNet features is itself becoming dated.

**Recommendation:**
- Short-term: Keep CLIP for compatibility, note it as legacy
- Medium-term: Add SigLIP2-based text conditioning via a LoRA adapter on Flux
- Long-term: Explore multi-encoder fusion for conditioning

### 3.5 PyTorch3D as Sole Differentiable Renderer (⭐⭐⭐ Important Update)

**Status in 2026:** PyTorch3D is still functional but alternatives have emerged:

- **3DGRUT (NVIDIA, 2025-2026):** CUDA-optimized, supports 3DGS, much faster
- **DiffSoup (CVPR 2026):** Extreme triangle simplification with differentiable binary opacity
- **nvdiffrast:** Still the performance leader for mesh rasterization

**Impact:** Your rendering-based evaluation and optimization are bottlenecked by PyTorch3D's speed.

**Recommendation:** Add an nvdiffrast backend option for faster rendering evaluation.

### 3.6 DensePose + SGHM as Partial Texture Extraction (⭐⭐ Minor Update)

**Status in 2026:** Both still work, but there are newer alternatives:

- **DensePose** (2019) is still used but has not been updated significantly
- **Sapiens (Meta, 2025)** provides much higher quality human parsing
- **SCHP (2024)** offers real-time human parsing with better accuracy
- **SMPLer-X (2024)** and **4D-Humans (2024)** provide better pose+shape estimation

**Impact:** Your partial texture quality ceiling is set by DensePose/SGHM quality.

**Recommendation:** Replace DensePose+SGHM with Sapiens-based human parsing. Keep DensePose as a fallback option.

### 3.7 ROMP for Pose Estimation (⭐⭐ Minor Update)

**Status in 2026:** ROMP (2021) is now superseded by:
- **SMPLer-X (2024):** Better accuracy, handles challenging poses
- **HMR 2.0 (2023):** Foundation model for human mesh recovery
- **Multi-HMR (2024):** Handles multiple people better

### 3.8 SR Model Selection (⭐⭐ Minor Update)

Your 7 SR models were SOTA in 2023-2024. In 2026:
- **SUPIR (2024-2025)** is the open-source SOTA for diffusion-based SR
- **FoundIR-v2 (2025)** is a unified restoration model covering 50+ tasks
- **Latent Upscaler Adapter (LUA, Nov 2025)** is 3× faster with comparable quality
- **MewZoom, UltraZoom** are new lightweight options

---

## 4. Scientific Rebuttal: Strengths & Weaknesses

### 4.1 Strengths

#### S1: Solid Engineering with Clear Decisions
Every engineering choice is justified with ablation experiments. The transition from SD1.4→SD1.5, 1500→2000 steps, 2→8 gradient accumulation, each backed by evidence. This is the mark of rigorous applied research.

#### S2: Multi-Dimensional Evaluation
UV-space metrics + rendering-based metrics + qualitative comparisons + failure case analysis. Few contemporary theses at this level provide such comprehensive evaluation. The scatter plots showing the UV-space vs. render-space performance trade-off are particularly insightful.

#### S3: Honest Failure Analysis
The appendix's systematic documentation of 6 failure categories (non-frontal faces, side faces, makeup confusion, garment length interference, etc.) demonstrates scientific integrity. Many papers hide their failures; you documented them and analyzed which models handle which failures best.

#### S4: Practical Data Efficiency Finding
The ~100-sample threshold for coherent 1024px completion is a genuine practical insight. This finding has implications beyond your specific pipeline.

#### S5: The SR-UV Texture Mismatch Insight
The observation that diffusion-based SR models (DiffBIR, StableSR) perform poorly on UV-space metrics but well on rendering quality is a **genuinely interesting scientific finding** that deserves more attention. It suggests something fundamental about the difference between texture-space and image-space quality assessment.

### 4.2 Weaknesses & Critiques

#### W1: No Comparison with TexDreamer (Major Gap)
TexDreamer (Liu et al., 2024) is the closest contemporary work — also achieving 1024² UV textures with diffusion models. Your thesis mentions TexDreamer in related work but **never compares against it quantitatively**. This is the single biggest omission.

**Rebuttal expectation:** A reviewer would demand HRHumanTex vs. TexDreamer on at least one shared dataset.

**Suggested fix:** Add TexDreamer comparison using their pretrained model on your DeepFashion/SHHQ test sets.

#### W2: The "Pseudo-GT" Problem (Conceptual Weakness)
Your evaluation chain uses:
1. SRFormer-upscaled textures as "pseudo-ground-truth" for UV-space metrics
2. RCAN-upscaled textures as "pseudo-ground-truth" for SR model comparison

This creates a **circular evaluation**: SRFormer-pretrained data is evaluated against SRFormer-generated pseudo-GT, which may favor SRFormer-trained models. You acknowledge this in passing but don't fully address the bias.

**Suggested fix:**
- Add evaluation against **real 1024px data** (e.g., from TexDreamer's ATLAS dataset, or render high-quality multi-view captures)
- Report the correlation between pseudo-GT metrics and a small set of human preference judgments
- Add a "no-reference" metric (e.g., CLIP-IQA, MANIQA) that doesn't depend on pseudo-GT

#### W3: Single-View Limitation Not Addressed
The entire pipeline works from a single RGB image. For visible regions, this works well. For occluded regions (back, sides), the model hallucinates. You acknowledge this as a limitation but don't propose solutions.

**Suggested fix:**
- Add a multi-view fusion module (even if only in discussion)
- Explore video-based texture accumulation
- Use multi-view diffusion (like MVD-HuGaS, 2026) to generate synthetic back views before texture projection

#### W4: No Ablation on Conditioning Strategy
Your inpainting is conditioned on a simple text prompt ("a sks texturemap") + partial UV image. There's no exploration of:
- Spatial conditioning (e.g., body part labels as additional channel)
- Pose conditioning (SMPL pose parameters as cross-attention input)
- Reference image conditioning (CLIP image embedding as additional condition)

**Suggested fix:** Add ControlNet-based body part conditioning or IP-Adapter reference image conditioning.

#### W5: Rendering Evaluation Uses Simplified Illumination
Your rendering evaluation uses only ambient white light, no specular highlights. This makes rendered images look flat and may not reflect real-world appearance.

**Suggested fix:** Use PBR (Physically Based Rendering) lighting with environment maps for more realistic evaluation.

#### W6: Limited Dataset Diversity
- DeepFashion: Clothing-focused, mostly Asian models
- SHHQ: More diverse but synthetic (StyleGAN-generated)
- No real-world diverse dataset (e.g., Flickr, COCO people)

**Suggested fix:** Add evaluation on UBC-Fashion, Market-1501, or a custom diverse test set.

#### W7: No User Study
All evaluation is quantitative. For a task where "visual quality" is the goal, a perceptual user study (even small-scale, e.g., 20 participants, A/B comparisons) would significantly strengthen the claims.

---

## 5. The 2026 Landscape: What Changed Since 2025

### 5.1 Diffusion Models

| Component | Your (2024-2025) | 2026 Standard |
|-----------|-----------------|---------------|
| Base Model | SD1.5 | Flux.1 / SD3.5 |
| Fine-Tuning | Full DreamBooth | LoRA / DoRA / DiffuseKronA |
| Inference | A1111 WebUI API | diffusers / ComfyUI |
| Text Encoder | CLIP ViT-L/14 | SigLIP2 / T5-XXL (Flux) |
| Inpainting | SD1.5 Inpainting | Flux.1-Fill / SD3.5 Inpaint |
| Resolution | 1024² (stretched from 512² native) | Native 1024²+ (Flux, SDXL) |

### 5.2 3D Human Reconstruction

| Component | Your (2024-2025) | 2026 Standard |
|-----------|-----------------|---------------|
| Geometry | SMPL-only | SMPL + 3DGS (CHROME, DcSplat, HumanSplatHMR) |
| Multi-View | Not used | Diffusion-based pseudo-multi-view (MVD-HuGaS) |
| Pose Estimation | ROMP | SMPLer-X / HMR 2.0 / Multi-HMR |
| Human Parsing | DensePose+SGHM | Sapiens / SCHP |
| Differentiable Rendering | PyTorch3D | 3DGRUT / nvdiffrast / DiffSoup |

### 5.3 Super-Resolution

| Component | Your (2024-2025) | 2026 Standard |
|-----------|-----------------|---------------|
| Best SR for UV (your finding) | RCAN (256→512), SRFormer (512→1024) | SUPIR (quality), LUA (speed) |
| Number of SR models benchmarked | 7 (comprehensive!) | Additional: SUPIR, FoundIR-v2, MewZoom |
| Diffusion-based SR | DiffBIR, StableSR (poor on UV) | SUPIR (much better, SDXL-based) |

### 5.4 Key New Papers to Cite (2025-2026)

1. **CHROME** (ICCV 2025) — Single-image clothed human reconstruction with occlusion resilience via 3DGS
2. **HumanRef-GS** (IEEE TCSVT 2025) — Reference-guided diffusion + 3DGS for image-to-3D human
3. **SyncHuman** (2025) — Synchronizing 2D and 3D generative models for single-view human reconstruction
4. **MVD-HuGaS** (arXiv 2026) — Multi-view human diffusion + 3DGS
5. **DcSplat** (AAAI 2026) — Dual-constraint human 3DGS
6. **SUPIR** (2024/2025) — SDXL-based SOTA image restoration/SR
7. **FoundIR-v2** (2025) — Unified diffusion-based restoration
8. **Flux.1** (Black Forest Labs, 2024-2025) — Flow-matching image generation
9. **DiffuseKronA** (WACV 2025) — Parameter-efficient diffusion fine-tuning
10. **Sapiens** (Meta, 2025) — Foundation model for human-centric vision tasks
11. **PromptAvatar** (arXiv 2026) — Multi-modal guided 3D avatar generation
12. **InfiniHuman** (SIGGRAPH Asia 2025) — Controllable photorealistic 3D human creation

---

## 6. Improvement Roadmap: v1 → v2 Architecture

### Phase 1: Immediate (1-2 weeks, before public release)

**Goal:** Clean up the existing codebase, fix obvious issues, add documentation.

- [ ] Add proper `README.md` with installation instructions, model downloads, quickstart
- [ ] Add `requirements.txt` with pinned versions
- [ ] Remove hardcoded absolute paths (e.g., `/workspace/HRHumanTex/`)
- [ ] Add a `config.yaml` for all paths and parameters
- [ ] Add a Colab notebook for easy reproducibility
- [ ] Add license file (MIT or Apache 2.0 recommended)
- [ ] Add BibTeX citation entry
- [ ] Re-run evaluation and include result CSV files in the repo

### Phase 2: Short-term (2-4 weeks, first round of commits)

**Goal:** Modernize the most outdated components while keeping backward compatibility.

- [ ] **Replace A1111 WebUI inference with diffusers**: Write a new `inpaint_with_diffusers.py` that uses `diffusers.StableDiffusionInpaintPipeline` directly
- [ ] **Add LoRA training option**: Add a `train_lora.py` script using `peft` library. Train LoRA weights for the best-performing variants (SwinIR, DiffBIR)
- [ ] **Replace ROMP with SMPLer-X or HMR 2.0**: For the rendering evaluation pipeline
- [ ] **Add Sapiens-based human parsing**: As alternative to DensePose+SGHM
- [ ] **Add TexDreamer comparison**: Quantitative comparison with pretrained TexDreamer
- [ ] **Add no-reference metrics**: CLIP-IQA, MUSIQ, MANIQA for pseudo-GT-independent evaluation

### Phase 3: Medium-term (1-2 months, major upgrade)

**Goal:** Architectural improvements that genuinely advance the state of the art.

- [ ] **HRHumanTex-v2 with Flux.1-Fill**: Fine-tune Flux.1-Fill (LoRA) on your UV texture datasets
  - Native 1024² generation quality
  - Better text conditioning via T5-XXL
  - Flow-matching instead of DDPM = fewer inference steps
- [ ] **ControlNet-based body part conditioning**: Train a lightweight ControlNet on SMPL body part segmentation maps to guide inpainting
- [ ] **IP-Adapter for reference image conditioning**: Use the input image's CLIP/SigLIP embedding as additional conditioning for texture inpainting
- [ ] **Multi-view texture refinement**: Use a multi-view diffusion model to generate synthetic back/side views, then project all views to UV space and fuse
- [ ] **nvdiffrast rendering backend**: For faster rendering-based evaluation and optimization
- [ ] **Texture optimization with SDS**: Replace the L2+LPIPS+VGG loss with Score Distillation Sampling from a pretrained diffusion model for higher-quality texture refinement

### Phase 4: Long-term (3-6 months, research contributions)

**Goal:** Novel research directions that could lead to publications.

- [ ] **HRHumanTex + 3DGS**: Instead of SMPL-only rendering, reconstruct a 3D Gaussian Splatting avatar with your estimated texture, enabling photorealistic novel-view synthesis
- [ ] **Video-based temporal texture fusion**: Accumulate partial textures across video frames for more complete coverage
- [ ] **Multi-modal conditioning**: Text description + reference image → complete UV texture
- [ ] **Interactive texture editing**: Allow users to edit UV texture regions with text prompts
- [ ] **Garment-specific texture estimation**: Separate upper/lower/full-body garment textures with proper layering (inspired by DAMA, 2026)

---

## 7. Code-Level Action Items

### 7.1 Critical Fixes (Before Public Release)

#### Remove Hardcoded Paths

```python
# BEFORE (BAD):
dataset_obj = load_obj("/workspace/HRHumanTex/sample-data/smpl_uv_20200910/smpl_uv.obj")
smpl = SMPL(model_path="/workspace/HRHumanTex/sample-data/SMPL/models/smpl", gender='neutral')

# AFTER (GOOD):
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SMPL_UV_OBJ = os.path.join(PROJECT_ROOT, "sample-data", "smpl_uv_20200910", "smpl_uv.obj")
SMPL_MODEL_DIR = os.path.join(PROJECT_ROOT, "sample-data", "SMPL", "models", "smpl")

dataset_obj = load_obj(SMPL_UV_OBJ)
smpl = SMPL(model_path=SMPL_MODEL_DIR, gender='neutral')
```

#### Fix the evaluate_texture_batch.py Bug

Line 193-197 has a bug — `sil_mask` loads from `dp_p` instead of `sil_p`:

```python
# BEFORE (BUG):
dp_mask = np.array(
    Image.open(dp_p).convert('L').resize((1024,1024), Image.NEAREST))
sil_mask = np.array(
    Image.open(dp_p).convert('L').resize((1024,1024), Image.NEAREST))  # BUG: loads dp_p again!

# AFTER (FIX):
dp_mask = np.array(
    Image.open(dp_p).convert('L').resize((1024,1024), Image.NEAREST))
sil_mask = np.array(
    Image.open(sil_p).convert('L').resize((1024,1024), Image.NEAREST))  # FIXED: loads sil_p
```

#### Fix the Mask Logic

The mask computation on line 199 has a subtle issue — the DensePose mask and silhouette mask may have different value ranges:

```python
# BEFORE:
mask = ((np.array(dp_mask)>0) & (np.array(sil_mask)>0)).astype(np.uint8)

# AFTER (more robust):
dp_binary = (dp_mask > 128).astype(np.uint8)
sil_binary = (sil_mask > 128).astype(np.uint8)
mask = (dp_binary & sil_binary).astype(np.uint8)
```

### 7.2 Architecture Improvements

#### New: inpaint_with_diffusers.py (replaces inpaint_with_A1111.py)

```python
"""
Modern inference using diffusers library (no WebUI dependency).
Supports SD1.5, SDXL, and Flux.1-Fill backends.
"""
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

class InpaintWithDiffusers:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        # Enable memory-efficient attention
        self.pipe.enable_xformers_memory_efficient_attention()
    
    def inpaint(self, image: Image, mask: Image, 
                prompt: str = "a sks texturemap",
                num_images: int = 8,
                cfg_scale: float = 2.0,
                denoising_strength: float = 0.8,
                width: int = 1024, height: int = 1024) -> list[Image]:
        """Run inpainting without a running WebUI server."""
        results = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_images_per_prompt=num_images,
            guidance_scale=cfg_scale,
            strength=denoising_strength,
            width=width,
            height=height,
        )
        return results.images
```

#### New: train_lora.py (replaces train_dreambooth.py for lightweight training)

```python
"""
LoRA-based DreamBooth training — ~50MB outputs instead of ~4GB.
"""
from diffusers import StableDiffusionInpaintPipeline
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1,
)

# Apply LoRA to UNet
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.unet = get_peft_model(pipe.unet, lora_config)

# Train only LoRA weights (~13M params vs ~860M for full UNet)
# ... (training loop similar to DreamBooth but with peft)
pipe.unet.save_pretrained("hrhumantex-swinir-lora")  # ~50MB output
```

### 7.3 Evaluation Improvements

#### Add No-Reference Metrics

```python
import torch
from pyiqa import create_metric

# No-reference metrics that don't need pseudo-GT
clip_iqa = create_metric('clipiqa').cuda()
musiq = create_metric('musiq').cuda()
maniqa = create_metric('maniqa').cuda()

# Use instead of or in addition to PSNR/SSIM vs. pseudo-GT
score_clip = clip_iqa(rendered_image_tensor)
score_musiq = musiq(rendered_image_tensor)
score_maniqa = maniqa(rendered_image_tensor)
```

### 7.4 New Dependencies

```txt
# Add to requirements.txt:
diffusers>=0.31.0
peft>=0.13.0
xformers>=0.0.27
pyiqa>=0.1.12
nvdiffrast  # optional, for faster rendering
sapiens  # optional, for better human parsing
```

---

## 8. Open-Sourcing Strategy

### 8.1 Release Plan

```
Week 1-2:  Push "v1.0-legacy" — the original code as-is, with a clear README
           stating this is the historical version from the MSc thesis (2025).
           Fix hardcoded paths, add requirements.txt, add license.

Week 3-6:  Commit Series A — Modernization without architecture changes
           - Replace A1111 with diffusers
           - Add LoRA training
           - Fix bugs (evaluate_texture_batch.py mask bug)
           - Add config.yaml
           - Add Colab notebook

Week 7-12: Commit Series B — New features
           - Flux.1-Fill fine-tuning (LoRA)
           - ControlNet body part conditioning
           - IP-Adapter reference conditioning
           - Sapiens-based human parsing
           - nvdiffrast rendering backend

Week 13-16: Commit Series C — Research extensions
           - Multi-view texture fusion
           - SDS-based texture optimization
           - Perceptual user study
           - TexDreamer comparison
```

### 8.2 Repository Structure (Recommended)

```
HRHumanTex/
├── README.md
├── LICENSE
├── CITATION.bib
├── requirements.txt
├── config.yaml
├── hr_human_tex/
│   ├── __init__.py
│   ├── partial_texture.py      # DensePose + SGHM → partial UV
│   ├── inpainting/
│   │   ├── __init__.py
│   │   ├── diffusers_backend.py  # New: diffusers-based inference
│   │   ├── a1111_backend.py      # Original: WebUI API (legacy)
│   │   └── flux_backend.py      # New: Flux.1-Fill (HRHumanTex-v2)
│   ├── super_resolution/
│   │   ├── __init__.py
│   │   ├── evaluate_sr.py
│   │   └── sr_models.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_dreambooth.py   # Original (legacy)
│   │   └── train_lora.py         # New: LoRA training
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_texture.py
│   │   ├── evaluate_rendering.py
│   │   └── no_reference_metrics.py  # New
│   └── optimization/
│       ├── __init__.py
│       └── optimize_uv_texture.py
├── scripts/
│   ├── colab_demo.ipynb
│   └── download_models.sh
├── sample-data/
│   └── smpl_uv_20200910/
├── models/                        # gitignored, download via script
│   ├── README.md                  # Links to all model downloads
│   └── ...
├── results/                       # Evaluation results and CSVs
│   └── ...
└── docs/
    ├── PIPELINE.md                # Detailed pipeline documentation
    └── paper_figures/             # Reproduced figures
```

### 8.3 Community Engagement

- **HuggingFace Spaces demo:** Deploy a Gradio app showing the texture estimation pipeline
- **Weights & Biases report:** Share the SR benchmarking results interactively
- **Twitter/X thread:** Visual thread showing before/after, key findings, and the pipeline animation
- **arXiv update:** If allowed, update the thesis citation on arXiv/Semantic Scholar with the GitHub link

### 8.4 Messaging/Narrative

When open-sourcing, frame the work as:

> **HRHumanTex** is a systematic study of super-resolution for diffusion-based human texture inpainting. Released as both a historical artifact (v1, SD1.5-based, 2025) and an actively maintained modernization (v2, Flux-based, 2026). The SR benchmarking findings (7 models on UV texture data) and the UV-space vs. render-space quality mismatch remain underexplored in the literature.

This narrative:
1. Acknowledges the dated technology (SD1.5) upfront → builds trust
2. Highlights the timeless contributions (SR benchmarking, UV vs. render quality gap)
3. Shows active maintenance → encourages community engagement

---

## 9. References

### Key Papers Since Your Thesis (2025-2026)

| # | Paper | Venue | Relevance |
|---|-------|-------|-----------|
| 1 | CHROME: Clothed Human Reconstruction with Occlusion-Resilience | ICCV 2025 | Single-image human with occlusion handling |
| 2 | HumanRef-GS: Image-to-3D Human Generation with Reference-Guided Diffusion | IEEE TCSVT 2025 | Reference-guided 3D human generation |
| 3 | MVD-HuGaS: Free-view 3D Human Rendering from a Single Image | arXiv 2026.03 | Multi-view diffusion + 3DGS |
| 4 | DcSplat: Dual-Constraint Human Gaussian Splatting | AAAI 2026 | Efficient single-view 3DGS |
| 5 | SyncHuman: Synchronizing 2D and 3D Generative Models | 2025 | 2D+3D generative fusion |
| 6 | DAMA: Disentangled Body-Anchored Gaussians | arXiv 2026.05 | Multi-layered garment avatars |
| 7 | CrowdGaussian: Reconstructing 3D Gaussians for Human Crowds | CVPR 2026 | Multi-person reconstruction |
| 8 | HumanSplatHMR: Closing the Loop HMR + Gaussian Splatting | arXiv 2026.05 | Joint pose+rendering optimization |
| 9 | SUPIR: Practicing Model Scaling for Photo-Realistic Image Restoration | CVPR 2024 | SOTA SR (SDXL-based) |
| 10 | FoundIR-v2: Diffusion-Based Image Restoration | 2025 | Unified restoration model |
| 11 | DiffuseKronA: Parameter Efficient Fine-tuning | WACV 2025 | 99.9% parameter reduction vs DreamBooth |
| 12 | DiffSoup: Direct Differentiable Rasterization | CVPR 2026 | Fast triangle differentiable rendering |
| 13 | PromptAvatar: Multi-modal Guided 3D Avatar Generation | arXiv 2026.03 | Text+image→3D avatar |
| 14 | InfiniHuman: Controllable Photorealistic 3D Human Creation | SIGGRAPH Asia 2025 | 111K identities dataset |
| 15 | Sapiens: Foundation for Human Vision Models | ECCV 2024/2025 | Better human parsing than DensePose+SGHM |
| 16 | SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation | NeurIPS 2023/2024 | Better pose estimation than ROMP |
| 17 | Flux.1 / Flux.1-Fill | Black Forest Labs 2024-2025 | Modern diffusion backbone |
| 18 | SigLIP2 / C-RADIOv4 | 2025 | CLIP replacements |

---

## Final Verdict

**Should you open-source this work?** ✅ **Yes, absolutely.**

The core contributions — systematic SR benchmarking on UV textures, the UV-space vs. render-space quality mismatch insight, the multi-level evaluation framework, and the data-efficiency findings — remain scientifically valuable in 2026. The engineering pipeline, while built on dated components, demonstrates a complete and functional system, which is rare and valuable.

**The "push original → iterate publicly" strategy is the right one.** It shows the research journey honestly: where the ideas started, what was state-of-the-art in 2024-2025, and how the work evolves with the field. This narrative of continuous improvement is more compelling than a polished-but-static release.

**Your biggest opportunity:** The SR-on-UV-texture benchmarking study is currently **unique in the literature**. No other paper has systematically compared 7 SR models specifically on UV texture data. If you extend this to include the 2025-2026 models (SUPIR, FoundIR-v2, LUA) and write it up, this could be a **standalone short paper** at a workshop like AI for 3D Generation (CVPR/ICCV workshop) or Eurographics.

Good luck with the open-source release! 🚀
