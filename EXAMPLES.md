# 📖 Usage Examples

Comprehensive examples for all common use cases.

---

## Table of Contents

1. [Classical Retinex Enhancement](#1-classical-retinex-enhancement)
2. [Training Models](#2-training-models)
3. [Evaluation & Comparison](#3-evaluation--comparison)
4. [Inference on Custom Images](#4-inference-on-custom-images)
5. [Advanced Usage](#5-advanced-usage)

---

## 1. Classical Retinex Enhancement

### Example 1.1: Single Image Enhancement (No Training)

```powershell
# Enhance a single image using Multi-Scale Retinex
python inference_demo.py retinex `
    --input lol_dataset\eval15\low\1.png `
    --output_dir retinex_results
```

**Output:**
- `SSR_sigma15.png` - Fine-scale enhancement
- `SSR_sigma80.png` - Medium-scale enhancement  
- `SSR_sigma250.png` - Coarse-scale enhancement
- `MSR.png` - Multi-scale combination (best)
- `retinex_comparison.png` - Side-by-side grid

**When to use:** Quick results without training

---

## 2. Training Models

### Example 2.1: Train Hybrid Retinex-Net (Recommended)

```powershell
# Train the hybrid model
python training\train.py --config configs\hybrid_retinex.yaml
```

**Expected output:**
```
Model: HybridRetinexNet
Parameters: 186,883
Device: cuda

Epoch 0:
  Train Loss: 0.3245
  Val PSNR: 20.15 dB
  Val SSIM: 0.7234

Epoch 50:
  Train Loss: 0.1123
  Val PSNR: 24.82 dB
  Val SSIM: 0.8567

...

Training completed!
Best PSNR: 25.34 dB
```

**Checkpoints saved to:** `experiments/hybrid_retinex_net/checkpoints/`

### Example 2.2: Train Zero-DCE Style (Self-Supervised)

```powershell
# No ground truth needed!
python training\train.py --config configs\dce_net.yaml
```

**Use when:** No paired training data available

### Example 2.3: Resume Training from Checkpoint

```powershell
python training\train.py `
    --config configs\hybrid_retinex.yaml `
    --resume experiments\hybrid_retinex_net\checkpoints\checkpoint_latest.pth
```

### Example 2.4: Monitor Training with TensorBoard

```powershell
# In a separate terminal
tensorboard --logdir experiments\hybrid_retinex_net\logs

# Open browser to http://localhost:6006
```

**What you'll see:**
- Training/validation loss curves
- PSNR/SSIM metrics over time
- Learning rate schedule
- Sample enhanced images

---

## 3. Evaluation & Comparison

### Example 3.1: Basic Evaluation

```powershell
python evaluation\evaluate.py `
    --config configs\hybrid_retinex.yaml `
    --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth
```

**Output:**
```
Evaluation Results
====================================
Deep Learning Model (HybridRetinexNet):
  PSNR: 25.34 dB
  SSIM: 0.8721
  NIQE: 4.23

Retinex Baseline (MSR):
  PSNR: 18.15 dB
  SSIM: 0.6489

✓ Results saved to evaluation_results/
```

### Example 3.2: Full Ablation Study

```powershell
python evaluation\evaluate.py `
    --config configs\hybrid_retinex.yaml `
    --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth `
    --ablation
```

**Output table (ablation_table.txt):**
```
Experiment                    PSNR        SSIM
--------------------------------------------------
Raw Input                    10.45      0.4234
SSR (σ=80)                   16.78      0.5812
MSR                          18.15      0.6489
DL Only                      22.89      0.8123
Hybrid (SSR + DL)            24.12      0.8456
Hybrid (MSR + DL)            25.34      0.8721
```

### Example 3.3: Custom Results Directory

```powershell
python evaluation\evaluate.py `
    --config configs\hybrid_retinex.yaml `
    --checkpoint path\to\checkpoint.pth `
    --results_dir my_evaluation_results
```

---

## 4. Inference on Custom Images

### Example 4.1: Single Image Enhancement

```powershell
python inference_demo.py single `
    --config configs\hybrid_retinex.yaml `
    --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth `
    --input my_dark_photo.jpg `
    --output enhanced_photo.jpg
```

**Input:** `my_dark_photo.jpg` (low-light image)
**Output:** `enhanced_photo.jpg` (enhanced image)

### Example 4.2: Batch Processing

```powershell
# Enhance all images in a folder
python inference_demo.py batch `
    --config configs\hybrid_retinex.yaml `
    --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth `
    --input_dir D:\my_photos\dark_images `
    --output_dir D:\my_photos\enhanced_images
```

**What happens:**
- Processes all `.jpg`, `.png`, `.bmp` files
- Saves enhanced images as `enhanced_<filename>`
- Creates comparison images as `comparison_<filename>`

### Example 4.3: Batch Processing (Enhanced Only, No Comparisons)

```powershell
python inference_demo.py batch `
    --config configs\hybrid_retinex.yaml `
    --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth `
    --input_dir input_folder `
    --output_dir output_folder `
    --no-compare
```

---

## 5. Advanced Usage

### Example 5.1: Custom Training with Modified Config

Create `configs/my_config.yaml`:

```yaml
experiment_dir: 'experiments/my_experiment'
seed: 42

model:
  name: 'HybridRetinexNet'
  use_retinex_input: true

data:
  root_dir: 'lol_dataset'
  batch_size: 4              # ← Reduced for limited GPU
  image_size: 128            # ← Smaller images
  num_workers: 2
  paired: true

loss:
  use_reconstruction: true
  use_perceptual: false      # ← Disabled for speed
  lambda_recon: 1.0
  lambda_exposure: 2.0       # ← Increased
  lambda_color: 5.0
  lambda_spatial: 10.0
  target_exposure: 0.5       # ← Darker output

optimizer:
  name: 'Adam'
  lr: 0.0001

scheduler:
  name: 'CosineAnnealingLR'
  T_max: 50                  # ← Fewer epochs

training:
  epochs: 50                 # ← Shorter training
  grad_clip: 0.5
```

Train:
```powershell
python training\train.py --config configs\my_config.yaml
```

### Example 5.2: Compare Different Models

Train all models:

```powershell
# 1. Hybrid Retinex-Net
python training\train.py --config configs\hybrid_retinex.yaml

# 2. DCE-Net
python training\train.py --config configs\dce_net.yaml

# 3. U-Net
python training\train.py --config configs\unet.yaml
```

Evaluate all:

```powershell
# Hybrid
python evaluation\evaluate.py `
    --config configs\hybrid_retinex.yaml `
    --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth `
    --results_dir results_hybrid

# DCE-Net
python evaluation\evaluate.py `
    --config configs\dce_net.yaml `
    --checkpoint experiments\dce_net\checkpoints\checkpoint_best.pth `
    --results_dir results_dce

# U-Net
python evaluation\evaluate.py `
    --config configs\unet.yaml `
    --checkpoint experiments\unet\checkpoints\checkpoint_best.pth `
    --results_dir results_unet
```

Compare results from `results_*/metrics.json`

### Example 5.3: GPU Selection

```powershell
# Use specific GPU (if you have multiple)
$env:CUDA_VISIBLE_DEVICES=0
python training\train.py --config configs\hybrid_retinex.yaml

# Use CPU only
$env:CUDA_VISIBLE_DEVICES=-1
python training\train.py --config configs\hybrid_retinex.yaml
```

### Example 5.4: Mixed Precision Training (Faster)

Modify config:

```yaml
training:
  epochs: 100
  grad_clip: 0.5
  use_amp: true  # ← Add this
```

### Example 5.5: Export Model for Deployment

```python
# export_model.py
import torch
from models import HybridRetinexNet

# Load trained model
model = HybridRetinexNet()
checkpoint = torch.load('experiments/hybrid_retinex_net/checkpoints/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
print("Model exported to model.onnx")
```

Run:
```powershell
python export_model.py
```

---

## 📊 Common Workflows

### Workflow 1: Quick Experiment

```powershell
# 1. Test on one image
python inference_demo.py retinex --input test.jpg --output_dir quick_test

# 2. Train small model
python training\train.py --config configs\dce_net.yaml

# 3. Evaluate
python evaluation\evaluate.py --config configs\dce_net.yaml --checkpoint experiments\dce_net\checkpoints\checkpoint_best.pth
```

### Workflow 2: Full Research Pipeline

```powershell
# 1. Train all models
python training\train.py --config configs\hybrid_retinex.yaml
python training\train.py --config configs\dce_net.yaml
python training\train.py --config configs\unet.yaml

# 2. Evaluate with ablation
python evaluation\evaluate.py --config configs\hybrid_retinex.yaml --checkpoint ... --ablation

# 3. Analyze results
# Compare metrics.json from each experiment
# Create tables and plots

# 4. Test on real-world images
python inference_demo.py batch --config configs\hybrid_retinex.yaml --checkpoint ... --input_dir real_photos --output_dir results
```

### Workflow 3: Production Deployment

```powershell
# 1. Train production model
python training\train.py --config configs\hybrid_retinex.yaml

# 2. Evaluate thoroughly
python evaluation\evaluate.py --config configs\hybrid_retinex.yaml --checkpoint best_model.pth --ablation

# 3. Export for deployment
python export_model.py

# 4. Batch process production data
python inference_demo.py batch --config configs\hybrid_retinex.yaml --checkpoint best_model.pth --input_dir production_data --output_dir processed_data
```

---

## 🔧 Troubleshooting Examples

### Fix: Training Too Slow

```yaml
# Reduce in config
data:
  batch_size: 4     # ← Smaller batches
  image_size: 128   # ← Smaller images
  num_workers: 2    # ← Fewer workers

training:
  epochs: 50        # ← Fewer epochs
```

### Fix: Results Too Bright

```yaml
loss:
  target_exposure: 0.4  # ← Reduce from 0.6
```

### Fix: Results Too Dark

```yaml
loss:
  target_exposure: 0.7  # ← Increase from 0.6
```

### Fix: Results Too Noisy

```yaml
loss:
  lambda_spatial: 20.0  # ← Increase from 10.0 (more smoothing)
```

### Fix: Wrong Colors

```yaml
loss:
  lambda_color: 10.0  # ← Increase from 5.0 (stronger color preservation)
```

---

## 📝 Additional Tips

1. **Always run quick_test.py first** to verify setup
2. **Start with DCE-Net** (fastest to train)
3. **Use TensorBoard** to monitor training
4. **Save multiple checkpoints** during training
5. **Compare against classical Retinex** to show improvement
6. **Test on your own images** to verify generalization

---

**For more examples, see:**
- `README.md` - Complete documentation
- `QUICKSTART.md` - Beginner guide
- Code comments - Inline examples

**Happy enhancing! 🚀**
