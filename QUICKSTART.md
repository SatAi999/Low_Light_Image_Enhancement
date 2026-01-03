# 🚀 Quick Start Guide

## For Absolute Beginners

### Step 1: Verify Dataset ✅

Your dataset is already in place:
```
lol_dataset/
├── our485/    # 485 training image pairs
│   ├── low/   # Low-light images
│   └── high/  # Normal-light images
└── eval15/    # 15 test image pairs
    ├── low/
    └── high/
```

### Step 2: Install Dependencies 📦

Open PowerShell in this directory and run:

```powershell
pip install -r requirements.txt
```

This installs: PyTorch, OpenCV, scikit-image, matplotlib, etc.

### Step 3: Run Quick Test 🧪

```powershell
python quick_test.py
```

This verifies everything is working correctly.

---

## Three Ways to Use This Project

### 🎨 Option 1: Classical Retinex (No Training)

**Enhance an image using only classical algorithms:**

```powershell
python inference_demo.py retinex --input lol_dataset\eval15\low\1.png --output_dir retinex_results
```

**What you get:**
- SSR with different scales
- MSR (multi-scale)
- Comparison grid
- **No training needed!**

**Use this when:**
- You want quick results
- No GPU available
- Understanding classical methods

---

### 🧠 Option 2: Train Deep Learning Model

**Train the hybrid Retinex + CNN model:**

```powershell
python training\train.py --config configs\hybrid_retinex.yaml
```

**What happens:**
- Trains for 100 epochs (~2-3 hours on GPU)
- Saves checkpoints to `experiments/hybrid_retinex_net/checkpoints/`
- Logs training progress to TensorBoard
- Automatically validates on eval15 set

**Monitor training:**
```powershell
tensorboard --logdir experiments\hybrid_retinex_net\logs
```
Then open http://localhost:6006 in your browser.

**Use this when:**
- You want best quality results
- GPU is available
- Research/production deployment

---

### 📊 Option 3: Evaluate & Compare

**After training, evaluate the model:**

```powershell
python evaluation\evaluate.py --config configs\hybrid_retinex.yaml --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth --ablation
```

**What you get:**
- Quantitative metrics (PSNR, SSIM, NIQE)
- Visual comparisons
- Ablation study comparing:
  - Raw input
  - SSR only
  - MSR only
  - Deep learning only
  - Hybrid (best)

**Results saved to:** `evaluation_results/`

---

## 🎯 Inference on Your Own Images

### Single Image

```powershell
python inference_demo.py single --config configs\hybrid_retinex.yaml --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth --input your_image.jpg --output enhanced.jpg
```

### Batch Processing

```powershell
python inference_demo.py batch --config configs\hybrid_retinex.yaml --checkpoint experiments\hybrid_retinex_net\checkpoints\checkpoint_best.pth --input_dir input_folder --output_dir output_folder
```

---

## 📋 Available Models

| Model | Config File | When to Use |
|-------|-------------|-------------|
| **HybridRetinexNet** | `configs/hybrid_retinex.yaml` | ⭐ Best quality & generalization |
| DCENet | `configs/dce_net.yaml` | Self-supervised, no GT needed |
| LightweightUNet | `configs/unet.yaml` | Strong baseline |

---

## 🎓 Understanding the Approach

### The Problem

Low-light images suffer from:
- Low brightness
- Low contrast
- Color distortion
- Noise amplification

### Classical Solution: Retinex

Decomposes image into:
- **Illumination (L)**: Lighting conditions
- **Reflectance (R)**: True object appearance

Formula: `I = R × L`

Goal: Recover R (which looks good under any lighting)

### Deep Learning Solution

Train a CNN to enhance images directly.

**Problem:** Overfits to training lighting conditions.

### Hybrid Solution (Our Approach) ⭐

1. Use Retinex to get R (physics-based)
2. Use CNN to refine R (data-driven)

**Advantages:**
- Better generalization
- More stable
- Interpretable
- Less data needed

---

## 📊 Expected Results

On LOL eval15 test set:

| Method | PSNR | SSIM |
|--------|------|------|
| Raw Input | ~10 dB | ~0.4 |
| MSR Only | ~18 dB | ~0.65 |
| DL Only | ~23 dB | ~0.81 |
| **Hybrid** | **~25 dB** | **~0.87** |

---

## 🔧 Troubleshooting

### "Out of Memory"

**Solution:** Reduce batch size in config:
```yaml
data:
  batch_size: 4  # or even 2
  image_size: 128  # or smaller
```

### "Dataset not found"

**Solution:** Check path in config:
```yaml
data:
  root_dir: 'lol_dataset'  # Make sure this points to your dataset
```

### "CUDA not available"

**Solution:** Models work on CPU too (just slower):
- Training: 24 hours vs 2-3 hours
- Inference: 1 sec vs 0.05 sec

---

## 📚 Next Steps

1. ✅ Run `quick_test.py` to verify setup
2. ✅ Try classical Retinex on a sample image
3. ✅ Train a model (start with DCENet - fastest)
4. ✅ Evaluate and compare different methods
5. ✅ Run inference on your own images

**For detailed explanations, see:**
- `README.md` - Complete documentation
- `ROADMAP.md` - Future plans
- `CONTRIBUTING.md` - How to contribute

---

## 🆘 Need Help?

- Check `README.md` for detailed documentation
- Open an issue on GitHub
- Review the code comments (they're extensive!)

**Happy enhancing! 🌟**
