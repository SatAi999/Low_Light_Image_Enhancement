# Quick Start Guide

## Launch the Web Demo

1. **Install Dependencies** (if not already done):
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Access the Interface**:
   - Open your browser to `http://localhost:8501`
   - The app will automatically open

## Features

### 📤 Upload & Enhance Tab
- Upload your low-light images (PNG, JPG, JPEG)
- Click "Enhance Image" to process
- View before/after comparison
- Download enhanced results
- See detailed metrics (PSNR, brightness, contrast)
- Visualize RGB histograms

### 🖼️ Gallery Tab
- Browse example enhancements
- See real results on LOL dataset
- Compare input vs output quality

### 📈 Training Metrics Tab
- View PSNR/SSIM training curves
- See model convergence
- Review training configuration
- Understand performance evolution

### ℹ️ About Tab
- Learn about the hybrid approach
- Technical details
- Performance metrics
- Future roadmap

## Settings (Sidebar)

- **Use Retinex Preprocessing**: Enable/disable MSR preprocessing
  - `False`: Pure CNN enhancement (current default)
  - `True`: Hybrid Retinex + CNN (experimental)
  
- **Show Histograms**: Toggle RGB histogram visualization

- **Show Detailed Metrics**: Toggle metric cards

## Tips

- For best results, use images with moderate low-light (not completely black)
- Images are automatically resized to 256x256 for processing
- Processing time: ~0.5-2 seconds per image on CPU
- GPU would be 10-50x faster if available

## Troubleshooting

**App won't start:**
- Ensure you're in the project directory
- Check that `experiments/hybrid_retinex_net/checkpoints/checkpoint_best.pth` exists
- Verify all dependencies are installed

**Enhancement looks poor:**
- Model trained for only 25 epochs (limited performance)
- Try toggling Retinex preprocessing
- Some images may be too dark/noisy for good results

**Slow performance:**
- Training was on CPU (no GPU detected)
- Consider using GPU for inference if available
- Reduce image size for faster processing

## Keyboard Shortcuts

- `R`: Rerun the app
- `C`: Clear cache
- `Ctrl/Cmd + K`: Open command palette
