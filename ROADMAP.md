# Project Roadmap & TODO

## Completed ✅
- [x] Project structure setup
- [x] Dataset loader with augmentations
- [x] Classical Retinex algorithms (SSR, MSR)
- [x] Deep learning models (U-Net, DCE-Net, Enhancement CNN, Hybrid)
- [x] Loss functions (reconstruction, perceptual, exposure, color, spatial)
- [x] Evaluation metrics (PSNR, SSIM, NIQE)
- [x] Training script with TensorBoard logging
- [x] Evaluation script with ablation studies
- [x] Inference demo for custom images
- [x] Comprehensive documentation
- [x] Configuration files for different experiments

## In Progress 🚧
- [ ] Train baseline models on LOL dataset
- [ ] Generate benchmark results
- [ ] Create sample output visualizations

## Future Enhancements 🔮

### Short Term
- [ ] Add pre-trained model weights download
- [ ] Implement additional metrics (LPIPS, FID)
- [ ] Add more data augmentation techniques
- [ ] Support for batch inference optimization
- [ ] Add model quantization for faster inference

### Medium Term
- [ ] Video enhancement support
- [ ] Real-time inference optimization (TensorRT, ONNX)
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Web demo interface (Gradio/Streamlit)
- [ ] Support for RAW image formats
- [ ] Multi-exposure fusion

### Long Term
- [ ] Attention mechanisms for detail recovery
- [ ] Transformer-based models
- [ ] Few-shot learning for domain adaptation
- [ ] Integration with HDR imaging pipeline
- [ ] Benchmark on additional datasets (LOL-v2, SID, SICE)
- [ ] Research paper preparation

## Known Issues 🐛
- NIQE implementation is simplified (use pyiqa for production)
- Perceptual loss requires VGG16 download on first run
- Multi-GPU training not fully tested

## Performance Targets 🎯
- [ ] PSNR > 25 dB on LOL eval15
- [ ] SSIM > 0.85 on LOL eval15
- [ ] Inference < 50ms on GPU (256x256)
- [ ] Model size < 5MB

## Documentation TODOs 📝
- [ ] Add architecture diagrams
- [ ] Create tutorial notebooks
- [ ] Add more usage examples
- [ ] Create FAQ section
- [ ] Add troubleshooting guide
- [ ] Video tutorials for training/inference

## Community 👥
- [ ] Set up GitHub Actions for CI/CD
- [ ] Add code of conduct
- [ ] Create contribution guidelines
- [ ] Set up issue templates
- [ ] Add unit tests
- [ ] Code coverage reports

---

**Last Updated:** January 2026
