"""
Streamlit Web App for Low-Light Image Enhancement
Professional showcase with interactive features
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import yaml
from pathlib import Path
import io
import time

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import project modules
from models import HybridRetinexNet
from retinex import apply_retinex_to_tensor
from utils import load_checkpoint
import torchvision.transforms as transforms

# Page config
st.set_page_config(
    page_title="Low-Light Image Enhancement",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
    }
    .highlight {
        background-color: #ffffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        config_path = "experiments/hybrid_retinex_net/config.yaml"
        checkpoint_path = "experiments/hybrid_retinex_net/checkpoints/checkpoint_best.pth"
        
        # Check if files exist
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # HybridRetinexNet only accepts in_channels, out_channels, and use_retinex_input
        model = HybridRetinexNet(
            in_channels=3,
            out_channels=3,
            use_retinex_input=config['model'].get('use_retinex_input', False)
        )
        
        load_checkpoint(checkpoint_path, model)
        model.eval()
        
        return model, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure the model has been trained and checkpoint exists.")
        raise e


def enhance_image(image, model, config, use_retinex=False):
    """Enhance a single image"""
    # Store original size
    original_size = image.size
    
    # Resize to model input size (256x256)
    image_resized = image.resize((256, 256), Image.LANCZOS)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image_resized).unsqueeze(0)
    
    # Apply enhancement
    with torch.no_grad():
        if use_retinex:
            _, reflectance = apply_retinex_to_tensor(
                img_tensor,
                method='MSR',
                scales=[15, 80, 250],
                gamma=1.2
            )
            enhanced_tensor = model(img_tensor, reflectance)
        else:
            enhanced_tensor = model(img_tensor)
    
    # Convert back to numpy
    enhanced_np = enhanced_tensor[0].cpu().permute(1, 2, 0).numpy()
    enhanced_np = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)
    
    # Resize back to original size
    enhanced_pil = Image.fromarray(enhanced_np)
    enhanced_pil = enhanced_pil.resize(original_size, Image.LANCZOS)
    enhanced_np = np.array(enhanced_pil)
    
    return enhanced_np


def calculate_metrics(original, enhanced):
    """Calculate image quality metrics"""
    # PSNR
    mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Average brightness
    orig_brightness = np.mean(original)
    enh_brightness = np.mean(enhanced)
    
    # Contrast (std deviation)
    orig_contrast = np.std(original)
    enh_contrast = np.std(enhanced)
    
    return {
        'psnr': psnr,
        'orig_brightness': orig_brightness,
        'enh_brightness': enh_brightness,
        'brightness_gain': (enh_brightness - orig_brightness) / orig_brightness * 100,
        'orig_contrast': orig_contrast,
        'enh_contrast': enh_contrast,
        'contrast_gain': (enh_contrast - orig_contrast) / orig_contrast * 100
    }


def create_histogram_comparison(original, enhanced):
    """Create histogram comparison plot"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['red', 'green', 'blue']
    channels = ['Red', 'Green', 'Blue']
    
    for i, (color, channel) in enumerate(zip(colors, channels)):
        axes[i].hist(original[:,:,i].ravel(), bins=256, color=color, alpha=0.5, label='Original')
        axes[i].hist(enhanced[:,:,i].ravel(), bins=256, color=color, alpha=0.7, label='Enhanced')
        axes[i].set_title(f'{channel} Channel', fontweight='bold')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.title("💡 Low-Light Image Enhancement")
    st.markdown("### Hybrid Retinex + Deep Learning Approach")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=HybridRetinexNet", 
                 width='stretch')
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **Physics-Guided**: Based on Retinex theory
        - **Deep Learning**: CNN refinement
        - **Lightweight**: Only 169K parameters
        - **Real-time**: Fast inference
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.metric("Best PSNR", "18.29 dB", "Epoch 16")
        st.metric("Best SSIM", "0.7239", "Epoch 10")
        st.metric("Parameters", "169,299", "Lightweight")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        use_retinex = st.checkbox("Use Retinex Preprocessing", value=False,
                                  help="Enable Multi-Scale Retinex preprocessing")
        show_histograms = st.checkbox("Show Histograms", value=True)
        show_metrics = st.checkbox("Show Detailed Metrics", value=True)
    
    # Load model
    with st.spinner("Loading model..."):
        try:
            model, config = load_model()
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Enhance", "🖼️ Gallery", "📈 Training Metrics", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Upload Your Low-Light Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload a low-light image to enhance"
            )
            
            if uploaded_file is not None:
                # Load and display original
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Original Image", width='stretch')
                
                # Enhancement button
                if st.button("🚀 Enhance Image", type="primary"):
                    try:
                        with st.spinner("Enhancing..."):
                            start_time = time.time()
                            enhanced = enhance_image(image, model, config, use_retinex)
                            processing_time = time.time() - start_time
                            
                            # Store in session state
                            st.session_state['original'] = np.array(image)
                            st.session_state['enhanced'] = enhanced
                            st.session_state['processing_time'] = processing_time
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error during enhancement: {str(e)}")
                        st.error("Please try with a different image or check the model configuration.")
        
        with col2:
            if 'enhanced' in st.session_state:
                enhanced = st.session_state['enhanced']
                st.image(enhanced, caption="Enhanced Image", width='stretch')
                
                # Download button
                enhanced_pil = Image.fromarray(enhanced)
                buf = io.BytesIO()
                enhanced_pil.save(buf, format='PNG')
                st.download_button(
                    label="📥 Download Enhanced Image",
                    data=buf.getvalue(),
                    file_name="enhanced_image.png",
                    mime="image/png"
                )
                
                st.success(f"✨ Enhancement completed in {st.session_state['processing_time']:.3f}s")
        
        # Metrics section
        if 'enhanced' in st.session_state and show_metrics:
            st.markdown("---")
            st.markdown("### 📊 Image Quality Metrics")
            
            metrics = calculate_metrics(st.session_state['original'], st.session_state['enhanced'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
            with col2:
                st.metric("Brightness Gain", f"{metrics['brightness_gain']:.1f}%", 
                         delta=f"{metrics['enh_brightness']:.1f}")
            with col3:
                st.metric("Contrast Gain", f"{metrics['contrast_gain']:.1f}%",
                         delta=f"{metrics['enh_contrast']:.1f}")
            with col4:
                st.metric("Processing Time", f"{st.session_state['processing_time']*1000:.1f} ms")
        
        # Histogram comparison
        if 'enhanced' in st.session_state and show_histograms:
            st.markdown("---")
            st.markdown("### 📊 Histogram Comparison")
            fig = create_histogram_comparison(st.session_state['original'], st.session_state['enhanced'])
            st.pyplot(fig)
    
    with tab2:
        st.markdown("### 🖼️ Example Results Gallery")
        st.markdown("Sample enhancements from the LOL dataset")
        
        # Display sample results
        sample_dir = Path("results/showcase")
        if sample_dir.exists():
            comparison_files = sorted(sample_dir.glob("comparison_*.png"))
            
            if comparison_files:
                for idx, img_path in enumerate(comparison_files[:3], 1):
                    st.image(str(img_path), caption=f"Example {idx}", width='stretch')
                    st.markdown("---")
            else:
                st.info("No gallery images found. Run training and inference first.")
        else:
            st.info("Gallery not available. Generate showcase images first.")
    
    with tab3:
        st.markdown("### 📈 Training Performance")
        
        # Display training metrics
        metrics_path = Path("results/showcase/training_metrics.png")
        if metrics_path.exists():
            st.image(str(metrics_path), width='stretch')
        
        # Display summary
        summary_path = Path("results/showcase/results_summary.png")
        if summary_path.exists():
            st.image(str(summary_path), width='stretch')
        
        # Training details
        st.markdown("### 🎓 Training Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Model Architecture:**
            - Base Channels: 48
            - Residual Blocks: 4
            - Total Parameters: 169,299
            - Architecture: Encoder-Decoder
            """)
        
        with col2:
            st.markdown("""
            **Training Setup:**
            - Learning Rate: 0.001
            - Batch Size: 16
            - Epochs: 25
            - Optimizer: Adam
            - Scheduler: CosineAnnealing
            """)
    
    with tab4:
        st.markdown("### ℹ️ About This Project")
        
        st.markdown("""
        <div class="highlight">
        <h4>🎯 Project Overview</h4>
        This project implements a hybrid approach to low-light image enhancement that combines 
        classical Retinex theory with modern deep learning techniques.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔬 Technical Approach")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Classical Foundation:**
            - Multi-Scale Retinex (MSR) decomposition
            - Physics-based illumination estimation
            - Color constancy principles
            - Robust to lighting variations
            """)
        
        with col2:
            st.markdown("""
            **Deep Learning Refinement:**
            - Lightweight CNN architecture
            - Residual learning strategy
            - Multi-scale feature extraction
            - Efficient 169K parameters
            """)
        
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        - **Hybrid Intelligence**: Combines physics understanding with data-driven learning
        - **Generalization**: Works across diverse lighting conditions
        - **Efficiency**: Lightweight model suitable for deployment
        - **Interpretability**: Physics-guided approach provides transparency
        """)
        
        st.markdown("### 📊 Performance Metrics")
        st.markdown("""
        - **PSNR**: 18.29 dB (Peak Signal-to-Noise Ratio)
        - **SSIM**: 0.7239 (Structural Similarity Index)
        - **Dataset**: LOL (485 training pairs)
        - **Training**: 25 epochs on CPU
        """)
        
        st.markdown("### 🚀 Future Improvements")
        st.markdown("""
        - Extended training (50-100 epochs) for better convergence
        - GPU acceleration for faster training
        - Perceptual loss integration for visual quality
        - Real-time video enhancement capability
        - Mobile deployment optimization
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        Created as a showcase project demonstrating:
        - Deep learning expertise
        - Computer vision knowledge
        - Physics-guided AI design
        - Production-ready implementation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
        <p>Low-Light Image Enhancement | HybridRetinexNet | 2026</p>
        <p>Combining Physics and Deep Learning for Robust Enhancement</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
