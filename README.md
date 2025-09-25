# FIRE 🔥

**Fidelity Image Resolution Enhancement** (FIRE) is a cutting-edge tool designed to enhance image resolution with unparalleled precision and fidelity. Whether you're working with low-resolution images, restoring old photographs, or improving the quality of visual data, FIRE leverages advanced algorithms to deliver stunning results.

With a focus on speed, accuracy, and ease of use, FIRE is perfect for developers, researchers, and creatives looking to elevate their image processing workflows. Dive in and experience the future of image resolution enhancement today!

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/2treesan/FIRE
cd FIRE
```

2. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the main script with your configurations:

```bash
# Train EDSR Medium model for 4x super-resolution
python main.py \
    loader.hr_dir="<path_to_hr>" \
    loader.lr_dir="<path_to_lr>" \
    model=edsr_medium \
    trainer.epochs=100

# Train Real-ESRGAN for real-world images  
python main.py \
    loader.hr_dir="<path_to_hr>" \
    loader.lr_dir="<path_to_lr>" \
    model=real_esrgan_medium \
    trainer.epochs=100

# Train lightweight model for fast inference
python main.py \
    loader.hr_dir="<path_to_hr>" \
    loader.lr_dir="<path_to_lr>" \
    model=edsr_small \
    trainer.epochs=50
```

- Replace `<path_to_hr>` and `<path_to_lr>` with the paths to your high-resolution and low-resolution datasets respectively. Note that each HR image must have a corresponding LR image with the same name.
- Available models: `edsr_large`, `edsr_medium`, `edsr_small`, `real_esrgan_large`, `real_esrgan_medium`, `real_esrgan_small`, `swinir_large`, `swinir_medium`, `swinir_small`, `edsr_lite`

### Model Usage Examples

```python
from src.model import create_model, get_recommended_model
import torch

# Create models programmatically
model = create_model('edsr_medium', scale=4)  
model = create_model('real_esrgan_large', scale=4)
model = create_model('swinir_small', upscale=4)

# Get recommended model for your use case
recommended = get_recommended_model('balanced')  # Returns 'real_esrgan_medium'
model = create_model(recommended, scale=4)

# Inference
x = torch.randn(1, 3, 64, 64)  # Low-resolution input
y = model(x)  # High-resolution output (1, 3, 256, 256) for 4x upscaling
```

## Model Benchmarking

Run the included benchmark script to compare all models:

```bash
python benchmark_models.py
```

This will test all models and provide detailed performance metrics including:
- Parameter count
- Inference time and FPS
- Memory usage (on GPU)
- Recommendations for different use cases

### Benchmark Results (CPU)

| Model | Parameters | Time (ms) | FPS | Best For |
|-------|------------|-----------|-----|----------|
| **edsr_lite** | 890K | 56.2 | 17.8 | Image restoration (1x) |
| **edsr_small** | 927K | 85.4 | 11.7 | Fast super-resolution |
| **swinir_small** | 736K | 144.8 | 6.9 | Compact transformer |
| **real_esrgan_small** | 1.5M | 120.3 | 8.3 | Real-world images |
| **swinir_medium** | 3.3M | 394.9 | 2.5 | Transformer quality |
| **edsr_medium** | 6.1M | 395.7 | 2.5 | Balanced performance |
| **real_esrgan_medium** | 6.6M | 449.6 | 2.2 | Real-world quality |
| **swinir_large** | 10.1M | 945.8 | 1.1 | Best transformer |
| **real_esrgan_large** | 16.7M | 916.8 | 1.1 | Best real-world |
| **edsr_large** | 43.1M | 2328.9 | 0.4 | Maximum quality |

## Available Models

### Image Restoration (1x)
- **EDSRLite**: A lightweight version of the Enhanced Deep Super-Resolution network for image restoration (same size input/output), optimized for faster performance while maintaining high-quality results.

### Super-Resolution Models (2x, 4x, 8x upscaling)

#### EDSR (Enhanced Deep Super-Resolution)
State-of-the-art CNN-based super-resolution with residual blocks and sub-pixel convolution:
- **EDSR Large**: Best quality with 43M parameters - ideal for offline processing
- **EDSR Medium**: Balanced performance with 6M parameters - good for most use cases  
- **EDSR Small**: Fast inference with 930K parameters - suitable for real-time applications

#### Real-ESRGAN 
Advanced super-resolution model designed for real-world images with complex degradations:
- **Real-ESRGAN Large**: Premium quality with 16.7M parameters - handles severe degradations
- **Real-ESRGAN Medium**: Balanced approach with 6.6M parameters - good real-world performance
- **Real-ESRGAN Small**: Lightweight with 1.5M parameters - faster real-world processing

#### SwinIR (Swin Transformer)
Transformer-based super-resolution leveraging window-based self-attention:
- **SwinIR Large**: Top transformer performance with 10.2M parameters
- **SwinIR Medium**: Efficient transformer with 3.3M parameters  
- **SwinIR Small**: Compact transformer with 740K parameters

### Model Selection Guide

| Use Case | Recommended Model | Parameters | Best For |
|----------|------------------|------------|----------|
| **Best Quality** | `edsr_large` | 43.1M | Offline processing, maximum quality |
| **Balanced** | `real_esrgan_medium` | 6.6M | General purpose, real-world images |
| **Fastest** | `edsr_small` | 930K | Real-time applications |
| **Real-world Images** | `real_esrgan_large` | 16.7M | Photos with noise, blur, compression |
| **Transformer** | `swinir_medium` | 3.3M | Latest transformer architecture |
