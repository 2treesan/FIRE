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
python main.py \
    loader.hr_dir="<path_to_hr>" \
    loader.lr_dir="<path_to_lr>" \
    model="<model_name>" \
    trainer.epochs=100 \
    # Add other configurations as needed
```

- Replace `<path_to_hr>` and `<path_to_lr>` with the paths to your high-resolution and low-resolution datasets respectively. Note that each HR image must have a corresponding LR image with the same name.
- Replace `<model_name>` with the model defined in the `src/model` directory.

## Available Models

### Restoration Models (Same Resolution I/O)
- **EDSRLite** (`edsr_lite`): A lightweight version of the Enhanced Deep Super-Resolution network, optimized for image restoration tasks.

### Classic Super-Resolution Models  
- **SRCNN** (`srcnn`): The pioneering deep learning model for super-resolution. Simple 3-layer CNN architecture.
- **SRCNN++** (`srcnn_plus`): Enhanced SRCNN with deeper architecture and optional residual connections.

### EDSR Family (True Super-Resolution with Upscaling)
- **EDSR** (`edsr_2x`, `edsr_4x`): Enhanced Deep Super-Resolution network with 2x or 4x upscaling capability.
- **EDSR Large** (`edsr_large_4x`): Large version of EDSR with 256 features and 32 residual blocks for maximum quality.
- **EDSR Baseline** (`edsr_baseline`): EDSR variant with batch normalization for comparison studies.

### Real-World Super-Resolution Models
- **Real-ESRGAN** (`real_esrgan_4x`): State-of-the-art model for practical super-resolution, designed to handle real-world degraded images.
- **Real-ESRGAN Light** (`real_esrgan_light_4x`): Lightweight version of Real-ESRGAN optimized for faster inference.

### Model Specifications

| Model | Parameters | Upscaling | Best Use Case |
|-------|------------|-----------|---------------|
| EDSRLite | ~20K | 1x (restoration) | Fast image denoising/restoration |
| SRCNN | ~20K | 1x (bicubic pre-upsampled) | Learning/research baseline |
| SRCNN++ | ~40K | 1x (bicubic pre-upsampled) | Enhanced baseline with residuals |
| EDSR 2x/4x | ~1.4M/1.5M | 2x/4x | High-quality upscaling |
| EDSR Large | ~43M | 4x | Maximum quality upscaling |
| Real-ESRGAN | ~17M | 4x | Real-world degraded images |
| Real-ESRGAN Light | ~2.3M | 4x | Fast real-world upscaling |
