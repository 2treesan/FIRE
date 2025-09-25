# FIRE Model Guide 🔥

This guide provides detailed information about all available super-resolution models in FIRE.

## Quick Start

```bash
# Train with EDSR 4x model
python main.py model=edsr_4x

# Train with Real-ESRGAN Light for fast inference
python main.py model=real_esrgan_light_4x

# Train with SRCNN for learning/research
python main.py model=srcnn
```

## Model Categories

### 🔧 Restoration Models (1x)
These models work on same-resolution input/output, focusing on denoising and restoration.

| Model | Config | Parameters | Use Case |
|-------|--------|------------|----------|
| EDSRLite | `edsr_lite` | ~890K | Fast restoration, denoising |

### 📚 Classic Super-Resolution
Educational and baseline models based on early SR research.

| Model | Config | Parameters | Input Processing | Use Case |
|-------|--------|------------|------------------|----------|
| SRCNN | `srcnn` | ~20K | Bicubic pre-upsampled | Learning, baseline |
| SRCNN++ | `srcnn_plus` | ~40K | Bicubic pre-upsampled | Enhanced baseline |

### 🚀 EDSR Family (True Super-Resolution)
Modern residual networks with learnable upsampling.

| Model | Config | Parameters | Upscaling | Quality | Speed |
|-------|--------|------------|-----------|---------|--------|
| EDSR 2x | `edsr_2x` | ~1.4M | 2x | High | Fast |
| EDSR 4x | `edsr_4x` | ~1.5M | 4x | High | Fast |
| EDSR Large | `edsr_large_4x` | ~43M | 4x | Highest | Slow |
| EDSR Baseline | `edsr_baseline` | ~1.4M | 2x/4x | High | Fast |

### 🌟 Real-World Super-Resolution
State-of-the-art models designed for practical applications.

| Model | Config | Parameters | Upscaling | Best For |
|-------|--------|------------|-----------|----------|
| Real-ESRGAN | `real_esrgan_4x` | ~17M | 4x | Degraded real images |
| Real-ESRGAN Light | `real_esrgan_light_4x` | ~2.3M | 4x | Fast real-world SR |

## Model Selection Guide

### Choose by Use Case

- **📖 Learning/Research**: Start with `srcnn` or `srcnn_plus`
- **🖼️ Clean Image Upscaling**: Use `edsr_2x` or `edsr_4x`
- **📱 Mobile/Fast Inference**: Use `real_esrgan_light_4x`
- **🏆 Maximum Quality**: Use `edsr_large_4x` or `real_esrgan_4x`
- **🔧 Image Restoration**: Use `edsr_lite`

### Choose by Computational Budget

- **Low (< 100K params)**: `srcnn`, `srcnn_plus`
- **Medium (1-3M params)**: `edsr_2x`, `edsr_4x`, `real_esrgan_light_4x`
- **High (> 10M params)**: `real_esrgan_4x`, `edsr_large_4x`

### Choose by Upscaling Factor

- **No upscaling (restoration)**: `edsr_lite`
- **2x upscaling**: `edsr_2x`, `edsr_baseline`
- **4x upscaling**: `edsr_4x`, `edsr_large_4x`, `real_esrgan_4x`, `real_esrgan_light_4x`

## Technical Details

### Architecture Highlights

**SRCNN Family**
- Simple 3-layer CNN architecture
- Requires bicubic pre-upsampling
- Good for understanding SR fundamentals

**EDSR Family**
- Residual blocks without batch normalization
- Sub-pixel convolution for upsampling
- Global and local residual connections

**Real-ESRGAN Family**
- Residual-in-Residual Dense Blocks (RRDB)
- Dense connections within blocks
- Designed for real-world degradations

### Performance Characteristics

**Memory Usage (approximate for 256x256 input)**
- SRCNN: ~50MB VRAM
- EDSR: ~500MB VRAM  
- Real-ESRGAN: ~800MB VRAM
- EDSR Large: ~2GB VRAM

**Training Time (relative, single GPU)**
- SRCNN: 1x (fastest)
- EDSR: 5x
- Real-ESRGAN Light: 8x
- Real-ESRGAN: 15x
- EDSR Large: 30x (slowest)

## Custom Configuration

You can create custom model configurations by copying and modifying existing configs:

```yaml
# cfg/model/my_custom_edsr.yaml
_target_: src.model.edsr.EDSR
scale: 3  # 3x upscaling
in_ch: 3
out_ch: 3
nf: 128   # more features
nb: 24    # more blocks
res_scale: 0.1
```

Then use with:
```bash
python main.py model=my_custom_edsr
```

## Model Implementation

All models are implemented in `src/model/` with clean, documented code:

- `srcnn.py`: SRCNN and SRCNN++ implementations
- `edsr.py`: EDSR family models
- `real_esrgan.py`: Real-ESRGAN implementations
- `edsr_lite.py`: Original restoration model

## Testing

Run the test suite to verify all models:

```bash
python test_models.py
```

Or see examples of usage:

```bash  
python example_usage.py
```

## References

1. **SRCNN**: Dong et al., "Image Super-Resolution Using Deep Convolutional Networks"
2. **EDSR**: Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution"
3. **Real-ESRGAN**: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"

---

For more information, see the main [README.md](README.md) or check the model implementations in `src/model/`.