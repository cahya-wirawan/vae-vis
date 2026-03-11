# VAE Latent Space Explorer (VAE-Explorer)

A project to visualize and explore the latent space of Variational Autoencoders (VAEs) trained on the MNIST and Animal Faces-HQ (AFHQ) datasets. This repository demonstrates how to bridge deep learning models from Python/PyTorch to high-performance web applications using Rust, WebAssembly, and WebGPU.

**Live Demo**: [https://cahya-wirawan.github.io/vae-explorer/](https://cahya-wirawan.github.io/vae-explorer/)

## Features

- **Latent Space Exploration**: Navigate the VAE's latent space using directional sliders to see how generated images morph and transition.
- **Spherical Interpolation (Slerp)**: Smoothly interpolate between two random latent samples with preview thumbnails of both endpoints.
- **Triple Backend**:
  - **Python/Streamlit**: A rapid prototype using PyTorch and Streamlit.
  - **Rust/WebAssembly**: A client-side implementation where the neural network decoder is hand-rolled in Rust and compiled to WASM.
  - **WebGPU Compute Shaders**: GPU-accelerated decoder inference using WGSL compute shaders, with automatic fallback to WASM on unsupported browsers.
- **Model Quantization**: Export weights in fp32, fp16 (50% size), or int8 (25% size) with automatic format detection on load.
- **Two VAE Architectures**:
  - **MNIST**: Simple fully-connected decoder, 2D latent space, 28×28 grayscale output.
  - **AFHQ Cats**: ResNet-style decoder with 5 upsampling stages, 256-dim latent space, 128×128 RGB output.

### Training Features

- **PyTorch Lightning**: Scalable training with automatic multi-GPU support (DDP, FSDP)
- **Multi-Loss Function**: Combines L1, VGG perceptual loss, SSIM, and KL divergence with configurable weights
- **Flexible Datasets**: Train on any HuggingFace image dataset via `--dataset` argument
- **Grayscale Mode**: Optimized augmentations for grayscale datasets like MNIST (`--grayscale`)
- **W&B Integration**: Log metrics and sample images to Weights & Biases (`--use_wandb`)
- **Mixed Precision**: Support for fp16 and bf16 training (`--precision`)

## Project Structure

```
vae-explorer/
├── src/
│   ├── training/                # Model training scripts
│   │   ├── mnist.py             # Legacy MNIST training script
│   │   └── dog.py               # Unified Lightning trainer (any HF dataset)
│   ├── python/                  # Python-based exploration and utilities
│   │   ├── app.py               # Streamlit visualization app (MNIST)
│   │   ├── dog_app.py           # Streamlit visualization app (AFHQ)
│   │   ├── export_weights.py    # Export MNIST weights to Rust source
│   │   ├── export_dog_weights.py # Export AFHQ decoder weights (binary, BN-folded)
│   │   └── quantize_dog.py      # Quantize & export (fp32/fp16/int8)
│   └── rust/
│       ├── vae_wasm/            # MNIST WASM project (weights baked in)
│       │   ├── src/
│       │   │   ├── lib.rs       # Hand-rolled FC decoder
│       │   │   └── weights.rs   # Generated weight constants
│       │   └── index.html       # MNIST explorer UI
│       └── dog_wgpu/            # AFHQ WebGPU + WASM project
│           ├── src/
│           │   └── lib.rs       # Hand-rolled ResNet decoder (WASM fallback)
│           ├── gpu-decoder.js   # WebGPU compute shader decoder (8 WGSL shaders)
│           ├── index.html       # AFHQ explorer UI (Explore + Interpolate modes)
│           └── pkg/             # wasm-pack build output
├── docs/                        # GitHub Pages deployment
└── README.md
```

## Getting Started

### 1. Training the Model

The unified training script `dog.py` supports any HuggingFace image dataset with PyTorch Lightning.

#### Dependencies
```bash
pip install torch torchvision lightning datasets wandb
```

#### AFHQ Dataset (default)
```bash
python src/training/dog.py --latent_dim 256 --epochs 500
```

#### MNIST Dataset
```bash
python src/training/dog.py \
    --dataset mnist \
    --img_size 28 \
    --latent_dim 64 \
    --grayscale \
    --epochs 100
```

#### Multi-GPU Training
```bash
# Use all available GPUs with DDP
python src/training/dog.py --gpus -1 --strategy ddp

# Use specific number of GPUs
python src/training/dog.py --gpus 4 --strategy ddp

# Mixed precision for faster training
python src/training/dog.py --gpus -1 --precision 16-mixed
```

#### Custom HuggingFace Dataset
```bash
python src/training/dog.py \
    --dataset "username/my-dataset" \
    --image_column "img" \
    --img_size 128
```

#### Training with W&B Logging
```bash
python src/training/dog.py \
    --use_wandb \
    --wandb_project my-vae-project \
    --wandb_run_name experiment-1
```

#### Full Training Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `huggan/AFHQ` | HuggingFace dataset name |
| `--image_column` | `image` | Column containing images |
| `--img_size` | `128` | Image resolution |
| `--latent_dim` | `256` | Latent space dimension |
| `--batch_size` | `196` | Batch size per GPU |
| `--epochs` | `500` | Number of training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--grayscale` | `false` | Disable flip/color augmentations |
| `--gpus` | `-1` | Number of GPUs (-1 = all) |
| `--strategy` | `auto` | DDP, FSDP, or auto |
| `--precision` | `32` | 32, 16-mixed, or bf16-mixed |
| `--kl_weight_max` | `0.001` | Maximum KL divergence weight |
| `--kl_warmup_epochs` | `30` | Epochs to ramp up KL weight |
| `--use_wandb` | `false` | Enable W&B logging |
| `--resume` | `None` | Path to checkpoint to resume |

### 2. Exporting & Quantizing Weights

#### MNIST → Rust Source
```bash
cd src/rust/vae_wasm
python ../../python/export_weights.py
```
This updates `src/weights.rs` with baked-in weight constants.

#### AFHQ → Binary (with optional quantization)

Export full-precision weights with BatchNorm folding:
```bash
python src/python/export_dog_weights.py src/training/dog_vae_256_best.pth \
    -o src/rust/dog_wgpu/weights.bin --latent-dim 256
```

Or export quantized weights for smaller file sizes:
```bash
# fp16 — ~23 MB (50% of fp32), negligible quality loss
python src/python/quantize_dog.py src/training/dog_vae_256_best.pth \
    --mode fp16 -o src/rust/dog_wgpu/weights.bin --latent-dim 256

# int8 — ~12 MB (25% of fp32), minor quality loss
python src/python/quantize_dog.py src/training/dog_vae_256_best.pth \
    --mode int8 -o src/rust/dog_wgpu/weights.bin --latent-dim 256
```

Quantized files include a 16-byte VAEQ header so the web app auto-detects the format.

### 3. Running the Visualization

#### Option A: Python (Streamlit)

```bash
cp decoder_weights.pth src/python/
streamlit run src/python/app.py
```

#### Option B: MNIST — Rust + WebAssembly

```bash
cd src/rust/vae_wasm
wasm-pack build --target web
python3 -m http.server 8000
```
Open `http://localhost:8000`.

#### Option C: AFHQ — WebGPU + WASM

```bash
cd src/rust/dog_wgpu

# Build the WASM fallback
wasm-pack build --target web --release

# Serve the directory
python3 -m http.server 8000
```
Open `http://localhost:8000`. The app will:
1. Try **WebGPU** first (GPU-accelerated, ~10-50ms per image).
2. Fall back to **WASM CPU** if WebGPU is unavailable.

The active backend is shown in the subtitle and status bar.

## How it Works

### MNIST VAE
1. **Encoder**: Compresses 28×28 grayscale images into a 2D latent space ($z_1, z_2$).
2. **Decoder**: Takes a 2D coordinate and reconstructs a 28×28 image.
3. **WASM**: Weights are baked into the Rust binary as compile-time constants.

### AFHQ ResNet VAE
1. **Encoder**: 5-stage strided convolutions with ResBlocks, compresses 128×128 RGB → 256-dim latent.
2. **Decoder**: `Linear(256→8192) → reshape(512,4,4) → 5× [Upsample(2×) + Conv2d + ReLU + ResBlock] → Conv2d(3) + Sigmoid`.
3. **BatchNorm Folding**: At export time, BatchNorm parameters are folded into Conv2d weights/biases, eliminating running mean/variance at inference.
4. **WebGPU Compute Shaders**: 8 WGSL compute shaders handle all decoder operations on the GPU:
   - `linear_relu` — Fully-connected layer with ReLU
   - `conv2d_3x3` — 3×3 convolution
   - `bilinear_upsample` — 2× bilinear upsampling
   - `relu`, `leaky_relu`, `sigmoid` — Activation functions
   - `add` — Residual connections
   - `to_rgba` — Converts float RGB output to RGBA pixel data
5. **Weight Quantization**: Supports fp32, fp16, and per-channel symmetric int8 with automatic dequantization on load.
6. **WASM Fallback**: The same decoder architecture is hand-rolled in Rust with all operations (convolution, upsampling, activations) implemented from scratch.

### Loss Function

The VAE loss combines four components:

$$\mathcal{L} = \mathcal{L}_{L1} + 0.5 \cdot \mathcal{L}_{perceptual} + 0.1 \cdot \mathcal{L}_{SSIM} + \beta \cdot D_{KL}$$

- **L1 Loss**: Pixel-wise reconstruction (sharper than MSE)
- **Perceptual Loss**: VGG19 feature matching at 4 layers (relu1_2, relu2_2, relu3_4, relu4_4)
- **SSIM Loss**: Structural similarity for better perceptual quality
- **KL Divergence**: Regularizes latent space; $\beta$ warms up over `--kl_warmup_epochs`
