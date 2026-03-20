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

## Project Structure

```
vae-explorer/
├── src/
│   ├── training/                # Model training scripts
│   │   ├── mnist.py             # Trains the VAE on MNIST dataset
│   │   └── dog.py               # Trains the ResNet VAE on AFHQ dataset
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

#### MNIST Dataset
```bash
python src/training/mnist.py
```
This generates `decoder_weights.pth`.

#### AFHQ Cats Dataset
Requires `datasets`, `torchvision`, and optionally `wandb`:
```bash
pip install datasets torchvision wandb
python src/training/dog.py --latent-dim 256
```
This generates `dog_vae_256_best.pth`.

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
