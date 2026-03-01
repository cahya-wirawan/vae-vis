# VAE Latent Space Explorer (VAE-Vis)

A project to visualize and explore the continuous latent space of Variational Autoencoders (VAE) trained on MNIST (28x28 grayscale) and Dog (128x128 RGB) datasets. This repository demonstrates how to bridge deep learning models from Python/PyTorch to high-performance web applications using Rust and WebAssembly.

## Features

- **Latent Exploration**: Navigate the VAE's latent space using sliders to see how digits or dog faces morph and transition.
- **Dual Implementations**:
  - **Python/Streamlit**: A rapid prototype using PyTorch and Streamlit.
  - **Rust/WebAssembly**: A high-performance, client-side implementation.
- **WebGPU Acceleration (Path A)**: Uses `wgpu` in Rust to offload neural network inference to the GPU for near-instant 60FPS exploration of complex models.
- **Baked-in Weights**: Deep learning weights are exported from PyTorch and compiled directly into the Rust binary.

## Project Structure

```
vae-vis/
├── src/
│   ├── training/           # Model training scripts
│   ├── python/             # Python-based exploration and utilities
│   └── rust/
│       ├── vae_wasm/       # MNIST Rust WebAssembly project (CPU-based)
│       └── dog_wasm/       # Dog Rust WebAssembly project (WebGPU-enabled)
│           ├── src/
│           │   ├── lib.rs  # Rust orchestration for WebGPU
│           │   └── shaders.wgsl # WGSL Compute Shaders for NN layers
│           └── index.html  # Frontend UI
└── README.md
```

## Getting Started

### 1. Training and Exporting
Follow the instructions in the previous sections to train the model and export `dog_decoder.bin`.

### 2. Building the WebGPU Version
Requires `wasm-pack`.

```bash
cd src/rust/dog_wasm
wasm-pack build --target web
python3 -m http.server 8000
```

Open `http://localhost:8000/test_gpu.html` to verify WebGPU initialization.

## How it Works (WebGPU Path)

1.  **Orchestration**: The Rust code initializes the WebGPU device and creates compute pipelines for each layer defined in `shaders.wgsl`.
2.  **GPU Memory**: All model weights (~45MB) are uploaded to `STORAGE` buffers on the GPU once at startup.
3.  **Inference**: When the latent sliders move, Rust dispatches a sequence of compute shader passes (Linear -> Upsample -> Conv2d -> etc.), keeping all intermediate data on the GPU to avoid bandwidth bottlenecks.
