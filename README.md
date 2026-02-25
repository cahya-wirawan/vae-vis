# VAE Latent Space Explorer (VAE-Vis)

A project to visualize and explore the continuous 2D latent space of a Variational Autoencoder (VAE) trained on the MNIST dataset. This repository demonstrates how to bridge deep learning models from Python/PyTorch to high-performance web applications using Rust and WebAssembly.

## Features

- **2D Latent Exploration**: Navigate the VAE's latent space using sliders to see how digits morph and transition.
- **Dual Implementations**:
  - **Python/Streamlit**: A rapid prototype using PyTorch and Streamlit.
  - **Rust/WebAssembly**: A high-performance, client-side implementation where the neural network decoder is hand-rolled in Rust and compiled to WASM.
- **Baked-in Weights**: Deep learning weights are exported from PyTorch and compiled directly into the Rust binary for instant, serverless inference in the browser.

## Project Structure

```
vae-vis/
├── src/
│   ├── training/           # Model training scripts
│   │   └── mnist.py        # Trains the VAE on MNIST dataset
│   ├── python/             # Python-based exploration and utilities
│   │   ├── app.py          # Streamlit visualization app
│   │   └── export_weights.py # Utility to export PyTorch weights to Rust
│   └── rust/
│       └── vae_wasm/       # Rust WebAssembly project
│           ├── src/
│           │   ├── lib.rs  # Hand-rolled decoder implementation
│           │   └── weights.rs # Generated model weights (constant arrays)
│           └── index.html  # Frontend UI for the WASM app
└── README.md
```

## Getting Started

### 1. Training the Model

Train the VAE on either the MNIST dataset or the Dog dataset.

#### MNIST Dataset
```bash
# Run from root
python src/training/mnist.py
```
This generates `decoder_weights.pth`.

#### Dog Dataset (Hugging Face)
Requires the `datasets` library: `pip install datasets`
```bash
# Run from root
python src/training/dog.py
```
This generates `dog_decoder_weights.pth`. To use it in the explorer, rename it:
```bash
cp dog_decoder_weights.pth decoder_weights.pth
```

### 2. Exporting Weights to Rust

To use the trained weights in the WebAssembly version, convert them into a Rust source file. The export script should be run from the Rust project directory.

```bash
# Ensure the weights are in the right place
cp decoder_weights.pth src/rust/vae_wasm/

# Move to the Rust project directory
cd src/rust/vae_wasm

# Run the export script
python ../../python/export_weights.py
```

This updates `src/rust/vae_wasm/src/weights.rs` with the trained weights.

### 3. Running the Visualization

#### Option A: Python (Streamlit)

The Streamlit app expects `decoder_weights.pth` in the same directory as `app.py`.

```bash
cp decoder_weights.pth src/python/
streamlit run src/python/app.py
```

#### Option B: Rust + WebAssembly

Requires `wasm-pack` and a simple local web server.

```bash
# Move to the Rust project directory
cd src/rust/vae_wasm

# Build the WebAssembly package
wasm-pack build --target web

# Serve the directory
python3 -m http.server 8000
```

Open `http://localhost:8000` in your browser.

## How it Works

1.  **Encoder**: Compresses 28x28 grayscale images into a 2D latent space ($z_1, z_2$).
2.  **Decoder**: Takes a 2D coordinate and reconstructs it back into a 28x28 image.
3.  **WASM Optimization**: The Rust implementation avoids heavy dependencies. It performs the matrix multiplications and activations (ReLU, Sigmoid) manually using the weights baked into the source code at compile time.
