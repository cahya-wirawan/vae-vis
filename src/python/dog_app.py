import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define VAE Decoder Architecture (Matches src/training/dog.py)
# ==========================================

class ResBlock(nn.Module):
    """Residual block matching training architecture."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.block(x))


class DogDecoder(nn.Module):
    """Decoder for 128x128 RGB VAE. Matches training Decoder + decoder_fc."""
    def __init__(self, latent_dim=128):
        super().__init__()

        flat_size = 512 * 4 * 4  # 8192

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResBlock(512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 4, 4)
        x = self.decoder(h)
        return x.squeeze(0).permute(1, 2, 0)

# ==========================================
# 2. Build the Web App Interface
# ==========================================
st.set_page_config(page_title="Dog VAE Latent Space", layout="centered")

st.title("🐕 Dog VAE Latent Space Explorer")

LATENT_DIM = 128

@st.cache_resource
def load_dog_model():
    model = DogDecoder(latent_dim=LATENT_DIM)
    paths = [
        'dog_vae_128.pth',
        'dog_vae_128_best.pth',
        'src/training/dog_vae_128.pth',
        'src/training/dog_vae_128_best.pth',
        'src/python/dog_vae_128.pth',
        '../training/dog_vae_128.pth',
        '../training/dog_vae_128_best.pth',
    ]

    loaded = False
    for path in paths:
        try:
            state_dict = torch.load(path, map_location='cpu', weights_only=True)

            # Filter to only decoder keys and remap them
            filtered_dict = {}
            for k, v in state_dict.items():
                if k.startswith('decoder_fc.') or k.startswith('dec'):
                    filtered_dict[k] = v

            if filtered_dict:
                model.load_state_dict(filtered_dict, strict=True)
                st.sidebar.success(f"✅ Loaded weights from: {path}")
                loaded = True
                break
        except Exception:
            continue

    if not loaded:
        st.sidebar.warning(f"⚠️ No weights found for {LATENT_DIM}D model. Using random initialization.")

    model.eval()
    return model

decoder = load_dog_model()

# PCA Projection Logic
@st.cache_data
def get_pca_projection(dim):
    """Generates a stable, orthogonal projection matrix from 2D to high-D."""
    rng = np.random.RandomState(42)  # Fixed seed for stable visualization
    v1 = rng.randn(dim)
    v2 = rng.randn(dim)
    v1 /= np.linalg.norm(v1)
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)
    return v1, v2

pc1, pc2 = get_pca_projection(LATENT_DIM)

st.markdown(
    f"""
This app visualizes a **{LATENT_DIM}D** Variational Autoencoder trained on dog images (128×128 RGB).
We use **PCA Projection** to map your 2D input to the high-dimensional latent space.
"""
)

st.divider()

# Sliders for latent space
col1, col2 = st.columns(2)
with col1:
    z1 = st.slider("Principal Component 1", -15.0, 15.0, 0.0, 0.1)
with col2:
    z2 = st.slider("Principal Component 2", -15.0, 15.0, 0.0, 0.1)

# Generate and Display
st.subheader("Generated Dog")

# Map 2D slider to 128D space
z_128 = (z1 * pc1) + (z2 * pc2)
z_tensor = torch.tensor(z_128, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    generated_image = decoder(z_tensor).numpy()

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(generated_image)
ax.axis("off")
st.pyplot(fig)

st.info(f"💡 Exploring the top 2 principal axes of the {LATENT_DIM}D space.")
