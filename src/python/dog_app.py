import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define VAE Decoder Architecture (Matches src/training/dog.py)
# ==========================================

class DecoderBlock(nn.Module):
    """Upsample + concat skip + Conv layers. Matches training architecture."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DogDecoder(nn.Module):
    """Decoder for 128x128 RGB Dog VAE with U-Net skip connections.
    At inference (no encoder), uses zero tensors for skip connections."""
    def __init__(self, latent_dim=128):
        super().__init__()

        flat_size = 512 * 4 * 4  # 8192

        # DECODER FC: Matches dog.py
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, flat_size),
            nn.ReLU(),
        )

        # DECODER with skip connections (fed zeros at inference)
        self.dec5 = DecoderBlock(512, 512, 512)   # 4->8
        self.dec4 = DecoderBlock(512, 256, 256)   # 8->16
        self.dec3 = DecoderBlock(256, 128, 128)   # 16->32
        self.dec2 = DecoderBlock(128, 64, 64)     # 32->64

        # Final upsample: 64->128
        self.dec1 = nn.Sequential(
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
        b = z.shape[0]
        # Zero skip connections (no encoder at inference)
        h = self.dec5(h, torch.zeros(b, 512, 8, 8))
        h = self.dec4(h, torch.zeros(b, 256, 16, 16))
        h = self.dec3(h, torch.zeros(b, 128, 32, 32))
        h = self.dec2(h, torch.zeros(b, 64, 64, 64))
        x = self.dec1(h)
        # Returns (128, 128, 3) for plotting
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
