import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define VAE Decoder Architecture (from dog.py)
# ==========================================

class DogDecoder(nn.Module):
    """Specific decoder for 256x256 RGB Dog dataset as defined in src/training/dog.py."""
    def __init__(self, latent_dim=2):
        super().__init__()
        
        flat_size = 512 * 8 * 8  # 32768
        
        # DECODER: Reconstructs 3x256x256 output
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # -> 256x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> 128x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32x128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # -> 3x256x256
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 8, 8)
        x = self.decoder(h)
        # Returns (3, 256, 256), we want (256, 256, 3) for plotting
        return x.squeeze(0).permute(1, 2, 0)

# ==========================================
# 2. Build the Web App Interface
# ==========================================
st.set_page_config(page_title="Dog VAE Latent Space", layout="centered")

st.title("🐕 Dog VAE Latent Space Explorer")

@st.cache_resource
def load_dog_model():
    model = DogDecoder()
    # Paths to look for weights
    paths = ['dog_vae_256.pth', 'dog_vae_256_best.pth', '../training/dog_vae_256.pth', 'src/python/dog_vae_256.pth']
    
    loaded = False
    for path in paths:
        try:
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            
            # Filter the state_dict to only include decoder parts
            filtered_dict = {}
            for k, v in state_dict.items():
                if k.startswith('decoder'):
                    filtered_dict[k] = v
            
            if filtered_dict:
                model.load_state_dict(filtered_dict, strict=True)
                st.sidebar.success(f"✅ Loaded weights from: {path}")
                loaded = True
                break
        except (FileNotFoundError, RuntimeError):
            continue
            
    if not loaded:
        st.sidebar.warning("⚠️ No weights found. Using random initialization.")
        
    model.eval()
    return model

decoder = load_dog_model()

st.markdown(
    """
This app visualizes the continuous nature of a Variational Autoencoder's latent space trained on the 
`huggan/few-shot-dog` dataset. Move the sliders to explore generated dogs!
"""
)

st.divider()

# Sliders for latent space
col1, col2 = st.columns(2)
with col1:
    z1 = st.slider("z1 (Feature A)", -4.0, 4.0, 0.0, 0.1)
with col2:
    z2 = st.slider("z2 (Feature B)", -4.0, 4.0, 0.0, 0.1)

# Generate and Display
st.subheader("Generated Dog")
z_tensor = torch.tensor([[z1, z2]], dtype=torch.float32)

with torch.no_grad():
    generated_image = decoder(z_tensor).numpy()

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(generated_image)
ax.axis("off")
st.pyplot(fig)

st.info("💡 The latent space is 2D, so each point (z1, z2) maps to a unique 256x256 dog image.")
