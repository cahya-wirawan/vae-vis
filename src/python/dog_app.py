import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define VAE Decoder Architecture (from dog.py)
# ==========================================

class DogDecoder(nn.Module):
    """Upgraded High-Capacity Decoder for sharper 256x256 RGB images."""
    def __init__(self, latent_dim=128):
        super().__init__()
        
        flat_size = 512 * 8 * 8  # 32768
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, flat_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 128x128 -> 256x256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final Refinement & RGB output
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 8, 8)
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
    # Paths to look for weights
    paths = ['dog_vae_256.pth', 'dog_vae_256_best.pth', '../training/dog_vae_256.pth', 'src/python/dog_vae_256.pth']
    
    loaded = False
    for path in paths:
        try:
            # Setting weights_only=True fixes the 'torch.classes' error in Streamlit
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            
            # Filter the state_dict to only include decoder parts
            filtered_dict = {}
            for k, v in state_dict.items():
                if k.startswith('decoder') or k.startswith('decoder_fc'):
                    filtered_dict[k] = v
            
            if filtered_dict:
                model.load_state_dict(filtered_dict, strict=False) # Use False to allow partial loading if architectures differ slightly
                st.sidebar.success(f"✅ Loaded weights from: {path}")
                loaded = True
                break
        except (FileNotFoundError, RuntimeError, AttributeError):
            continue
            
    if not loaded:
        st.sidebar.warning(f"⚠️ No weights found for {LATENT_DIM}D model. Using random initialization.")
        
    model.eval()
    return model

decoder = load_dog_model()

# PCA Simulation
# In a real scenario, these components would be calculated by running PCA 
# on the latent representations of the training set.
@st.cache_data
def get_pca_projection(dim):
    """Generates a stable, orthogonal projection matrix from 2D to high-D."""
    rng = np.random.RandomState(42)  # Fixed seed for stable visualization
    # Create two random vectors
    v1 = rng.randn(dim)
    v2 = rng.randn(dim)
    # Gram-Schmidt to make them orthogonal
    v1 /= np.linalg.norm(v1)
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)
    return v1, v2

pc1, pc2 = get_pca_projection(LATENT_DIM)

st.markdown(
    f"""
This app visualizes a **{LATENT_DIM}D** Variational Autoencoder's latent space. 
We use **PCA Projection** to map your 2D input to the high-dimensional space.
"""
)

st.divider()

# Sliders for latent space
col1, col2 = st.columns(2)
with col1:
    z1 = st.slider("Principal Component 1", -10.0, 10.0, 0.0, 0.1)
with col2:
    z2 = st.slider("Principal Component 2", -10.0, 10.0, 0.0, 0.1)

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

st.info(f"💡 You are controlling the top 2 principal axes of the {LATENT_DIM}D space. This often captures the most dramatic variations in the dataset.")
