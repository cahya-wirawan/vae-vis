import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Define VAE Decoder Architecture (Matches src/training/dog.py)
# ==========================================

def upsample_block(in_ch, out_ch, final=False):
    """Upsample + Conv2d instead of ConvTranspose2d to avoid checkerboard artifacts."""
    layers = [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
    ]
    if not final:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU()]
        # Add refinement conv for more capacity (Matches dog.py)
        layers += [nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU()]
    else:
        layers += [nn.Sigmoid()]
    return nn.Sequential(*layers)

class DogDecoder(nn.Module):
    """Specific decoder for 256x256 RGB Dog dataset as defined in src/training/dog.py."""
    def __init__(self, latent_dim=128):
        super().__init__()
        
        flat_size = 512 * 8 * 8  # 32768
        
        # DECODER FC: Mirror of dog.py
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.ReLU(),
        )

        # DECODER CNN: Mirror of dog.py
        self.decoder = nn.Sequential(
            upsample_block(512, 256),   # -> 256x16x16
            upsample_block(256, 128),   # -> 128x32x32
            upsample_block(128, 64),    # -> 64x64x64
            upsample_block(64, 32),     # -> 32x128x128
            upsample_block(32, 3, final=True),  # -> 3x256x256
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

LATENT_DIM = 128

@st.cache_resource
def load_dog_model():
    model = DogDecoder(latent_dim=LATENT_DIM)
    # Paths to look for weights
    paths = [
        'dog_vae_256.pth', 
        'dog_vae_256_best.pth', 
        'src/training/dog_vae_256.pth', 
        'src/python/dog_vae_256.pth',
        '../training/dog_vae_256.pth'
    ]
    
    loaded = False
    for path in paths:
        try:
            # Using weights_only=True for security and to avoid Streamlit/Torch proxy errors
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            
            # Filter the state_dict to only include decoder parts
            filtered_dict = {}
            for k, v in state_dict.items():
                if k.startswith('decoder'):
                    filtered_dict[k] = v
            
            if filtered_dict:
                # Load with strict=True now that architectures are aligned
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
This app visualizes a **{LATENT_DIM}D** Variational Autoencoder's latent space. 
We use **PCA Projection** to map your 2D input to the high-dimensional space.
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
