import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Define VAE Decoder Architectures
# ==========================================

class SimpleDecoder(nn.Module):
    """Original decoder for 28x28 grayscale MNIST."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 28 * 28)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(28, 28)

class ConvolutionalDecoder(nn.Module):
    """New decoder for 256x256 RGB Dog dataset."""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.decoder_input = nn.Linear(latent_dim, 65536)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # -> 128x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 32x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # -> 3x256x256
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 16, 16)
        x = self.decoder(h)
        # Returns (3, 256, 256), we want (256, 256, 3) for plotting
        return x.squeeze(0).permute(1, 2, 0)

# ==========================================
# 2. Build the Web App Interface
# ==========================================
st.set_page_config(page_title="VAE Latent Space Explorer", layout="centered")

st.title("🌌 VAE Latent Space Explorer")

# Model Selection
model_type = st.sidebar.selectbox(
    "Choose Model",
    ["MNIST (28x28 Grayscale)", "Dogs (256x256 RGB)"]
)

@st.cache_resource
def load_vae_model(m_type):
    if m_type == "MNIST (28x28 Grayscale)":
        model = SimpleDecoder()
        path = 'decoder_weights.pth'
        is_rgb = False
    else:
        model = ConvolutionalDecoder()
        path = 'dog_vae_256.pth'
        is_rgb = True
    
    try:
        state_dict = torch.load(path, map_location='cpu')
        # Handle case where full VAE state dict is saved (like in dog.py)
        # and we only want the decoder parts.
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith('decoder') or k.startswith('fc3') or k.startswith('fc2') or k.startswith('fc1'):
                # For ConvolutionalDecoder, keys match 'decoder_input' and 'decoder'
                # For SimpleDecoder, they match 'fc1', 'fc2', 'fc3'
                filtered_dict[k] = v
        
        # If we didn't find specific prefixes, just try loading directly
        if not filtered_dict:
            model.load_state_dict(state_dict)
        else:
            # Check if we are loading into SimpleDecoder or ConvolutionalDecoder
            # ConvolutionalDecoder expects 'decoder_input' and 'decoder'
            # SimpleDecoder expects 'fc1', 'fc2', 'fc3'
            model.load_state_dict(filtered_dict, strict=False)
            
        st.sidebar.success(f"✅ Loaded {path}")
    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ '{path}' not found. Using random weights.")
        
    model.eval()
    return model, is_rgb

decoder, is_rgb = load_vae_model(model_type)

st.markdown(
    f"""
This app visualizes the continuous nature of a Variational Autoencoder's latent space. 
Currently exploring: **{model_type}**.
"""
)

st.divider()

# Sliders for latent space
col1, col2 = st.columns(2)
with col1:
    z1 = st.slider("z1 (X-axis)", -3.0, 3.0, 0.0, 0.1)
with col2:
    z2 = st.slider("z2 (Y-axis)", -3.0, 3.0, 0.0, 0.1)

# Generate and Display
st.subheader("Generated Output")
z_tensor = torch.tensor([[z1, z2]], dtype=torch.float32)

with torch.no_grad():
    generated_image = decoder(z_tensor).numpy()

fig, ax = plt.subplots(figsize=(4, 4))
if is_rgb:
    ax.imshow(generated_image)
else:
    ax.imshow(generated_image, cmap="magma", interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
