import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleDecoder, ConvolutionalDecoder

# ==========================================
# 1. Build the Web App Interface
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
        path = 'decoder_weights.pth'
        is_rgb = False
        # Default fallback if no file or metadata is found
        model = SimpleDecoder(latent_dim=2)
    else:
        path = 'dog_vae_256.pth'
        is_rgb = True
        model = ConvolutionalDecoder(latent_dim=2)
    
    try:
        # Load the saved file (could be a state_dict or a payload dict)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        
        # 1. Extract state_dict and handle metadata
        if isinstance(payload, dict) and 'state_dict' in payload:
            state_dict = payload['state_dict']
            metadata = payload.get('metadata', {})
            
            # If we have metadata for MNIST, re-instantiate with correct dims
            if m_type == "MNIST (28x28 Grayscale)" and metadata.get('architecture') == 'fc_mnist':
                model = SimpleDecoder(
                    latent_dim=metadata.get('latent_dim', 2),
                    hidden_dim1=metadata.get('hidden_dim1', 128),
                    hidden_dim2=metadata.get('hidden_dim2', 256)
                )
                st.sidebar.info(f"Loaded Metadata: Latent={metadata['latent_dim']}, H1={metadata['hidden_dim1']}, H2={metadata['hidden_dim2']}")
        else:
            # Handle legacy format where file is just the state_dict
            state_dict = payload

        # 2. Key Filtering (for full VAE checkpoints or files with different prefixes)
        filtered_dict = {}
        for k, v in state_dict.items():
            if m_type == "MNIST (28x28 Grayscale)":
                # Handle prefixes from full VAE training (dec_fc1 -> fc1)
                if k.startswith('dec_'):
                    new_key = k.replace('dec_', '')
                    filtered_dict[new_key] = v
                elif k in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
                    filtered_dict[k] = v
            else:
                # For ConvolutionalDecoder
                if k.startswith('decoder') or k.startswith('decoder_input'):
                    filtered_dict[k] = v
        
        # 3. Load weights into the model
        if not filtered_dict:
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(filtered_dict, strict=False)
            
        st.sidebar.success(f"✅ Loaded weights from: {path}")
    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ '{path}' not found. Using random weights.")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        
    model.eval()
    return model, is_rgb

decoder, is_rgb = load_vae_model(model_type)

# ==========================================
# 2. Latent Space Exploration
# ==========================================
st.markdown(
    f"""
This app visualizes the continuous nature of a Variational Autoencoder's latent space. 
Currently exploring: **{model_type}**.
"""
)

# 1. Detect Latent Dimension from the model
if model_type == "MNIST (28x28 Grayscale)":
    latent_dim = decoder.fc1.in_features
else:
    latent_dim = decoder.decoder_input.in_features

st.sidebar.divider()
st.sidebar.write(f"🔢 **Model Latent Dim:** {latent_dim}")

st.divider()

# 2. Generate Dynamic Sliders
z_values = []
if latent_dim <= 10:
    # For small latent spaces, show all dimensions
    st.write(f"Adjusting all {latent_dim} latent variables:")
    cols = st.columns(min(latent_dim, 4))
    for i in range(latent_dim):
        with cols[i % 4]:
            val = st.slider(f"z{i+1}", -4.0, 4.0, 0.0, 0.1, key=f"z{i}")
            z_values.append(val)
else:
    # For high latent spaces (like 128D), control the first 2 and zero the rest
    # (Or you could implement PCA axis selection here)
    st.info(f"💡 High-dimensional latent space ({latent_dim}D). Controlling first 2 dimensions, others set to 0.0.")
    col1, col2 = st.columns(2)
    with col1:
        z1 = st.slider("z1", -5.0, 5.0, 0.0, 0.1)
    with col2:
        z2 = st.slider("z2", -5.0, 5.0, 0.0, 0.1)
    z_values = [z1, z2] + [0.0] * (latent_dim - 2)

# 3. Generate and Display
st.subheader("Generated Output")
z_tensor = torch.tensor([z_values], dtype=torch.float32)

with torch.no_grad():
    # Model forward pass
    generated_image = decoder(z_tensor)
    # Remove batch dimension for plotting
    generated_image = generated_image.squeeze(0).cpu().numpy()

fig, ax = plt.subplots(figsize=(4, 4))
if is_rgb:
    ax.imshow(generated_image)
else:
    ax.imshow(generated_image, cmap="magma", interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
