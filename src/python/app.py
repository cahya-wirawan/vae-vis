import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Define the VAE Decoder Architecture
# ==========================================
class SimpleDecoder(nn.Module):
    """
    A minimal decoder that takes a 2D latent vector (z1, z2)
    and scales it up to a 28x28 pixel image (like MNIST).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 28 * 28)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(
            self.fc3(x)
        )  # Sigmoid pushes values between 0 and 1 (pixel intensity)
        return x.view(28, 28)


# Cache the model so the random weights stay the same as you move the sliders
@st.cache_resource
def load_model():
    model = SimpleDecoder()
    
    # LOAD YOUR TRAINED WEIGHTS HERE
    try:
        model.load_state_dict(torch.load('decoder_weights.pth'))
        st.sidebar.success("✅ Loaded trained weights!")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ 'decoder_weights.pth' not found. Using random weights.")
        
    model.eval()
    return model

def load_model_simple():
    model = SimpleDecoder()
    model.eval()  # Set to evaluation mode
    return model


decoder = load_model()

# ==========================================
# 2. Build the Web App Interface
# ==========================================
st.set_page_config(page_title="VAE Latent Space Explorer", layout="centered")

st.title("🌌 VAE Latent Space Explorer")
st.markdown(
    """
This app visualizes the continuous nature of a Variational Autoencoder's latent space. 
Adjust the $z_1$ and $z_2$ sliders below to navigate the 2D plane and watch how the decoder generates a new image based on your coordinates.

*Note: This decoder uses random initialization to demonstrate the concept. You will see smooth, continuous noise patterns morphing into each other.*
"""
)

st.divider()

# Create two columns for the sliders
col1, col2 = st.columns(2)

with col1:
    z1 = st.slider(
        "Latent Variable $z_1$ (X-axis)",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
    )

with col2:
    z2 = st.slider(
        "Latent Variable $z_2$ (Y-axis)",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
    )

# ==========================================
# 3. Generate and Display the Image
# ==========================================
st.subheader("Generated Output")

# Convert the slider values into a PyTorch tensor
z_tensor = torch.tensor([[z1, z2]], dtype=torch.float32)

# Pass the latent vector through the decoder
with torch.no_grad():
    generated_image = decoder(z_tensor).numpy()

# Display using Matplotlib
fig, ax = plt.subplots(figsize=(4, 4))
# Use a colormap to make the patterns easier to see
cax = ax.imshow(generated_image, cmap="magma", interpolation="bilinear")
ax.axis("off")  # Hide axes

# Render the plot in Streamlit
st.pyplot(fig)

st.divider()
st.caption(
    "To use this with a real dataset (like MNIST or CelebA), replace `SimpleDecoder` with your trained PyTorch model and load your saved weights (`.pth` file)."
)
