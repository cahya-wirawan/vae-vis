import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
BATCH_SIZE = 16  # Reduced batch size for larger 256x256 images
EPOCHS = 100     # Higher resolution needs more training
LEARNING_RATE = 1e-4
LATENT_DIM = 2   # Keeping it 2D for visualization compatibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")

# Load the dataset from Hugging Face
print("Loading huggan/few-shot-dog dataset...")
dataset = load_dataset("huggan/few-shot-dog", split="train")

# Updated transforms: Resize to 256x256 and keep RGB channels
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def transform_fn(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset.set_transform(transform_fn)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 2. Define the VAE Architecture (Convolutional)
# ==========================================
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        
        # ENCODER: Compresses 3x256x256 input
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # -> 32x128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> 64x64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # -> 128x32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# -> 256x16x16
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Flattened size: 256 * 16 * 16 = 65536
        self.fc_mu = nn.Linear(65536, latent_dim)
        self.fc_logvar = nn.Linear(65536, latent_dim)
        
        # DECODER: Reconstructs 3x256x256 output
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

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 16, 16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(reconstructed_x, x, mu, logvar):
    # Binary Cross Entropy over all pixels and channels
    BCE = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = VAE(LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. Training Loop
# ==========================================
model.train()
print("Starting training...")
for epoch in range(EPOCHS):
    train_loss = 0
    for batch in train_loader:
        data = batch["pixel_values"].to(DEVICE)
        
        optimizer.zero_grad()
        reconstructed_batch, mu, logvar = model(data)
        
        loss = vae_loss(reconstructed_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    avg_loss = train_loss / len(dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")

# ==========================================
# 5. Save Weights
# ==========================================
# Save the full model state dict since the architecture has changed significantly
torch.save(model.state_dict(), 'dog_vae_256.pth')
print("\n✅ Training complete! Saved 'dog_vae_256.pth'.")
print("Note: This architecture is Convolutional and output is 256x256 RGB.")
