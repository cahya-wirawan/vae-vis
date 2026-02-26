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
BATCH_SIZE = 32  # Smaller batch for small dataset + larger model
EPOCHS = 200  # More epochs since augmentation adds variety
LEARNING_RATE = 3e-4
LATENT_DIM = 2  # Keeping it 2D for visualization compatibility
KL_WEIGHT_MAX = 0.5  # Beta for beta-VAE (balances reconstruction vs KL)
KL_WARMUP_EPOCHS = 30  # Gradually increase KL weight over this many epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")

# Load the dataset from Hugging Face
print("Loading huggan/few-shot-dog dataset...")
dataset = load_dataset("huggan/few-shot-dog", split="train")
print(f"Dataset size: {len(dataset)} images")

# Data augmentation to combat overfitting on small dataset
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
    ]
)

# Simpler transform without augmentation for validation/export
transform_clean = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


def transform_fn(examples):
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]
    del examples["image"]
    return examples


dataset.set_transform(transform_fn)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ==========================================
# 2. Define the VAE Architecture (Convolutional)
# ==========================================
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        # ENCODER: Compresses 3x256x256 input
        # Uses BatchNorm + LeakyReLU for stable gradients
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # -> 32x128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> 64x64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # -> 128x32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # -> 256x16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # -> 512x8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),  # -> 32768
        )

        # Intermediate FC layers to ease the bottleneck (32768 -> 512 -> latent)
        flat_size = 512 * 8 * 8  # 32768
        self.fc_hidden = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # DECODER: Reconstructs 3x256x256 output
        # Mirror the encoder with intermediate FC layers
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
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # -> 3x256x256
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.fc_hidden(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstructed_x, x, mu, logvar, kl_weight=1.0):
    # Use 'mean' reduction for stable, interpretable loss values
    MSE = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    # KL Divergence: mean over batch for consistency
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return MSE + kl_weight * KLD, MSE.item(), KLD.item()


model = VAE(LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Print model parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ==========================================
# 4. Training Loop
# ==========================================
model.train()
print("Starting training...")
best_loss = float("inf")

for epoch in range(EPOCHS):
    train_loss = 0
    train_mse = 0
    train_kld = 0
    num_batches = 0

    # KL warm-up: linearly increase kl_weight from 0 to KL_WEIGHT_MAX
    kl_weight = min(1.0, epoch / KL_WARMUP_EPOCHS) * KL_WEIGHT_MAX

    for batch in train_loader:
        data = batch["pixel_values"].to(DEVICE)

        optimizer.zero_grad()
        reconstructed_batch, mu, logvar = model(data)

        loss, mse_val, kld_val = vae_loss(
            reconstructed_batch, data, mu, logvar, kl_weight
        )
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        train_loss += loss.item()
        train_mse += mse_val
        train_kld += kld_val
        num_batches += 1
        optimizer.step()

    scheduler.step()

    avg_loss = train_loss / num_batches
    avg_mse = train_mse / num_batches
    avg_kld = train_kld / num_batches

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f} | "
            f"MSE: {avg_mse:.6f} | KLD: {avg_kld:.6f} | "
            f"KL_w: {kl_weight:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "dog_vae_256_best.pth")

# ==========================================
# 5. Save Weights
# ==========================================
# Save the final model and report best result
torch.save(model.state_dict(), "dog_vae_256.pth")
print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
print("Saved 'dog_vae_256.pth' (final) and 'dog_vae_256_best.pth' (best).")
print("Note: This architecture is Convolutional and output is 256x256 RGB.")
