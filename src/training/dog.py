import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms, models
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

# ==========================================
# 1. Hyperparameters & Setup
# ==========================================
BATCH_SIZE = 196
EPOCHS = 100
LEARNING_RATE = 2e-4
LATENT_DIM = 128
IMG_SIZE = 128
KL_WEIGHT_MAX = 0.0001  # Very low: prioritize reconstruction quality first
KL_WARMUP_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = "samples"

os.makedirs(SAMPLE_DIR, exist_ok=True)
print(f"Training on: {DEVICE}")

# Load the AFHQ dataset from Hugging Face
print("Loading huggan/AFHQ dataset...")
dataset = load_dataset("huggan/AFHQ", split="train")
print(f"Dataset size: {len(dataset)} images")

# Data augmentation (lighter — dataset is much larger now)
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ]
)

# Clean transform for visualization
transform_clean = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
)


# ==========================================
# 2. Perceptual Loss (VGG Feature Matching)
# ==========================================
class VGGPerceptualLoss(nn.Module):
    """Compares VGG19 features between prediction and target.
    This penalizes blurriness at the feature level."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:18]))  # relu3_4
        self.slice4 = nn.Sequential(*list(vgg[18:27]))  # relu4_4
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self.normalize(pred)
        target = self.normalize(target)
        loss = 0.0
        x, y = pred, target
        for slicer in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = slicer(x)
            with torch.no_grad():
                y = slicer(y)
            loss += nn.functional.l1_loss(x, y)
        return loss


# ==========================================
# 3. VAE Architecture (ResNet-style, no skip connections)
# ==========================================
class ResBlock(nn.Module):
    """Residual block: adds input back to output for better gradient flow."""

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


class Encoder(nn.Module):
    """Downsamples 3x128x128 → flat vector. No skip connections stored."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 3x128x128 -> 64x64x64
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResBlock(64),
            # 64x64x64 -> 128x32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResBlock(128),
            # 128x32x32 -> 256x16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResBlock(256),
            # 256x16x16 -> 512x8x8
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            ResBlock(512),
            # 512x8x8 -> 512x4x4
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Upsamples 512x4x4 → 3x128x128. All information comes from latent z only."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 512x4x4 -> 512x8x8
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResBlock(512),
            # 512x8x8 -> 256x16x16
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            # 256x16x16 -> 128x32x32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            # 128x32x32 -> 64x64x64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            # 64x64x64 -> 3x128x128
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        # Bottleneck FC: 512*4*4 = 8192
        flat_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

        # Decoder FC: latent -> spatial
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_size),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.flatten(1)
        mu = self.fc_mu(h_flat)
        logvar = torch.clamp(self.fc_logvar(h_flat), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ==========================================
# 4. Loss Function
# ==========================================
def vae_loss(reconstructed_x, x, mu, logvar, perceptual_fn, kl_weight=1.0):
    # L1 loss: sharper than MSE
    L1 = nn.functional.l1_loss(reconstructed_x, x, reduction="mean")
    # Perceptual loss
    P_LOSS = perceptual_fn(reconstructed_x, x)
    # KL Divergence
    logvar_safe = torch.clamp(logvar, min=-10.0, max=10.0)
    KLD = -0.5 * torch.mean(
        torch.sum(1 + logvar_safe - mu.pow(2) - logvar_safe.exp(), dim=1)
    )
    recon_loss = L1 + 0.5 * P_LOSS
    total = recon_loss + kl_weight * KLD
    return total, L1.item(), P_LOSS.item(), KLD.item()


model = VAE(LATENT_DIM).to(DEVICE)
perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")


# ==========================================
# 5. Visualization Helper
# ==========================================
@torch.no_grad()
def save_samples(model, epoch, data_batch):
    """Save reconstruction + random generation samples every N epochs."""
    model.eval()
    # Reconstruct training images
    recon, _, _ = model(data_batch[:8])
    comparison = torch.cat([data_batch[:8], recon], dim=0)
    save_image(comparison, f"{SAMPLE_DIR}/recon_epoch_{epoch+1:03d}.png", nrow=8)

    # Generate from random latent vectors
    z_random = torch.randn(8, LATENT_DIM, device=DEVICE)
    generated = model.decode(z_random)
    save_image(generated, f"{SAMPLE_DIR}/gen_epoch_{epoch+1:03d}.png", nrow=8)
    model.train()


# ==========================================
# 6. Training Loop
# ==========================================
model.train()
print("Starting training...")
best_loss = float("inf")
viz_batch = None  # Will store a fixed batch for consistent visualization

for epoch in range(EPOCHS):
    train_loss = 0
    train_l1 = 0
    train_perc = 0
    train_kld = 0
    num_batches = 0

    kl_weight = min(1.0, epoch / KL_WARMUP_EPOCHS) * KL_WEIGHT_MAX

    for batch in train_loader:
        data = batch["pixel_values"].to(DEVICE)

        # Store first batch for visualization
        if viz_batch is None:
            viz_batch = data.clone()

        optimizer.zero_grad()
        reconstructed_batch, mu, logvar = model(data)

        loss, l1_val, perc_val, kld_val = vae_loss(
            reconstructed_batch, data, mu, logvar, perceptual_loss_fn, kl_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        train_loss += loss.item()
        train_l1 += l1_val
        train_perc += perc_val
        train_kld += kld_val
        num_batches += 1
        optimizer.step()

    scheduler.step()

    avg_loss = train_loss / num_batches
    avg_l1 = train_l1 / num_batches
    avg_perc = train_perc / num_batches
    avg_kld = train_kld / num_batches

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f} | "
            f"L1: {avg_l1:.6f} | Perc: {avg_perc:.4f} | KLD: {avg_kld:.4f} | "
            f"KL_w: {kl_weight:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    # Save visualization every 25 epochs
    if (epoch + 1) % 25 == 0 and viz_batch is not None:
        save_samples(model, epoch, viz_batch)
        print(f"  → Saved samples to {SAMPLE_DIR}/")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "dog_vae_128_best.pth")

# ==========================================
# 7. Save Final Weights
# ==========================================
torch.save(model.state_dict(), "dog_vae_128.pth")
print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
print(f"Model saved: dog_vae_128.pth / dog_vae_128_best.pth")
print(f"Sample images in: {SAMPLE_DIR}/")
