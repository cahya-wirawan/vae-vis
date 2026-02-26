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
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 2e-4
LATENT_DIM = 128
IMG_SIZE = 128           # Reduced from 256: much more tractable for ~200 images
KL_WEIGHT_MAX = 0.005    # Raised from 0.0005: keeps latent space usable for generation
KL_WARMUP_EPOCHS = 80    # Slow warmup to let reconstruction stabilize first
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = "samples"   # Directory for periodic visualization

os.makedirs(SAMPLE_DIR, exist_ok=True)
print(f"Training on: {DEVICE}")

# Load the dataset from Hugging Face
print("Loading huggan/few-shot-dog dataset...")
dataset = load_dataset("huggan/few-shot-dog", split="train")
print(f"Dataset size: {len(dataset)} images")

# Data augmentation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
])

# Clean transform for visualization
transform_clean = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def transform_fn(examples):
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]
    del examples["image"]
    return examples


dataset.set_transform(transform_fn)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ==========================================
# 2. Perceptual Loss (VGG Feature Matching)
# ==========================================
class VGGPerceptualLoss(nn.Module):
    """Compares VGG19 features between prediction and target.
    This penalizes blurriness at the feature level."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:18]))  # relu3_4
        self.slice4 = nn.Sequential(*list(vgg[18:27])) # relu4_4
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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
# 3. VAE with U-Net Skip Connections
# ==========================================
class EncoderBlock(nn.Module):
    """Conv -> BN -> LeakyReLU, halves spatial dims."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample + Conv + BN + ReLU + Refine conv.
    Accepts skip connection (concatenated on channel dim)."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # in_ch + skip_ch because of the concatenated skip connection
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


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # ENCODER: 3x128x128 -> bottleneck
        # Each block halves spatial dims and stores activations for skip connections
        self.enc1 = EncoderBlock(3, 64)      # -> 64x64x64
        self.enc2 = EncoderBlock(64, 128)    # -> 128x32x32
        self.enc3 = EncoderBlock(128, 256)   # -> 256x16x16
        self.enc4 = EncoderBlock(256, 512)   # -> 512x8x8
        self.enc5 = EncoderBlock(512, 512)   # -> 512x4x4

        # Bottleneck FC: 512*4*4 = 8192
        flat_size = 512 * 4 * 4
        self.fc_hidden = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # DECODER FC: latent -> spatial
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, flat_size),
            nn.ReLU(),
        )

        # DECODER with skip connections from encoder
        # Each block doubles spatial dims and concatenates encoder features
        self.dec5 = DecoderBlock(512, 512, 512)   # 4->8,  concat enc4 (512)
        self.dec4 = DecoderBlock(512, 256, 256)   # 8->16, concat enc3 (256)
        self.dec3 = DecoderBlock(256, 128, 128)   # 16->32, concat enc2 (128)
        self.dec2 = DecoderBlock(128, 64, 64)     # 32->64, concat enc1 (64)

        # Final upsample: 64->128, no skip (input level)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
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
        # Run encoder, storing intermediate activations for skip connections
        s1 = self.enc1(x)    # 64x64x64
        s2 = self.enc2(s1)   # 128x32x32
        s3 = self.enc3(s2)   # 256x16x16
        s4 = self.enc4(s3)   # 512x8x8
        h = self.enc5(s4)    # 512x4x4

        h_flat = h.flatten(1)
        h_fc = self.fc_hidden(h_flat)
        mu = self.fc_mu(h_fc)
        logvar = torch.clamp(self.fc_logvar(h_fc), min=-10.0, max=10.0)
        return mu, logvar, [s1, s2, s3, s4]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips=None):
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 4, 4)

        if skips is not None:
            s1, s2, s3, s4 = skips
            h = self.dec5(h, s4)   # 4->8,  use enc4
            h = self.dec4(h, s3)   # 8->16, use enc3
            h = self.dec3(h, s2)   # 16->32, use enc2
            h = self.dec2(h, s1)   # 32->64, use enc1
        else:
            # For generation without encoder (sampling from latent space)
            # Use zeros as skip connections
            b = z.shape[0]
            h = self.dec5(h, torch.zeros(b, 512, 8, 8, device=z.device))
            h = self.dec4(h, torch.zeros(b, 256, 16, 16, device=z.device))
            h = self.dec3(h, torch.zeros(b, 128, 32, 32, device=z.device))
            h = self.dec2(h, torch.zeros(b, 64, 64, 64, device=z.device))

        h = self.dec1(h)       # 64->128
        return h

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skips), mu, logvar


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
    # Show original vs reconstruction side by side
    comparison = torch.cat([data_batch[:8], recon], dim=0)
    save_image(comparison, f"{SAMPLE_DIR}/recon_epoch_{epoch+1:03d}.png", nrow=8)

    # Generate from random latent vectors
    z_random = torch.randn(8, LATENT_DIM, device=DEVICE)
    generated = model.decode(z_random, skips=None)
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
