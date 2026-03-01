import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time
from datetime import datetime
import random
import numpy as np
import wandb

# ==========================================
# 1. Perceptual Loss (VGG Feature Matching)
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
# 2. VAE Architecture (ResNet-style, no skip connections)
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
# 3. Loss Function
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


# ==========================================
# 4. Visualization Helper
# ==========================================
@torch.no_grad()
def save_samples(model, epoch, data_batch, sample_dir, latent_dim, device, writer=None, use_wandb=False):
    """Save reconstruction + random generation samples every N epochs."""
    model.eval()
    # Reconstruct training images
    recon, _, _ = model(data_batch[:8])
    comparison = torch.cat([data_batch[:8], recon], dim=0)
    save_path = f"{sample_dir}/recon_epoch_{epoch+1:03d}.png"
    save_image(comparison, save_path, nrow=8)
    
    if writer:
        grid = make_grid(comparison, nrow=8)
        writer.add_image("Reconstruction", grid, epoch + 1)
    
    if use_wandb:
        wandb.log({"Reconstruction": wandb.Image(save_path)}, step=epoch + 1)

    # Generate from random latent vectors
    z_random = torch.randn(8, latent_dim, device=device)
    generated = model.decode(z_random)
    gen_path = f"{sample_dir}/gen_epoch_{epoch+1:03d}.png"
    save_image(generated, gen_path, nrow=8)
    
    if writer:
        grid_gen = make_grid(generated, nrow=8)
        writer.add_image("Generation", grid_gen, epoch + 1)
        
    if use_wandb:
        wandb.log({"Generation": wandb.Image(gen_path)}, step=epoch + 1)
        
    model.train()


# ==========================================
# 5. Main Training Function
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train a VAE on AFHQ Dog dataset")
    parser.add_argument("--batch_size", type=int, default=196, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=256, help="Dimension of latent space")
    parser.add_argument("--img_size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--kl_weight_max", type=float, default=0.0001, help="Max KL divergence weight")
    parser.add_argument("--kl_warmup_epochs", type=int, default=30, help="Epochs for KL weight warmup")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path (default: output/run_timestamp)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset_lr", action="store_true", help="Reset learning rate when resuming")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="vae-vis-dog", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_root = f"output/run_{timestamp}"
    else:
        output_root = args.output_dir
    
    sample_dir = os.path.join(output_root, "samples")
    checkpoint_dir = os.path.join(output_root, "checkpoints")
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {output_root}")

    # Setup Logging
    writer = SummaryWriter(log_dir=log_dir)
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"run_{timestamp}",
            config=vars(args)
        )

    # Load the AFHQ dataset from Hugging Face
    print("Loading huggan/AFHQ dataset...")
    dataset = load_dataset("huggan/AFHQ", split="train")
    print(f"Dataset size: {len(dataset)} images")

    # Data augmentation
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Initialize Model, Loss, Optimizer
    model = VAE(args.latent_dim).to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        if args.reset_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - start_epoch, eta_min=1e-6
            )
            print(f"Resuming from epoch {start_epoch}, best_loss: {best_loss:.6f}, LR reset to {args.lr}")
        else:
            print(f"Resuming from epoch {start_epoch}, best_loss: {best_loss:.6f}")
    elif args.resume:
        print(f"WARNING: Checkpoint '{args.resume}' not found, training from scratch.")

    # Training Loop
    model.train()
    print(f"Starting training from epoch {start_epoch}...")
    viz_batch = None  # Will store a fixed batch for consistent visualization

    for epoch in range(start_epoch, args.epochs):
        train_loss = 0
        train_l1 = 0
        train_perc = 0
        train_kld = 0
        num_batches = 0

        kl_weight = min(1.0, epoch / args.kl_warmup_epochs) * args.kl_weight_max

        for batch in train_loader:
            data = batch["pixel_values"].to(device)

            # Store first batch for visualization
            if viz_batch is None:
                viz_batch = data[:8].clone()

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
        current_lr = scheduler.get_last_lr()[0]

        # Log to TensorBoard
        writer.add_scalar("Loss/Total", avg_loss, epoch + 1)
        writer.add_scalar("Loss/L1", avg_l1, epoch + 1)
        writer.add_scalar("Loss/Perceptual", avg_perc, epoch + 1)
        writer.add_scalar("Loss/KLD", avg_kld, epoch + 1)
        writer.add_scalar("Meta/KL_Weight", kl_weight, epoch + 1)
        writer.add_scalar("Meta/LR", current_lr, epoch + 1)

        # Log to WandB
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss/total": avg_loss,
                "loss/l1": avg_l1,
                "loss/perceptual": avg_perc,
                "loss/kld": avg_kld,
                "meta/kl_weight": kl_weight,
                "meta/lr": current_lr,
            }, step=epoch + 1)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.6f} | "
                f"L1: {avg_l1:.6f} | Perc: {avg_perc:.4f} | KLD: {avg_kld:.4f} | "
                f"KL_w: {kl_weight:.5f} | LR: {current_lr:.2e}"
            )

        # Save visualization every 25 epochs
        if (epoch + 1) % 25 == 0 and viz_batch is not None:
            save_samples(model, epoch, viz_batch, sample_dir, args.latent_dim, device, writer, args.use_wandb)
            print(f"  → Saved samples to {sample_dir}/")

        # Save latest checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            "args": args,
        }
        torch.save(checkpoint_data, os.path.join(output_root, "latest_checkpoint.pth"))

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth"))

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_root, "best_model.pth"))

    # Save Final Weights
    torch.save(model.state_dict(), os.path.join(output_root, "final_model.pth"))
    writer.close()
    if args.use_wandb:
        wandb.finish()
    print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    print(f"Models saved in: {output_root}")

if __name__ == "__main__":
    main()
