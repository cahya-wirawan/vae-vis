import argparse
import os
import random
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import make_grid, save_image

try:
    import wandb as _wandb
    from pytorch_lightning.loggers import WandbLogger as _WandbLogger
except ImportError:
    _wandb = None
    _WandbLogger = None


# ==========================================
# 1. Perceptual Loss (VGG Feature Matching)
# ==========================================
class VGGPerceptualLoss(nn.Module):
    """Compares VGG features between prediction and target.
    Uses 3 slices (up to relu3_4) on downsampled input for efficiency."""

    PERC_SIZE = 64  # Downsample before VGG to reduce cost

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:18]))  # relu3_4
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def to_01(self, x):
        """Denormalize from [-1, 1] to [0, 1] for VGG input."""
        return x * 0.5 + 0.5

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        # Downsample to reduce VGG compute cost
        pred = nn.functional.interpolate(
            pred, size=self.PERC_SIZE, mode="bilinear", align_corners=False
        )
        target = nn.functional.interpolate(
            target, size=self.PERC_SIZE, mode="bilinear", align_corners=False
        )
        pred = self.normalize(self.to_01(pred))
        target = self.normalize(self.to_01(target))
        loss = 0.0
        x, y = pred, target
        for slicer in [self.slice1, self.slice2, self.slice3]:
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
            nn.Tanh(),
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
# 3. SSIM Loss
# ==========================================
def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """1-D Gaussian kernel (for SSIM window)."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).unsqueeze(0).unsqueeze(0)  # 1x1xN


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute mean SSIM between x and y (both Bx3xHxW in [-1,1] or [0,1]).
    Returns a scalar loss = 1 - SSIM (so lower is better)."""
    kernel_1d = _fspecial_gauss_1d(window_size, 1.5).to(x.device, x.dtype)
    C = x.shape[1]  # channels
    # Build separable 2-D Gaussian per channel
    kernel_2d = kernel_1d.T @ kernel_1d  # size x size
    window = kernel_2d.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_x = nn.functional.conv2d(x, window, padding=pad, groups=C)
    mu_y = nn.functional.conv2d(y, window, padding=pad, groups=C)
    mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x_sq = nn.functional.conv2d(x * x, window, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = nn.functional.conv2d(y * y, window, padding=pad, groups=C) - mu_y_sq
    sigma_xy = nn.functional.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return 1.0 - ssim_map.mean()  # 1 - SSIM as loss


# ==========================================
# 4. Loss Function
# ==========================================
def vae_loss(
    reconstructed_x, x, mu, logvar, perceptual_fn,
    kl_weight=1.0, perc_weight=0.5, ssim_weight=0.1,
):
    # L1 pixel loss (averaged over all elements)
    L1 = nn.functional.l1_loss(reconstructed_x, x, reduction="mean")

    # Perceptual loss (VGG features)
    P_LOSS = perceptual_fn(reconstructed_x, x)

    # SSIM structural loss
    SSIM_LOSS = ssim(reconstructed_x, x) if ssim_weight > 0 else torch.tensor(0.0)

    # KL Divergence — normalised per data dimension for scale-invariance
    # sum over latent dims, mean over batch, then divide by pixel count
    # so recon and KL are both "per pixel" and comparable across resolutions
    logvar_safe = torch.clamp(logvar, min=-10.0, max=10.0)
    n_pixels = reconstructed_x.shape[1] * reconstructed_x.shape[2] * reconstructed_x.shape[3]
    KLD = -0.5 * torch.sum(1 + logvar_safe - mu.pow(2) - logvar_safe.exp()) / (
        mu.shape[0] * n_pixels
    )

    recon_loss = L1 + perc_weight * P_LOSS + ssim_weight * SSIM_LOSS
    total = recon_loss + kl_weight * KLD
    return total, L1.item(), P_LOSS.item(), KLD.item(), (
        SSIM_LOSS.item() if ssim_weight > 0 else 0.0
    )


# ==========================================
# 5. Visualization Helpers
# ==========================================
@torch.no_grad()
def _latent_interpolation(model, latent_dim, device, n_pairs=2, n_steps=8):
    """Spherical linear interpolation between random z pairs.
    Returns a (n_pairs*n_steps) batch of decoded images in [0,1]."""
    images = []
    for _ in range(n_pairs):
        z0 = torch.randn(latent_dim, device=device)
        z1 = torch.randn(latent_dim, device=device)
        z0_n = z0 / z0.norm()
        z1_n = z1 / z1.norm()
        omega = torch.acos(torch.clamp(torch.dot(z0_n, z1_n), -1.0, 1.0))
        sin_omega = torch.sin(omega).clamp(min=1e-6)
        for t in torch.linspace(0, 1, n_steps):
            z_t = (torch.sin((1 - t) * omega) / sin_omega) * z0 + (
                torch.sin(t * omega) / sin_omega
            ) * z1
            images.append(model.decode(z_t.unsqueeze(0)))
    return torch.cat(images, dim=0) * 0.5 + 0.5  # [-1,1] → [0,1]


# ==========================================
# 6. Data Module
# ==========================================
class AFHQDataModule(pl.LightningDataModule):
    def __init__(self, img_size=128, batch_size=196, num_workers=4, seed=None):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None

    def setup(self, stage=None):
        if self.dataset is not None:
            return
        print("Loading huggan/AFHQ dataset...")
        self.dataset = load_dataset("huggan/AFHQ", split="train")
        print(f"Dataset size: {len(self.dataset)} images")

        img_size = self.hparams.img_size
        self.aug_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        def transform_fn(examples):
            examples["pixel_values"] = [
                self.aug_transform(image.convert("RGB"))
                for image in examples["image"]
            ]
            del examples["image"]
            return examples

        self.dataset.set_transform(transform_fn)

    def train_dataloader(self):
        generator = None
        worker_init_fn = None
        seed = self.hparams.seed

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

            def worker_init_fn(worker_id):
                worker_seed = seed + worker_id
                np.random.seed(worker_seed)
                random.seed(worker_seed)

        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )


# ==========================================
# 7. Lightning Module
# ==========================================
class VAELightningModule(pl.LightningModule):
    def __init__(
        self,
        latent_dim=256,
        lr=2e-4,
        epochs=500,
        kl_weight_max=0.0001,
        kl_weight_min=None,
        kl_warmup_epochs=30,
        perc_weight=0.5,
        perc_warmup_epochs=50,
        ssim_weight=0.1,
        sample_dir="samples",
        fid_every=0,
        fid_n_samples=1000,
        batch_size=196,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vae = VAE(latent_dim)
        self.perceptual_loss_fn = VGGPerceptualLoss()

        self.viz_batch = None
        self._kl_weight = 0.0
        self._perc_weight = 0.0

    def forward(self, x):
        return self.vae(x)

    def decode(self, z):
        return self.vae.decode(z)

    # ── Schedule ──────────────────────────────────────────
    def _compute_schedule_weights(self):
        """Compute KL and perceptual weights for the current epoch."""
        epoch = self.current_epoch
        hp = self.hparams

        # KL weight warmup
        kl_weight = min(1.0, epoch / max(1, hp.kl_warmup_epochs)) * hp.kl_weight_max
        if hp.kl_weight_min is not None:
            ramp = min(1.0, epoch / max(1, hp.kl_warmup_epochs))
            kl_weight = hp.kl_weight_min + (hp.kl_weight_max - hp.kl_weight_min) * ramp

        # Perceptual weight ramp: 0 → perc_weight over perc_warmup_epochs
        perc_weight = min(1.0, epoch / max(1, hp.perc_warmup_epochs)) * hp.perc_weight

        self._kl_weight = kl_weight
        self._perc_weight = perc_weight

    def on_train_epoch_start(self):
        self._compute_schedule_weights()

    # ── Training ──────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        data = batch["pixel_values"]

        # Capture first batch for fixed visualization
        if self.viz_batch is None:
            self.viz_batch = data[:8].clone()

        recon, mu, logvar = self.vae(data)
        loss, l1_val, perc_val, kld_val, ssim_val = vae_loss(
            recon, data, mu, logvar, self.perceptual_loss_fn,
            kl_weight=self._kl_weight,
            perc_weight=self._perc_weight,
            ssim_weight=self.hparams.ssim_weight,
        )

        # Log all metrics (sync_dist=True for DDP correctness)
        self.log("loss/total", loss, prog_bar=True, sync_dist=True)
        self.log("loss/L1", l1_val, sync_dist=True)
        self.log("loss/perceptual", perc_val, sync_dist=True)
        self.log("loss/SSIM", ssim_val, sync_dist=True)
        self.log("loss/KLD", kld_val, sync_dist=True)
        self.log("schedule/kl_weight", self._kl_weight)
        self.log("schedule/perc_weight", self._perc_weight)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm before optimizer step."""
        total_norm_sq = 0.0
        for p in self.vae.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        self.log("grad/norm", total_norm_sq ** 0.5)

    # ── Epoch-end hooks ───────────────────────────────────
    def on_train_epoch_end(self):
        epoch = self.current_epoch
        hp = self.hparams

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{hp.epochs}] | "
                f"KL_w: {self._kl_weight:.5f} | P_w: {self._perc_weight:.3f}"
            )

        # Save visualization every 25 epochs (rank 0 only)
        if (epoch + 1) % 25 == 0 and self.viz_batch is not None:
            if self.global_rank == 0:
                self._save_visualization(epoch)

        # FID / LPIPS evaluation
        if hp.fid_every > 0 and (epoch + 1) % hp.fid_every == 0:
            if self.global_rank == 0:
                self._compute_fid_lpips(epoch)

    # ── Visualization ─────────────────────────────────────
    @torch.no_grad()
    def _save_visualization(self, epoch):
        """Save recon grid, random samples, and slerp interpolation."""
        self.vae.eval()
        device = self.viz_batch.device
        latent_dim = self.hparams.latent_dim
        sample_dir = self.hparams.sample_dir
        os.makedirs(sample_dir, exist_ok=True)

        recon, _, _ = self.vae(self.viz_batch)
        orig_vis = self.viz_batch * 0.5 + 0.5
        recon_vis = recon * 0.5 + 0.5
        comparison = torch.cat([orig_vis, recon_vis], dim=0)
        save_image(
            comparison,
            os.path.join(sample_dir, f"recon_epoch_{epoch+1:04d}.png"),
            nrow=8,
        )

        z_random = torch.randn(8, latent_dim, device=device)
        gen_vis = self.vae.decode(z_random) * 0.5 + 0.5
        save_image(
            gen_vis,
            os.path.join(sample_dir, f"gen_epoch_{epoch+1:04d}.png"),
            nrow=8,
        )

        interp_grid = _latent_interpolation(self.vae, latent_dim, device, n_steps=8)
        save_image(
            interp_grid,
            os.path.join(sample_dir, f"interp_epoch_{epoch+1:04d}.png"),
            nrow=8,
        )

        # Log images to all loggers
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                writer = logger.experiment
                writer.add_image(
                    "samples/reconstruction", make_grid(comparison, nrow=8), epoch
                )
                writer.add_image(
                    "samples/random", make_grid(gen_vis, nrow=8), epoch
                )
                writer.add_image(
                    "samples/interpolation", make_grid(interp_grid, nrow=8), epoch
                )
            if _WandbLogger is not None and isinstance(logger, _WandbLogger):
                logger.experiment.log(
                    {
                        "samples/reconstruction": _wandb.Image(
                            make_grid(comparison, nrow=8)
                        ),
                        "samples/random": _wandb.Image(make_grid(gen_vis, nrow=8)),
                        "samples/interpolation": _wandb.Image(
                            make_grid(interp_grid, nrow=8)
                        ),
                    },
                    step=self.global_step,
                )

        print(f"  → Saved samples to {sample_dir}/")
        self.vae.train()

    # ── FID / LPIPS ───────────────────────────────────────
    @torch.no_grad()
    def _compute_fid_lpips(self, epoch):
        """Compute FID and LPIPS diversity on generated samples."""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.lpip import (
                LearnedPerceptualImagePatchSimilarity,
            )
        except ImportError:
            print(
                "  ⚠ torchmetrics not installed — skipping FID/LPIPS. "
                "Install with: pip install torchmetrics[image]"
            )
            return

        self.vae.eval()
        device = self.device
        hp = self.hparams
        n = hp.fid_n_samples

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="squeeze", normalize=True
        ).to(device)

        # Collect real images
        real_count = 0
        for data_batch in self.trainer.train_dataloader:
            imgs = data_batch["pixel_values"].to(device) * 0.5 + 0.5
            fid.update(imgs, real=True)
            real_count += imgs.shape[0]
            if real_count >= n:
                break

        # Generate fake images
        gen_count = 0
        lpips_vals = []
        while gen_count < n:
            cur_batch = min(hp.batch_size, n - gen_count)
            z = torch.randn(cur_batch, hp.latent_dim, device=device)
            fake = self.vae.decode(z) * 0.5 + 0.5
            fid.update(fake, real=False)
            if fake.shape[0] >= 2:
                half = fake.shape[0] // 2
                lpips_vals.append(
                    lpips_metric(fake[0::2][:half], fake[1::2][:half]).item()
                )
            gen_count += cur_batch

        fid_score = fid.compute().item()
        avg_lpips = np.mean(lpips_vals) if lpips_vals else 0.0

        self.log("metrics/FID", fid_score)
        self.log("metrics/LPIPS_diversity", avg_lpips)
        print(f"  → FID: {fid_score:.2f} | LPIPS diversity: {avg_lpips:.4f}")

        del fid, lpips_metric
        self.vae.train()

    # ── Optimizer / Scheduler ─────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ==========================================
# 8. CLI Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet-style VAE on AFHQ (multi-GPU via Lightning DDP)"
    )

    # Core hyperparameters
    parser.add_argument("--batch-size", type=int, default=196)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # KL weight schedule
    parser.add_argument("--kl-weight-max", type=float, default=0.0001)
    parser.add_argument(
        "--kl-weight-min", type=float, default=None,
        help="Starting KL weight (ramps to --kl-weight-max over warmup)",
    )
    parser.add_argument("--kl-warmup-epochs", type=int, default=30)

    # Loss weights
    parser.add_argument(
        "--perc-weight", type=float, default=0.5,
        help="Perceptual loss coefficient (ramp target)",
    )
    parser.add_argument(
        "--perc-warmup-epochs", type=int, default=50,
        help="Epochs to ramp perceptual weight from 0 to --perc-weight",
    )
    parser.add_argument(
        "--ssim-weight", type=float, default=0.1,
        help="SSIM loss coefficient (0 to disable)",
    )

    # Checkpointing
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs (always saves best)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: runs/<timestamp>_z<latent_dim>)",
    )

    # Resume / weight loading
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to Lightning .ckpt to resume training (model + optimizer + epoch)",
    )
    parser.add_argument(
        "--load-weights", type=str, default=None,
        help="Load VAE weights from a raw .pth state_dict (no optimizer/epoch resume)",
    )
    parser.add_argument(
        "--reset-lr", action="store_true",
        help="When using --resume, load weights only and reset optimizer/scheduler",
    )

    # Hardware / multi-GPU
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="Lightning accelerator: auto, gpu, cpu, tpu",
    )
    parser.add_argument(
        "--devices", type=str, default="auto",
        help="Devices: 'auto', count (e.g. '2'), or list (e.g. '0,1')",
    )
    parser.add_argument(
        "--strategy", type=str, default="auto",
        help="Strategy: auto, ddp, fsdp, deepspeed, etc.",
    )
    parser.add_argument(
        "--precision", type=str, default="16-mixed",
        help="Training precision: 16-mixed, bf16-mixed, 32-true",
    )

    # Data
    parser.add_argument("--num-workers", type=int, default=4)

    # Metrics
    parser.add_argument(
        "--fid-every", type=int, default=0,
        help="Compute FID/LPIPS every N epochs (0=disable; needs torchmetrics)",
    )
    parser.add_argument(
        "--fid-n-samples", type=int, default=1000,
        help="Number of generated samples for FID/LPIPS computation",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable W&B logging (pip install wandb)",
    )
    parser.add_argument("--wandb-project", type=str, default="vae-afhq")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()


# ==========================================
# 9. Main
# ==========================================
def main():
    args = parse_args()

    # Seed everything (torch, numpy, random, CUDA)
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", f"{timestamp}_z{args.latent_dim}")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Data Module ───────────────────────────────────────
    datamodule = AFHQDataModule(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Lightning Module ──────────────────────────────────
    model = VAELightningModule(
        latent_dim=args.latent_dim,
        lr=args.lr,
        epochs=args.epochs,
        kl_weight_max=args.kl_weight_max,
        kl_weight_min=args.kl_weight_min,
        kl_warmup_epochs=args.kl_warmup_epochs,
        perc_weight=args.perc_weight,
        perc_warmup_epochs=args.perc_warmup_epochs,
        ssim_weight=args.ssim_weight,
        sample_dir=sample_dir,
        fid_every=args.fid_every,
        fid_n_samples=args.fid_n_samples,
        batch_size=args.batch_size,
    )

    # Load raw .pth weights (no optimizer/epoch resume)
    if args.load_weights and os.path.isfile(args.load_weights):
        ckpt = torch.load(args.load_weights, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            # Lightning checkpoint format → extract VAE weights
            vae_state = {
                k.replace("vae.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("vae.")
            }
            model.vae.load_state_dict(vae_state)
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # Old training checkpoint format
            model.vae.load_state_dict(ckpt["model_state_dict"])
        else:
            # Raw state_dict
            model.vae.load_state_dict(ckpt)
        print(f"Loaded VAE weights from: {args.load_weights}")

    total_params = sum(p.numel() for p in model.vae.parameters())
    trainable = sum(p.numel() for p in model.vae.parameters() if p.requires_grad)
    print(f"VAE parameters: {total_params:,} (trainable: {trainable:,})")

    # ── Loggers ───────────────────────────────────────────
    loggers = [TensorBoardLogger(save_dir=out_dir, name="tb")]
    if args.wandb:
        if _WandbLogger is not None:
            loggers.append(
                _WandbLogger(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name or os.path.basename(out_dir),
                    save_dir=out_dir,
                    config=vars(args),
                )
            )
        else:
            print("WARNING: --wandb requested but wandb not installed.")

    # ── Callbacks ─────────────────────────────────────────
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        # Periodic checkpoints (keep last 3)
        ModelCheckpoint(
            dirpath=os.path.join(out_dir, "checkpoints"),
            filename="epoch_{epoch:04d}",
            every_n_epochs=args.save_every,
            save_top_k=3,
            monitor=None,  # No metric → keeps last 3 periodic saves
            save_last=True,
        ),
        # Best model by total loss
        ModelCheckpoint(
            dirpath=out_dir,
            filename=f"dog_vae_{args.latent_dim}_best",
            monitor="loss/total",
            mode="min",
            save_top_k=1,
        ),
    ]

    # ── Trainer ───────────────────────────────────────────
    # Parse devices: "auto", int string like "2", or comma-separated "0,1"
    devices = args.devices
    if devices != "auto":
        devices = (
            [int(d) for d in devices.split(",")]
            if "," in devices
            else int(devices)
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy,
        precision=args.precision,
        gradient_clip_val=1.0,
        logger=loggers,
        callbacks=callbacks,
        deterministic="warn" if args.seed is not None else False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    # ── Resume handling ───────────────────────────────────
    ckpt_path = None
    if args.resume and os.path.isfile(args.resume):
        if args.reset_lr:
            # Load model weights only; optimizer & scheduler start fresh
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded weights from {args.resume}, LR reset to {args.lr}")
        else:
            # Full resume: model + optimizer + scheduler + epoch
            ckpt_path = args.resume
            print(f"Will resume from: {args.resume}")
    elif args.resume:
        print(f"WARNING: Checkpoint '{args.resume}' not found, training from scratch.")

    # ── Train ─────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Save final raw state_dict for easy loading in dog_app.py
    if trainer.is_global_zero:
        final_path = os.path.join(out_dir, f"dog_vae_{args.latent_dim}.pth")
        torch.save(model.vae.state_dict(), final_path)
        print(f"\n✅ Training complete!")
        print(f"Final weights: {final_path}")
        print(f"Best model: {out_dir}/dog_vae_{args.latent_dim}_best.ckpt")
        print(f"Sample images: {sample_dir}")
        print(f"TensorBoard: tensorboard --logdir {out_dir}/tb")


if __name__ == "__main__":
    main()
