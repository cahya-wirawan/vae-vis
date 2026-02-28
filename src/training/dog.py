import argparse
import glob
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import make_grid, save_image

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


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
# 4. CLI Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet-style VAE on AFHQ animal faces"
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
        "--kl-weight-min",
        type=float,
        default=None,
        help="Set when resuming to ramp from old KL weight to --kl-weight-max",
    )
    parser.add_argument("--kl-warmup-epochs", type=int, default=30)

    # Loss weights
    parser.add_argument(
        "--perc-weight", type=float, default=0.5,
        help="Perceptual loss coefficient (ramp target)",
    )
    parser.add_argument(
        "--perc-warmup-epochs", type=int, default=50,
        help="Epochs to linearly ramp perceptual weight from 0 to --perc-weight",
    )
    parser.add_argument(
        "--ssim-weight", type=float, default=0.1,
        help="SSIM loss coefficient (0 to disable)",
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs (always saves best)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: runs/<timestamp>_z<latent_dim>)",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth to resume from",
    )
    parser.add_argument(
        "--reset-lr",
        action="store_true",
        help="Reset LR schedule when resuming (use --lr value)",
    )

    # Data
    parser.add_argument("--num-workers", type=int, default=4)

    # Metrics
    parser.add_argument(
        "--fid-every", type=int, default=0,
        help="Compute FID/LPIPS every N epochs (0 to disable; requires torchmetrics)",
    )
    parser.add_argument(
        "--fid-n-samples", type=int, default=1000,
        help="Number of generated samples for FID/LPIPS computation",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging (requires `pip install wandb`)",
    )
    parser.add_argument("--wandb-project", type=str, default="vae-afhq")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()


# ==========================================
# 5. Visualization Helpers
# ==========================================
@torch.no_grad()
def save_samples(model, epoch, data_batch, latent_dim, device, sample_dir, writer=None):
    """Save reconstruction + random generation + interpolation grids."""
    model.eval()
    recon, _, _ = model(data_batch[:8])
    # Denormalize from [-1,1] to [0,1] for saving / TB
    orig_vis = data_batch[:8] * 0.5 + 0.5
    recon_vis = recon * 0.5 + 0.5
    comparison = torch.cat([orig_vis, recon_vis], dim=0)
    save_image(
        comparison,
        os.path.join(sample_dir, f"recon_epoch_{epoch+1:04d}.png"),
        nrow=8,
    )

    z_random = torch.randn(8, latent_dim, device=device)
    generated = model.decode(z_random)
    gen_vis = generated * 0.5 + 0.5
    save_image(
        gen_vis,
        os.path.join(sample_dir, f"gen_epoch_{epoch+1:04d}.png"),
        nrow=8,
    )

    # Latent interpolation: slerp between two random z vectors, 8 steps
    interp_grid = _latent_interpolation(model, latent_dim, device, n_steps=8)
    save_image(
        interp_grid,
        os.path.join(sample_dir, f"interp_epoch_{epoch+1:04d}.png"),
        nrow=8,
    )

    # Log image grids to TensorBoard
    if writer is not None:
        writer.add_image("samples/reconstruction", make_grid(comparison, nrow=8), epoch)
        writer.add_image("samples/random", make_grid(gen_vis, nrow=8), epoch)
        writer.add_image("samples/interpolation", make_grid(interp_grid, nrow=8), epoch)

    # Log to W&B
    if _wandb is not None and _wandb.run is not None:
        _wandb.log({
            "samples/reconstruction": _wandb.Image(make_grid(comparison, nrow=8)),
            "samples/random": _wandb.Image(make_grid(gen_vis, nrow=8)),
            "samples/interpolation": _wandb.Image(make_grid(interp_grid, nrow=8)),
        }, step=epoch)

    model.train()


@torch.no_grad()
def _latent_interpolation(model, latent_dim, device, n_pairs=2, n_steps=8):
    """Spherical linear interpolation between random z pairs.
    Returns a (n_pairs*n_steps) batch of decoded images in [0,1]."""
    images = []
    for _ in range(n_pairs):
        z0 = torch.randn(latent_dim, device=device)
        z1 = torch.randn(latent_dim, device=device)
        # Normalise for slerp
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


def _compute_gradient_norm(model) -> float:
    """Total L2 gradient norm across all parameters (for logging)."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


# ==========================================
# 6. Main
# ==========================================
def main():
    args = parse_args()

    # Seed & determinism
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", f"{timestamp}_z{args.latent_dim}")
    sample_dir = os.path.join(out_dir, "samples")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── TensorBoard ───────────────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    # Log hyperparameters as text for easy reference
    writer.add_text("hparams", str(vars(args)), 0)

    # ── Weights & Biases ──────────────────────────────────
    use_wandb = args.wandb and _wandb is not None
    if args.wandb and _wandb is None:
        print("WARNING: --wandb requested but `wandb` not installed. Skipping.")
    if use_wandb:
        _wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or os.path.basename(out_dir),
            config=vars(args),
            dir=out_dir,
        )
        _wandb.watch(model, log="gradients", log_freq=100)
        print(f"W&B run: {_wandb.run.url}")

    # ── Data ──────────────────────────────────────────────
    print("Loading huggan/AFHQ dataset...")
    dataset = load_dataset("huggan/AFHQ", split="train")
    print(f"Dataset size: {len(dataset)} images")

    img_size = args.img_size
    aug_transform = transforms.Compose(
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
            aug_transform(image.convert("RGB")) for image in examples["image"]
        ]
        del examples["image"]
        return examples

    dataset.set_transform(transform_fn)

    # Seeded DataLoader for reproducibility
    loader_generator = None
    worker_init = None
    if args.seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)

        def worker_init(worker_id):
            worker_seed = args.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        generator=loader_generator,
        worker_init_fn=worker_init,
    )

    # ── Model / Optimizer / Scheduler / AMP ───────────────
    model = VAE(args.latent_dim).to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # ── Resume ────────────────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")
    resumed_kl_weight = None  # Will hold last kl_weight from checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        resumed_kl_weight = checkpoint.get("kl_weight", None)
        if args.reset_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - start_epoch, eta_min=1e-6
            )
            print(
                f"Resuming from epoch {start_epoch}, best_loss: {best_loss:.6f}, "
                f"LR reset to {args.lr}"
            )
        else:
            print(f"Resuming from epoch {start_epoch}, best_loss: {best_loss:.6f}")
        if resumed_kl_weight is not None:
            print(f"  Restored kl_weight: {resumed_kl_weight:.6f}")
    elif args.resume:
        print(f"WARNING: Checkpoint '{args.resume}' not found, training from scratch.")

    # ── Training Loop ─────────────────────────────────────
    model.train()
    print(f"Starting training from epoch {start_epoch}...")
    viz_batch = None

    for epoch in range(start_epoch, args.epochs):
        train_loss = 0
        train_l1 = 0
        train_perc = 0
        train_kld = 0
        train_ssim = 0
        train_grad_norm = 0
        num_batches = 0

        # KL weight schedule
        kl_weight = min(1.0, epoch / args.kl_warmup_epochs) * args.kl_weight_max
        # Gradual KL ramp when resuming with a higher kl-weight-max
        if args.kl_weight_min is not None and start_epoch > 0:
            ramp_progress = min(
                1.0, (epoch - start_epoch) / args.kl_warmup_epochs
            )
            kl_weight = (
                args.kl_weight_min
                + (args.kl_weight_max - args.kl_weight_min) * ramp_progress
            )
        # If resuming and kl-weight-min not set, use checkpoint kl_weight as
        # starting floor for the warmup schedule (avoids sudden jumps)
        elif resumed_kl_weight is not None and epoch == start_epoch:
            kl_weight = resumed_kl_weight

        # Perceptual weight ramp: 0 → perc_weight over perc_warmup_epochs
        perc_weight = (
            min(1.0, epoch / max(1, args.perc_warmup_epochs)) * args.perc_weight
        )

        for batch in train_loader:
            data = batch["pixel_values"].to(device)

            if viz_batch is None:
                viz_batch = data.clone()

            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                reconstructed_batch, mu, logvar = model(data)
                loss, l1_val, perc_val, kld_val, ssim_val = vae_loss(
                    reconstructed_batch, data, mu, logvar, perceptual_loss_fn,
                    kl_weight=kl_weight, perc_weight=perc_weight,
                    ssim_weight=args.ssim_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Track gradient norm (after unscale, before step)
            grad_norm = _compute_gradient_norm(model)

            train_loss += loss.item()
            train_l1 += l1_val
            train_perc += perc_val
            train_kld += kld_val
            train_ssim += ssim_val
            train_grad_norm += grad_norm
            num_batches += 1
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        avg_loss = train_loss / num_batches
        avg_l1 = train_l1 / num_batches
        avg_perc = train_perc / num_batches
        avg_kld = train_kld / num_batches
        avg_ssim = train_ssim / num_batches
        avg_grad_norm = train_grad_norm / num_batches
        current_lr = scheduler.get_last_lr()[0]

        # ── TensorBoard scalars ───────────────────────────
        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/L1", avg_l1, epoch)
        writer.add_scalar("loss/perceptual", avg_perc, epoch)
        writer.add_scalar("loss/SSIM", avg_ssim, epoch)
        writer.add_scalar("loss/KLD", avg_kld, epoch)
        writer.add_scalar("schedule/kl_weight", kl_weight, epoch)
        writer.add_scalar("schedule/perc_weight", perc_weight, epoch)
        writer.add_scalar("schedule/lr", current_lr, epoch)
        writer.add_scalar("grad/norm", avg_grad_norm, epoch)

        # ── W&B scalars ───────────────────────────────────
        if use_wandb:
            _wandb.log({
                "loss/total": avg_loss,
                "loss/L1": avg_l1,
                "loss/perceptual": avg_perc,
                "loss/SSIM": avg_ssim,
                "loss/KLD": avg_kld,
                "schedule/kl_weight": kl_weight,
                "schedule/perc_weight": perc_weight,
                "schedule/lr": current_lr,
                "grad/norm": avg_grad_norm,
                "epoch": epoch,
            }, step=epoch)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.6f} | "
                f"L1: {avg_l1:.6f} | Perc: {avg_perc:.4f} | SSIM: {avg_ssim:.4f} | "
                f"KLD: {avg_kld:.4f} | KL_w: {kl_weight:.5f} | "
                f"P_w: {perc_weight:.3f} | GradN: {avg_grad_norm:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        # Save visualization every 25 epochs
        if (epoch + 1) % 25 == 0 and viz_batch is not None:
            save_samples(
                model, epoch, viz_batch, args.latent_dim, device, sample_dir,
                writer=writer,
            )
            print(f"  → Saved samples to {sample_dir}/")

        # Optional FID / LPIPS evaluation
        if args.fid_every > 0 and (epoch + 1) % args.fid_every == 0:
            _log_fid_lpips(model, train_loader, args, device, epoch, writer)

        # Save checkpoint every N epochs (keep last 3)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "kl_weight": kl_weight,
                "best_loss": best_loss,
            }
            ckpt_path = os.path.join(
                ckpt_dir, f"checkpoint_epoch{epoch+1:04d}.pth"
            )
            torch.save(checkpoint_data, ckpt_path)
            ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pth")))
            for old_ckpt in ckpts[:-3]:
                os.remove(old_ckpt)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, f"dog_vae_{args.latent_dim}_best.pth"),
            )

    # ── Save Final Weights ────────────────────────────────
    torch.save(
        model.state_dict(),
        os.path.join(out_dir, f"dog_vae_{args.latent_dim}.pth"),
    )
    writer.close()
    if use_wandb:
        _wandb.finish()
    print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    print(f"Models saved in: {out_dir}")
    print(f"Sample images in: {sample_dir}")
    print(f"TensorBoard logs: {os.path.join(out_dir, 'tb')}")
    print(f"  → tensorboard --logdir {out_dir}/tb")


# ==========================================
# 7. FID / LPIPS (optional, requires torchmetrics)
# ==========================================
@torch.no_grad()
def _log_fid_lpips(model, train_loader, args, device, epoch, writer):
    """Generate samples and compute FID + LPIPS against real data.
    Requires: pip install torchmetrics[image]  (includes torchmetrics + lpips)."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError:
        print("  ⚠ torchmetrics not installed — skipping FID/LPIPS. "
              "Install with: pip install torchmetrics[image]")
        return

    model.eval()
    n = args.fid_n_samples
    batch = args.batch_size

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device)

    # Collect real images
    real_count = 0
    for data_batch in train_loader:
        imgs = data_batch["pixel_values"].to(device)
        imgs_01 = imgs * 0.5 + 0.5  # [-1,1] → [0,1]
        fid.update(imgs_01, real=True)
        real_count += imgs.shape[0]
        if real_count >= n:
            break

    # Generate fake images
    gen_count = 0
    lpips_vals = []
    while gen_count < n:
        cur_batch = min(batch, n - gen_count)
        z = torch.randn(cur_batch, args.latent_dim, device=device)
        fake = model.decode(z) * 0.5 + 0.5
        fid.update(fake, real=False)
        # LPIPS: compare consecutive pairs
        if fake.shape[0] >= 2:
            lpips_vals.append(
                lpips(fake[0::2][:fake.shape[0]//2], fake[1::2][:fake.shape[0]//2]).item()
            )
        gen_count += cur_batch

    fid_score = fid.compute().item()
    avg_lpips = np.mean(lpips_vals) if lpips_vals else 0.0

    writer.add_scalar("metrics/FID", fid_score, epoch)
    writer.add_scalar("metrics/LPIPS_diversity", avg_lpips, epoch)
    if _wandb is not None and _wandb.run is not None:
        _wandb.log({
            "metrics/FID": fid_score,
            "metrics/LPIPS_diversity": avg_lpips,
        }, step=epoch)
    print(f"  → FID: {fid_score:.2f} | LPIPS diversity: {avg_lpips:.4f}")

    del fid, lpips
    model.train()


if __name__ == "__main__":
    main()
