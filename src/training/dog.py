import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import os
import argparse
from datetime import datetime
import random
import numpy as np

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

try:
    import wandb
except ImportError:  # Optional dependency unless --use_wandb is enabled.
    wandb = None


# ==========================================
# 1. SSIM Loss
# ==========================================
def gaussian_kernel(window_size: int, sigma: float, channels: int):
    """Create a 2D Gaussian kernel for SSIM."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_1d = g.unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim_loss(pred, target, window_size=11, sigma=1.5):
    """Compute SSIM loss (1 - SSIM) between prediction and target.
    Returns a value in [0, 1] where 0 means identical images."""
    channels = pred.shape[1]
    kernel = gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = nn.functional.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_target = nn.functional.conv2d(target, kernel, padding=window_size // 2, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = nn.functional.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = nn.functional.conv2d(target ** 2, kernel, padding=window_size // 2, groups=channels) - mu_target_sq
    sigma_pred_target = nn.functional.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1 - ssim_map.mean()


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
def vae_loss(reconstructed_x, x, mu, logvar, perceptual_fn, kl_weight=1.0, ssim_weight=0.1):
    # L1 loss: sharper than MSE
    L1 = nn.functional.l1_loss(reconstructed_x, x, reduction="mean")
    # Perceptual loss
    P_LOSS = perceptual_fn(reconstructed_x, x)
    # SSIM loss: structural similarity
    SSIM = ssim_loss(reconstructed_x, x)
    # KL Divergence
    logvar_safe = torch.clamp(logvar, min=-10.0, max=10.0)
    KLD = -0.5 * torch.mean(
        torch.sum(1 + logvar_safe - mu.pow(2) - logvar_safe.exp(), dim=1)
    )
    recon_loss = L1 + 0.5 * P_LOSS + ssim_weight * SSIM
    total = recon_loss + kl_weight * KLD
    return total, L1, P_LOSS, SSIM, KLD


# ==========================================
# 5. Lightning Module
# ==========================================
class VAELightningModule(L.LightningModule):
    def __init__(self, latent_dim=256, lr=2e-4, kl_weight_max=0.001,
                 kl_warmup_epochs=30, ssim_weight=0.5, epochs=500, sample_dir="samples"):
        super().__init__()
        self.save_hyperparameters()
        self.model = VAE(latent_dim)
        self.perceptual_loss_fn = VGGPerceptualLoss()
        self.viz_batch = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch["pixel_values"]

        # Store first batch for visualization
        if self.viz_batch is None:
            self.viz_batch = data[:8].clone()

        reconstructed, mu, logvar = self.model(data)

        kl_weight = (
            min(1.0, self.current_epoch / max(1, self.hparams.kl_warmup_epochs))
            * self.hparams.kl_weight_max
        )
        loss, l1, perc, ssim, kld = vae_loss(
            reconstructed, data, mu, logvar, self.perceptual_loss_fn, kl_weight,
            ssim_weight=self.hparams.ssim_weight
        )

        self.log("loss/total", loss, prog_bar=True, sync_dist=True)
        self.log("loss/l1", l1, sync_dist=True)
        self.log("loss/perceptual", perc, sync_dist=True)
        self.log("loss/ssim", ssim, sync_dist=True)
        self.log("loss/kld", kld, sync_dist=True)
        self.log("meta/kl_weight", kl_weight, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 25 == 0 and self.viz_batch is not None:
            self._save_samples()

    def _log_image_to_loggers(self, tag, image_tensor, step):
        loggers = []
        if self.trainer is not None:
            loggers = self.trainer.loggers or []

        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(tag, image_tensor, step)
            elif isinstance(logger, WandbLogger) and wandb is not None:
                logger.experiment.log({tag: [wandb.Image(image_tensor)]}, step=step)

    @torch.no_grad()
    def _save_samples(self):
        self.model.eval()
        sample_dir = self.hparams.sample_dir
        os.makedirs(sample_dir, exist_ok=True)
        epoch = self.current_epoch

        # Reconstruct training images
        viz = self.viz_batch.to(self.device)
        recon, _, _ = self.model(viz)
        comparison = torch.cat([viz, recon], dim=0)
        recon_path = os.path.join(sample_dir, f"recon_epoch_{epoch+1:03d}.png")
        save_image(comparison, recon_path, nrow=8)
        grid = make_grid(comparison, nrow=8)
        self._log_image_to_loggers("Reconstruction", grid, epoch + 1)

        # Generate from random latent vectors
        z = torch.randn(8, self.hparams.latent_dim, device=self.device)
        generated = self.model.decode(z)
        gen_path = os.path.join(sample_dir, f"gen_epoch_{epoch+1:03d}.png")
        save_image(generated, gen_path, nrow=8)
        grid_gen = make_grid(generated, nrow=8)
        self._log_image_to_loggers("Generation", grid_gen, epoch + 1)

        self.model.train()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}


# ==========================================
# 6. Lightning DataModule
# ==========================================
class HuggingFaceImageDataModule(L.LightningDataModule):
    def __init__(self, dataset_name="huggan/AFHQ", image_column="image", img_size=128, batch_size=196, num_workers=4, grayscale=False):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None

    def setup(self, stage=None):
        # Build transforms - skip flip and color jitter for grayscale datasets
        transform_list = [
            transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
        ]
        if not self.hparams.grayscale:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        image_col = self.hparams.image_column

        def transform_fn(examples):
            examples["pixel_values"] = [
                transform(image.convert("RGB")) for image in examples[image_col]
            ]
            del examples[image_col]
            return examples

        print(f"Loading {self.hparams.dataset_name} dataset...")
        self.dataset = load_dataset(self.hparams.dataset_name, split="train")
        self.dataset.set_transform(transform_fn)
        print(f"Dataset size: {len(self.dataset)} images")

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )


# ==========================================
# 7. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a VAE on AFHQ dataset (Lightning DDP)"
    )
    parser.add_argument("--batch_size", type=int, default=196)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="huggan/AFHQ",
                        help="HuggingFace dataset name (default: huggan/AFHQ)")
    parser.add_argument("--image_column", type=str, default="image",
                        help="Column name containing images in the dataset (default: image)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Disable augmentations unsuitable for grayscale datasets (e.g., MNIST)")
    parser.add_argument("--kl_weight_max", type=float, default=0.001)
    parser.add_argument("--ssim_weight", type=float, default=0.5,
                        help="Weight for SSIM loss component (default: 0.5)")
    parser.add_argument("--kl_warmup_epochs", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to Lightning checkpoint (.ckpt) to resume from",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Multi-GPU / Lightning
    parser.add_argument(
        "--gpus", type=int, default=-1,
        help="Number of GPUs (-1 = all available)",
    )
    parser.add_argument(
        "--strategy", type=str, default="auto",
        choices=["auto", "ddp", "fsdp", "ddp_find_unused_parameters_true"],
    )
    parser.add_argument(
        "--precision", type=str, default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vae-vis-dog")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or f"output/run_{timestamp}"
    sample_dir = os.path.join(output_root, "samples")
    checkpoint_dir = os.path.join(output_root, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Output directory: {output_root}")

    # Loggers
    loggers = [TensorBoardLogger(save_dir=output_root, name="logs")]
    if args.use_wandb:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name or f"run_{timestamp}",
                save_dir=output_root,
                config=vars(args),
            )
        )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch_{epoch:03d}",
            save_top_k=3,
            monitor="loss/total",
            mode="min",
            save_last=True,
            every_n_epochs=50,
        ),
        ModelCheckpoint(
            dirpath=output_root,
            filename="best_model",
            save_top_k=1,
            monitor="loss/total",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Devices
    if torch.cuda.is_available():
        devices = args.gpus if args.gpus > 0 else "auto"
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"

    # Module + DataModule
    vae_module = VAELightningModule(
        latent_dim=args.latent_dim,
        lr=args.lr,
        kl_weight_max=args.kl_weight_max,
        kl_warmup_epochs=args.kl_warmup_epochs,
        ssim_weight=args.ssim_weight,
        epochs=args.epochs,
        sample_dir=sample_dir,
    )
    data_module = HuggingFaceImageDataModule(
        dataset_name=args.dataset,
        image_column=args.image_column,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grayscale=args.grayscale,
    )

    total_params = sum(p.numel() for p in vae_module.model.parameters())
    trainable_params = sum(
        p.numel() for p in vae_module.model.parameters() if p.requires_grad
    )
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=args.strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic=False,
    )

    trainer.fit(vae_module, datamodule=data_module, ckpt_path=args.resume)

    # Save raw state dict for inference / export
    if trainer.is_global_zero:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt:
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            raw_sd = {}
            for k, v in ckpt["state_dict"].items():
                # Strip "model." prefix added by LightningModule
                new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
                raw_sd[new_key] = v
            save_path = os.path.join(output_root, "best_model.pth")
            torch.save(raw_sd, save_path)
            print(f"Saved raw state dict → {save_path}")

    print(f"\n✅ Training complete!")
    print(f"Models saved in: {output_root}")


if __name__ == "__main__":
    main()
