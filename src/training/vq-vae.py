import argparse
import os
from datetime import datetime

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import make_grid, save_image

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
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel_1d = g.unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def ssim_loss(pred, target, window_size=11, sigma=1.5):
    """Compute SSIM loss (1 - SSIM) between prediction and target."""
    channels = pred.shape[1]
    kernel = gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)

    c1 = 0.01**2
    c2 = 0.03**2

    mu_pred = nn.functional.conv2d(
        pred, kernel, padding=window_size // 2, groups=channels
    )
    mu_target = nn.functional.conv2d(
        target, kernel, padding=window_size // 2, groups=channels
    )

    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = (
        nn.functional.conv2d(pred**2, kernel, padding=window_size // 2, groups=channels)
        - mu_pred_sq
    )
    sigma_target_sq = (
        nn.functional.conv2d(
            target**2, kernel, padding=window_size // 2, groups=channels
        )
        - mu_target_sq
    )
    sigma_pred_target = (
        nn.functional.conv2d(
            pred * target, kernel, padding=window_size // 2, groups=channels
        )
        - mu_pred_target
    )

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )

    return 1 - ssim_map.mean()


# ==========================================
# 2. Perceptual Loss
# ==========================================
class VGGPerceptualLoss(nn.Module):
    """Compares VGG19 features between prediction and target."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))
        self.slice2 = nn.Sequential(*list(vgg[4:9]))
        self.slice3 = nn.Sequential(*list(vgg[9:18]))
        self.slice4 = nn.Sequential(*list(vgg[18:27]))
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
# 3. VQ-VAE Architecture
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, hidden_channels=128, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels // 2,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, hidden_channels=128, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels // 2, 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z_q):
        return self.net(z_q)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, use_restart=True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_restart = use_restart

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    def forward(self, z_e):
        # BCHW -> BHWC for vector quantization lookup.
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z_e_perm.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * flat_z @ self.embedding.weight.t()
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = nn.functional.one_hot(
            encoding_indices, num_classes=self.num_embeddings
        ).type(flat_z.dtype)

        # Count usage for Dead Code Revival
        usage = encodings.sum(dim=0)
        num_restarts = 0

        if self.training and self.use_restart:
            # Sync usage across GPUs for DDP consistency
            if dist.is_initialized():
                dist.all_reduce(usage, op=dist.ReduceOp.SUM)
            
            # Identify indices that were never used in this batch
            dead_indices = (usage == 0).nonzero(as_tuple=True)[0]
            num_dead = dead_indices.numel()
            
            if num_dead > 0:
                # Limit restarts to the available vectors in flat_z
                num_to_restart = min(num_dead, flat_z.size(0))
                dead_indices = dead_indices[:num_to_restart]
                
                if dist.is_initialized():
                    # Synchronized random restart: rank 0 picks and broadcasts
                    if dist.get_rank() == 0:
                        indices = torch.randperm(flat_z.size(0), device=flat_z.device)[:num_to_restart]
                        random_latents = flat_z[indices]
                    else:
                        random_latents = torch.empty(num_to_restart, self.embedding_dim, device=flat_z.device)
                    dist.broadcast(random_latents, src=0)
                else:
                    # Local random restart
                    indices = torch.randperm(flat_z.size(0), device=flat_z.device)[:num_to_restart]
                    random_latents = flat_z[indices]
                
                with torch.no_grad():
                    self.embedding.weight.data[dead_indices] = random_latents
                num_restarts = num_to_restart

        z_q = encodings @ self.embedding.weight
        z_q = z_q.view(z_e_perm.shape)

        e_latent_loss = nn.functional.mse_loss(z_q.detach(), z_e_perm)
        q_latent_loss = nn.functional.mse_loss(z_q, z_e_perm.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator.
        z_q = z_e_perm + (z_q - z_e_perm).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return z_q, vq_loss, perplexity, num_restarts


class VQVAE(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
        use_restart=True,
    ):
        super().__init__()
        self.encoder = Encoder(hidden_channels=hidden_channels, embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            use_restart=use_restart,
        )
        self.decoder = Decoder(hidden_channels=hidden_channels, embedding_dim=embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, num_restarts = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, perplexity, num_restarts


# ==========================================
# 4. Lightning Module
# ==========================================
class VQVAELightningModule(L.LightningModule):
    def __init__(
        self,
        lr=2e-4,
        epochs=500,
        sample_dir="samples",
        hidden_channels=128,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
        perceptual_weight=0.5,
        ssim_weight=0.5,
        use_restart=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = VQVAE(
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            use_restart=use_restart,
        )
        self.perceptual_loss_fn = VGGPerceptualLoss()
        self.viz_batch = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch["pixel_values"]

        if self.viz_batch is None:
            self.viz_batch = data[:8].clone()

        reconstructed, vq_loss, perplexity, num_restarts = self.model(data)

        l1 = nn.functional.l1_loss(reconstructed, data, reduction="mean")
        perc = self.perceptual_loss_fn(reconstructed, data)
        ssim = ssim_loss(reconstructed, data)

        recon_loss = l1 + self.hparams.perceptual_weight * perc + self.hparams.ssim_weight * ssim
        loss = recon_loss + vq_loss

        self.log("loss/total", loss, prog_bar=True, sync_dist=True)
        self.log("loss/l1", l1, sync_dist=True)
        self.log("loss/perceptual", perc, sync_dist=True)
        self.log("loss/ssim", ssim, sync_dist=True)
        self.log("loss/vq", vq_loss, sync_dist=True)
        self.log("meta/perplexity", perplexity, sync_dist=True)
        self.log("meta/num_restarts", float(num_restarts), sync_dist=True)

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

        viz = self.viz_batch.to(self.device)
        recon, _, _, _ = self.model(viz)
        comparison = torch.cat([viz, recon], dim=0)
        recon_path = os.path.join(sample_dir, f"recon_epoch_{epoch + 1:03d}.png")
        save_image(comparison, recon_path, nrow=8)

        grid = make_grid(comparison, nrow=8)
        self._log_image_to_loggers("Reconstruction", grid, epoch + 1)

        # Sample random codes for generation snapshots.
        b, _, h, w = viz.shape
        code_h, code_w = h // 4, w // 4
        rand_idx = torch.randint(
            0,
            self.hparams.num_embeddings,
            (b, code_h, code_w),
            device=self.device,
        )
        quantized = self.model.quantizer.embedding(rand_idx)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        generated = self.model.decoder(quantized)

        gen_path = os.path.join(sample_dir, f"gen_epoch_{epoch + 1:03d}.png")
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
# 5. Lightning DataModule
# ==========================================
class HuggingFaceImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name="huggan/AFHQ",
        image_column="image",
        img_size=128,
        batch_size=196,
        num_workers=4,
        grayscale=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None

    def setup(self, stage=None):
        transform_list = [
            transforms.Resize((self.hparams.img_size, self.hparams.img_size)),
        ]
        if not self.hparams.grayscale:
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1
                    ),
                ]
            )
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        image_col = self.hparams.image_column

        def transform_fn(examples):
            pixel_values = []
            for image in examples[image_col]:
                if self.hparams.grayscale:
                    image = image.convert("L").convert("RGB")
                else:
                    image = image.convert("RGB")
                pixel_values.append(transform(image))
            examples["pixel_values"] = pixel_values
            if image_col in examples:
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
# 6. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a VQ-VAE on HuggingFace image datasets (Lightning DDP)"
    )
    parser.add_argument("--batch_size", type=int, default=196)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--dataset", type=str, default="huggan/AFHQ")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--grayscale", action="store_true")

    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)

    parser.add_argument("--perceptual_weight", type=float, default=0.5)
    parser.add_argument("--ssim_weight", type=float, default=0.5)
    parser.add_argument("--no_restart", action="store_false", dest="use_restart")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to Lightning checkpoint (.ckpt) to resume from",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--gpus", type=int, default=-1, help="Number of GPUs (-1 = all available)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "fsdp", "ddp_find_unused_parameters_true"],
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vq-vae-vis")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or f"output_vqvae/run_{timestamp}"
    sample_dir = os.path.join(output_root, "samples")
    checkpoint_dir = os.path.join(output_root, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Output directory: {output_root}")

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

    if torch.cuda.is_available():
        devices = args.gpus if args.gpus > 0 else "auto"
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"

    vqvae_module = VQVAELightningModule(
        lr=args.lr,
        epochs=args.epochs,
        sample_dir=sample_dir,
        hidden_channels=args.hidden_channels,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        use_restart=args.use_restart,
    )
    data_module = HuggingFaceImageDataModule(
        dataset_name=args.dataset,
        image_column=args.image_column,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grayscale=args.grayscale,
    )

    total_params = sum(p.numel() for p in vqvae_module.model.parameters())
    trainable_params = sum(
        p.numel() for p in vqvae_module.model.parameters() if p.requires_grad
    )
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

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

    trainer.fit(vqvae_module, datamodule=data_module, ckpt_path=args.resume)

    if trainer.is_global_zero:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt:
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            raw_sd = {}
            for k, v in ckpt["state_dict"].items():
                new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
                raw_sd[new_key] = v
            save_path = os.path.join(output_root, "best_model.pth")
            torch.save(raw_sd, save_path)
            print(f"Saved raw state dict -> {save_path}")

    print("\nTraining complete!")
    print(f"Models saved in: {output_root}")


if __name__ == "__main__":
    main()
