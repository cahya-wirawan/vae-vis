import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import make_grid, save_image

# ==========================================
# 1. Losses (Perceptual & SSIM)
# ==========================================
class VGGPerceptualLoss(nn.Module):
    PERC_SIZE = 64
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))
        self.slice2 = nn.Sequential(*list(vgg[4:9]))
        self.slice3 = nn.Sequential(*list(vgg[9:18]))
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = nn.functional.interpolate(pred, size=self.PERC_SIZE, mode="bilinear", align_corners=False)
        target = nn.functional.interpolate(target, size=self.PERC_SIZE, mode="bilinear", align_corners=False)
        # Denormalize [-1, 1] to [0, 1] then normalize for VGG
        pred = (pred * 0.5 + 0.5 - self.mean) / self.std
        target = (target * 0.5 + 0.5 - self.mean) / self.std
        loss = 0.0
        x, y = pred, target
        for slicer in [self.slice1, self.slice2, self.slice3]:
            x, y = slicer(x), slicer(y)
            loss += nn.functional.l1_loss(x, y)
        return loss

def ssim_loss(x, y, window_size=11):
    """Simple 1-SSIM loss implementation."""
    C = x.shape[1]
    kernel = torch.ones((C, 1, window_size, window_size), device=x.device) / (window_size**2)
    mu_x = nn.functional.conv2d(x, kernel, padding=window_size//2, groups=C)
    mu_y = nn.functional.conv2d(y, kernel, padding=window_size//2, groups=C)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    sigma_x_sq = nn.functional.conv2d(x*x, kernel, padding=window_size//2, groups=C) - mu_x_sq
    sigma_y_sq = nn.functional.conv2d(y*y, kernel, padding=window_size//2, groups=C) - mu_y_sq
    sigma_xy = nn.functional.conv2d(x*y, kernel, padding=window_size//2, groups=C) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1)*(sigma_x_sq + sigma_y_sq + C2))
    return 1.0 - ssim_map.mean()

# ==========================================
# 2. VAE Components
# ==========================================
class ResBlock(nn.Module):
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
    def forward(self, x): return self.act(x + self.block(x))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), ResBlock(256),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), ResBlock(512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), nn.Flatten()
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), ResBlock(512),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 3. Lightning Module
# ==========================================
class VAELitModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        flat_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(flat_size, self.hparams.latent_dim)
        self.fc_logvar = nn.Linear(flat_size, self.hparams.latent_dim)
        self.decoder_fc = nn.Sequential(nn.Linear(self.hparams.latent_dim, flat_size), nn.ReLU())
        
        self.perceptual_loss = VGGPerceptualLoss()
        self.viz_batch = None

    def decode(self, z):
        return self.decoder(self.decoder_fc(z).view(-1, 512, 4, 4))

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        if self.viz_batch is None: self.viz_batch = x[:8].detach()
        
        recon, mu, logvar = self(x)
        
        # Schedules
        kl_w = min(1.0, self.current_epoch / self.hparams.kl_warmup_epochs) * self.hparams.kl_weight_max
        p_w = min(1.0, self.current_epoch / self.hparams.perc_warmup_epochs) * self.hparams.perc_weight
        
        # Losses
        l1 = nn.functional.l1_loss(recon, x)
        perc = self.perceptual_loss(recon, x)
        ssim = ssim_loss(recon, x) if self.hparams.ssim_weight > 0 else 0.0
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (x.shape[0] * 128 * 128) # norm by pixels
        
        loss = l1 + p_w * perc + self.hparams.ssim_weight * ssim + kl_w * kld
        
        self.log_dict({
            "train/loss": loss, "train/l1": l1, "train/perc": perc, "train/ssim": ssim,
            "train/kld": kld, "train/kl_w": kl_w, "train/p_w": p_w
        }, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 25 == 0 and self.viz_batch is not None:
            self.eval()
            with torch.no_grad():
                recon, _, _ = self(self.viz_batch)
                # Random samples
                z = torch.randn(8, self.hparams.latent_dim, device=self.device)
                gen = self.decode(z)
                
                vis = make_grid(torch.cat([self.viz_batch * 0.5 + 0.5, recon * 0.5 + 0.5, gen * 0.5 + 0.5]), nrow=8)
                save_image(vis, f"samples/epoch_{self.current_epoch+1:03d}.png")
                
                # Log to all loggers
                for logger in self.loggers:
                    if isinstance(logger, TensorBoardLogger):
                        logger.experiment.add_image("visuals", vis, self.current_epoch)
                    elif isinstance(logger, WandbLogger):
                        logger.log_image(key="visuals", images=[vis])
            self.train()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6)
        return [opt], [sched]

# ==========================================
# 4. Main
# ==========================================
def transform_batch(examples, transform):
    examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["image"]]
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--kl-weight-max", type=float, default=0.0001)
    parser.add_argument("--kl-warmup-epochs", type=int, default=30)
    parser.add_argument("--perc-weight", type=float, default=0.5)
    parser.add_argument("--perc-warmup-epochs", type=int, default=50)
    parser.add_argument("--ssim-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    os.makedirs("samples", exist_ok=True)
    
    # Dataset
    dataset = load_dataset("huggan/AFHQ", split="train")
    aug = transforms.Compose([
        transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset.set_transform(lambda e: transform_batch(e, aug))
    
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )

    # Loggers
    loggers = [TensorBoardLogger("runs", name="vae_afhq")]
    if args.wandb:
        loggers.append(WandbLogger(project="vae-afhq"))

    model = VAELitModule(args)
    
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
        logger=loggers,
        callbacks=[
            ModelCheckpoint(save_top_k=3, monitor="train/loss", mode="min", filename="vae-{epoch:02d}-{loss:.4f}"),
            LearningRateMonitor(logging_interval="step")
        ],
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
