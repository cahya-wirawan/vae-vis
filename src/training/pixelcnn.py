import argparse
import os
import sys
import importlib.util
from datetime import datetime

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

try:
    import wandb
except ImportError:
    wandb = None

# ==========================================
# 1. Model Components (Gated PixelCNN)
# ==========================================

class GatedActivation(nn.Module):
    def forward(self, x):
        tanh, sig = x.chunk(2, dim=1)
        return torch.tanh(tanh) * torch.sigmoid(sig)

class GatedMaskedConv2d(nn.Module):
    """
    Gated Masked Convolution used in PixelCNN++ to avoid blind spots.
    Uses separate vertical and horizontal stacks to maintain causality.
    """
    def __init__(self, in_dim, out_dim, kernel_size, mask_type, residual=True):
        super().__init__()
        self.mask_type = mask_type
        self.residual = residual
        
        # Vertical stack: sees pixels above the current pixel
        self.v_conv = nn.Conv2d(in_dim, 2 * out_dim, (kernel_size // 2 + 1, kernel_size), 
                                padding=(kernel_size // 2, kernel_size // 2))
        
        # Horizontal stack: sees pixels to the left of the current pixel
        self.h_conv = nn.Conv2d(in_dim, 2 * out_dim, (1, kernel_size // 2 + 1), 
                                padding=(0, kernel_size // 2))
        
        # Connections between stacks
        self.v_to_h = nn.Conv2d(2 * out_dim, 2 * out_dim, 1)
        self.h_out = nn.Conv2d(out_dim, out_dim, 1)
        self.gate = GatedActivation()

    def forward(self, x_v, x_h):
        # Shift x_v down by 1 to ensure causality
        x_v_shifted = F.pad(x_v, (0, 0, 1, 0))[:, :, :-1, :]
        v_out = self.v_conv(x_v_shifted)
        v_out = v_out[:, :, :x_v.size(-2), :] # Ensure same height
        
        h_input = x_h
        if self.mask_type == 'A':
            # Strictly causal: cannot see current pixel
            h_input = F.pad(h_input, (1, 0, 0, 0))[:, :, :, :-1]
            
        h_out = self.h_conv(h_input)
        h_out = h_out[:, :, :, :x_h.size(-1)] # Ensure same width
        
        v_to_h = self.v_to_h(v_out)
        h_out = h_out + v_to_h
        
        # Gated Activation units
        v_tanh, v_sig = v_out.chunk(2, 1)
        v_out = torch.tanh(v_tanh) * torch.sigmoid(v_sig)
        
        h_tanh, h_sig = h_out.chunk(2, 1)
        h_out = torch.tanh(h_tanh) * torch.sigmoid(h_sig)
        h_out = self.h_out(h_out)
        
        if self.residual and self.mask_type == 'B':
            h_out = h_out + x_h
            
        return v_out, h_out

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, num_layers=12, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)
        
        self.layers = nn.ModuleList()
        # First layer: mask type A (strictly causal)
        self.layers.append(GatedMaskedConv2d(hidden_dim, hidden_dim, 7, 'A', residual=False))
        
        # Subsequent layers: mask type B (can see current row up to current pixel)
        for _ in range(num_layers - 1):
            self.layers.append(GatedMaskedConv2d(hidden_dim, hidden_dim, 3, 'B'))
            
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_embeddings, 1)
        )

    def forward(self, x):
        # x: (B, H, W) discrete indices
        x = self.embedding(x).permute(0, 3, 1, 2) # (B, C, H, W)
        
        v, h = x, x
        for layer in self.layers:
            v, h = layer(v, h)
            
        return self.out_conv(h)

# ==========================================
# 2. Lightning Module
# ==========================================

class PixelCNNTrainer(L.LightningModule):
    def __init__(self, vqvae_module, lr=1e-3, num_layers=12, hidden_dim=128, sample_dir="samples_pixelcnn"):
        super().__init__()
        self.save_hyperparameters(ignore=['vqvae_module'])
        self.vqvae = vqvae_module
        self.vqvae.eval()
        self.vqvae.freeze()
        
        num_embeddings = self.vqvae.hparams.num_embeddings
        self.model = PixelCNN(num_embeddings, num_layers=num_layers, hidden_dim=hidden_dim)

    def _get_indices(self, x):
        """Encodes images into discrete indices using the fixed VQ-VAE encoder."""
        with torch.no_grad():
            z_e = self.vqvae.model.encoder(x)
            quantizer = self.vqvae.model.quantizer
            
            # Replicate quantization logic to get indices without side effects
            z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
            flat_z = z_e_perm.view(-1, quantizer.embedding_dim)
            distances = (
                flat_z.pow(2).sum(dim=1, keepdim=True)
                + quantizer.embedding.weight.pow(2).sum(dim=1)
                - 2 * flat_z @ quantizer.embedding.weight.t()
            )
            indices = torch.argmin(distances, dim=1)
            return indices.view(z_e.size(0), z_e.size(2), z_e.size(3))

    def training_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        indices = self._get_indices(x)
        
        logits = self.model(indices)
        loss = F.cross_entropy(logits, indices)
        
        self.log("loss/train", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def on_train_epoch_end(self):
        # Periodically sample from the model only on the main process
        if self.trainer.is_global_zero and (self.current_epoch + 1) % 10 == 0:
            self._sample_and_log()

    def _log_image_to_loggers(self, tag, image_tensor, step):
        loggers = []
        if self.trainer is not None:
            loggers = self.trainer.loggers or []

        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(tag, image_tensor, step)
            elif isinstance(logger, WandbLogger) and wandb is not None:
                logger.experiment.log({tag: [wandb.Image(image_tensor)]}, step=step)

    def _sample_and_log(self):
        self.model.eval()
        device = self.device
        # Latent dimensions (assuming 128x128 input -> 32x32 latent grid)
        h, w = 32, 32 
        num_samples = 8
        samples = torch.zeros(num_samples, h, w, dtype=torch.long, device=device)
        
        # Autoregressive pixel-by-pixel sampling
        for i in range(h):
            for j in range(w):
                logits = self.model(samples)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
        
        # Decode indices back to image space using VQ-VAE decoder
        quantized = self.vqvae.model.quantizer.embedding(samples)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        reconstructions = self.vqvae.model.decoder(quantized)
        
        os.makedirs(self.hparams.sample_dir, exist_ok=True)
        save_path = os.path.join(self.hparams.sample_dir, f"epoch_{self.current_epoch+1:03d}.png")
        save_image(reconstructions, save_path, nrow=4)
        
        grid = make_grid(reconstructions, nrow=4)
        self._log_image_to_loggers("Prior Samples", grid, self.current_epoch + 1)
        
        self.model.train()

# ==========================================
# 3. Main Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train PixelCNN prior for VQ-VAE")
    parser.add_argument("--vqvae_ckpt", type=str, required=True, help="Path to VQ-VAE checkpoint (.ckpt)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="output_pixelcnn")
    
    # Wandb args
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vq-vae-prior")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Dataset args (should match VQ-VAE training parameters)
    parser.add_argument("--dataset", type=str, default="huggan/AFHQ")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    
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

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    # Load VQ-VAE module dynamically from vq-vae.py
    vq_vae_path = os.path.join(os.path.dirname(__file__), "vq-vae.py")
    spec = importlib.util.spec_from_file_location("vq_vae", vq_vae_path)
    vq_vae = importlib.util.module_from_spec(spec)
    sys.modules["vq_vae"] = vq_vae
    spec.loader.exec_module(vq_vae)
    
    VQVAELightningModule = vq_vae.VQVAELightningModule
    HuggingFaceImageDataModule = vq_vae.HuggingFaceImageDataModule

    # Load the pre-trained VQ-VAE model
    print(f"Loading VQ-VAE from {args.vqvae_ckpt}")
    vqvae_module = VQVAELightningModule.load_from_checkpoint(args.vqvae_ckpt)
    
    # Devices
    if torch.cuda.is_available():
        devices = args.gpus if args.gpus > 0 else "auto"
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"

    # Initialize Loggers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loggers = [TensorBoardLogger(save_dir=args.output_dir, name="logs")]
    if args.use_wandb:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name or f"pixelcnn_{timestamp}",
                save_dir=args.output_dir,
                config=vars(args),
            )
        )

    # Initialize PixelCNN module
    sample_dir = os.path.join(args.output_dir, "samples")
    pixelcnn_module = PixelCNNTrainer(
        vqvae_module, 
        lr=args.lr, 
        num_layers=args.num_layers, 
        hidden_dim=args.hidden_dim,
        sample_dir=sample_dir
    )
    
    # Prepare Data
    datamodule = HuggingFaceImageDataModule(
        dataset_name=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Trainer configuration
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=args.strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=args.output_dir,
        logger=loggers,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=os.path.join(args.output_dir, "checkpoints"),
                filename="pixelcnn-{epoch:02d}",
                save_top_k=3,
                monitor="loss/train",
                mode="min"
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
        ]
    )
    
    trainer.fit(pixelcnn_module, datamodule=datamodule)

if __name__ == "__main__":
    main()
