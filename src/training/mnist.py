import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
import os
import random
import numpy as np
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

# ==========================================
# 1. Define the VAE Architecture
# ==========================================
class VAE(nn.Module):
    """
    VAE for 28x28 MNIST images.
    Architecture is intentionally kept as Fully Connected layers to match
    the SimpleDecoder used in the Streamlit visualization app.
    """
    def __init__(self, latent_dim=2, hidden_dim1=256, hidden_dim2=128):
        super().__init__()
        
        # ENCODER: Compresses 28x28 image down to hidden layers
        self.enc_fc1 = nn.Linear(28 * 28, hidden_dim1)
        self.enc_fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # Latent Space: Mean and Log-Variance
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        
        # DECODER: Matches the SimpleDecoder architecture
        # Note: hidden_dim1/2 are swapped here to mirror the encoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim2)
        self.dec_fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.dec_fc3 = nn.Linear(hidden_dim1, 28 * 28)

    def encode(self, x):
        h = torch.relu(self.enc_fc1(x))
        h = torch.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        h = torch.relu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

# ==========================================
# 2. Loss Function
# ==========================================
def vae_loss(reconstructed_x, x, mu, logvar, beta=1.0):
    BCE = nn.functional.binary_cross_entropy(reconstructed_x, x.view(-1, 28 * 28), reduction='sum') / x.size(0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = BCE + beta * KLD
    return total_loss, BCE, KLD

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Training')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=2, help='Latent space dimension')
    parser.add_argument('--hidden-dim1', type=int, default=256, help='Hidden layer 1 size')
    parser.add_argument('--hidden-dim2', type=int, default=128, help='Hidden layer 2 size')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='HuggingFace dataset name (default: mnist)')
    parser.add_argument('--image-column', type=str, default='image',
                        help='Column name containing images (default: image)')
    parser.add_argument('--kl-weight-max', type=float, default=1.0, help='Maximum KL weight')
    parser.add_argument('--kl-warmup', type=int, default=5, help='KL warmup epochs')
    parser.add_argument('--output-dir', type=str, default='checkpoints_mnist', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Use AMP')
    parser.add_argument('--clip-grad', type=float, default=1.0, help='Gradient clipping')
    # Logging
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='vae-mnist', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    if args.amp and device.type != 'cuda':
        args.amp = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize W&B
    if args.use_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"run_{timestamp}",
            config=vars(args),
        )
        print("📊 W&B logging enabled")

    # Load dataset from HuggingFace
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_col = args.image_column
    
    def transform_fn(examples):
        # Convert images to tensors, handling both grayscale and RGB
        pixel_values = []
        for img in examples[image_col]:
            # Convert to grayscale if needed for MNIST-like datasets
            if img.mode != 'L':
                img = img.convert('L')
            pixel_values.append(transform(img))
        examples["pixel_values"] = pixel_values
        return examples
    
    print(f"Loading {args.dataset} dataset...")
    dataset = load_dataset(args.dataset, split="train")
    dataset.set_transform(transform_fn)
    print(f"Dataset size: {len(dataset)} images")
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
    )

    model = VAE(latent_dim=args.latent_dim, hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Store first batch for visualization
    viz_batch = None

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        train_bce = 0
        train_kld = 0
        kl_weight = args.kl_weight_max * min(1.0, epoch / args.kl_warmup) if args.kl_warmup > 0 else args.kl_weight_max

        for batch_idx, batch in enumerate(train_loader):
            data = batch["pixel_values"]
            # Stack list of tensors into batch tensor
            if isinstance(data, list):
                data = torch.stack(data)
            # Store first batch for visualization
            if viz_batch is None:
                viz_batch = data[:8].clone()
            data = data.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                recon, mu, logvar = model(data)
                loss, bce, kld = vae_loss(recon, data, mu, logvar, beta=kl_weight)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_bce += bce.item()
            train_kld += kld.item()
            
        num_batches = len(train_loader)
        avg_loss = train_loss / num_batches
        avg_bce = train_bce / num_batches
        avg_kld = train_kld / num_batches
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}) KL weight: {kl_weight:.3f}")

        # Log to W&B
        if args.use_wandb and wandb is not None:
            log_dict = {
                "loss/total": avg_loss,
                "loss/bce": avg_bce,
                "loss/kld": avg_kld,
                "meta/kl_weight": kl_weight,
                "epoch": epoch + 1,
            }
            # Log sample reconstructions every 5 epochs
            if (epoch + 1) % 5 == 0 and viz_batch is not None:
                model.eval()
                with torch.no_grad():
                    viz = viz_batch.to(device)
                    recon_viz, _, _ = model(viz)
                    recon_viz = recon_viz.view(-1, 1, 28, 28)
                    comparison = torch.cat([viz, recon_viz], dim=0)
                    grid = make_grid(comparison, nrow=8)
                    # Save locally
                    save_image(comparison, os.path.join(sample_dir, f"recon_epoch_{epoch+1:03d}.png"), nrow=8)
                    # Log to W&B
                    log_dict["Reconstruction"] = wandb.Image(grid)
                model.train()
            wandb.log(log_dict, step=epoch + 1)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }
        torch.save(checkpoint, os.path.join(run_dir, f"vae_mnist_checkpoint_epoch_{epoch+1}.pth"))

    # Save final decoder weights for Streamlit with metadata
    decoder_payload = {
        'metadata': {
            'latent_dim': args.latent_dim,
            'hidden_dim1': args.hidden_dim2,  # Swapped to match SimpleDecoder order (latent -> dim1 -> dim2)
            'hidden_dim2': args.hidden_dim1,
            'architecture': 'fc_mnist'
        },
        'state_dict': {
            'fc1.weight': model.dec_fc1.weight.detach().cpu(),
            'fc1.bias': model.dec_fc1.bias.detach().cpu(),
            'fc2.weight': model.dec_fc2.weight.detach().cpu(),
            'fc2.bias': model.dec_fc2.bias.detach().cpu(),
            'fc3.weight': model.dec_fc3.weight.detach().cpu(),
            'fc3.bias': model.dec_fc3.bias.detach().cpu(),
        }
    }
    
    torch.save(decoder_payload, os.path.join(run_dir, 'decoder_weights.pth'))
    torch.save(decoder_payload, 'decoder_weights.pth')
    
    # Finish W&B run
    if args.use_wandb and wandb is not None:
        wandb.finish()
    
    print(f"\n✅ Training complete!")
    print(f"📦 Checkpoints saved to: {run_dir}")
    print(f"🎯 Final decoder weights with metadata saved to: decoder_weights.pth")

if __name__ == "__main__":
    main()
