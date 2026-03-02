#!/usr/bin/env python3
"""Export VAE decoder weights with BatchNorm folded into Conv2d layers.

Usage:
    python export_dog_weights.py checkpoint.pth -o weights.bin
    python export_dog_weights.py checkpoint.ckpt --latent-dim 256 -o weights.bin
"""

import argparse
import os

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: PyTorch not found. Run this in an environment with PyTorch.")
    exit(1)


# ---- Minimal model definition (avoids importing dog.py + all its deps) ----

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

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), ResBlock(256),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), ResBlock(512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), ResBlock(512),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        flat_size = 512 * 4 * 4
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, flat_size), nn.ReLU())

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(self.decoder_fc(z).view(-1, 512, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def fold_conv_bn(conv_weight, conv_bias, bn_rm, bn_rv, bn_w, bn_b, eps=1e-5):
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.size(0))
    factor = bn_w / torch.sqrt(bn_rv + eps)
    folded_weight = conv_weight * factor.view(-1, 1, 1, 1)
    folded_bias = (conv_bias - bn_rm) * factor + bn_b
    return folded_weight, folded_bias


def write_tensor(f, tensor):
    """Write tensor as raw little-endian float32 bytes (fast via numpy)."""
    f.write(tensor.detach().contiguous().cpu().float().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser(description="Export VAE decoder weights")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to .pth or .ckpt checkpoint")
    parser.add_argument("-o", "--output", default=None,
                        help="Output binary path (default: weights.bin next to checkpoint)")
    parser.add_argument("--latent-dim", type=int, default=256,
                        help="Latent dimension (default: 256)")
    args = parser.parse_args()

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # Try common locations
        for candidate in [
            "src/training/dog_vae_256_best.pth",
            "src/training/dog_vae_128_best.pth",
            "src/training/dog_vae_128.pth",
        ]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"Error: Could not find checkpoint at {ckpt_path}")
        return

    out_path = args.output or "weights.bin"

    print(f"Loading checkpoint {ckpt_path} (latent_dim={args.latent_dim})...")

    # Load model with proper architecture
    model = VAE(latent_dim=args.latent_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Handle raw state_dict, Lightning checkpoint, or model_state_dict wrapper
    if "state_dict" in ckpt:
        sd = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(sd)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    state_dict = model.state_dict()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "wb") as f:
        # 1. FC layer (no BN)
        write_tensor(f, state_dict["decoder_fc.0.weight"])
        write_tensor(f, state_dict["decoder_fc.0.bias"])

        # Helper: fold Conv+BN and write
        def write_folded(conv_prefix, bn_prefix):
            cw = state_dict[f"{conv_prefix}.weight"]
            cb = state_dict.get(f"{conv_prefix}.bias", None)
            bm = state_dict[f"{bn_prefix}.running_mean"]
            bv = state_dict[f"{bn_prefix}.running_var"]
            bw = state_dict[f"{bn_prefix}.weight"]
            bb = state_dict[f"{bn_prefix}.bias"]
            fw, fb = fold_conv_bn(cw, cb, bm, bv, bw, bb)
            write_tensor(f, fw)
            write_tensor(f, fb)

        def write_resblock(prefix):
            write_folded(f"{prefix}.block.0", f"{prefix}.block.1")
            write_folded(f"{prefix}.block.3", f"{prefix}.block.4")

        # 2. Stages 1-4: Conv+BN+ReLU+ResBlock
        #    Sequential indices: conv=1+5k, bn=2+5k, resblock=4+5k
        for ci, bi, ri in [(1, 2, 4), (6, 7, 9), (11, 12, 14), (16, 17, 19)]:
            write_folded(f"decoder.net.{ci}", f"decoder.net.{bi}")
            write_resblock(f"decoder.net.{ri}")

        # 3. Stage 5: Conv(64→32)+BN → Conv(32→3), no BN on final
        write_folded("decoder.net.21", "decoder.net.22")
        write_tensor(f, state_dict["decoder.net.24.weight"])
        write_tensor(f, state_dict.get("decoder.net.24.bias", torch.zeros(3)))

    total_bytes = os.path.getsize(out_path)
    print(f"Exported to {out_path} ({total_bytes:,} bytes, {total_bytes / 1024 / 1024:.1f} MB)")

    # Quick verification
    with torch.no_grad():
        z = torch.randn(1, args.latent_dim)
        out = model.decode(z)
        print(f"Verification: output range=[{out.min():.4f}, {out.max():.4f}], mean={out.mean():.4f}")


if __name__ == "__main__":
    main()
