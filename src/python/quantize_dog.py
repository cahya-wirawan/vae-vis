#!/usr/bin/env python3
"""Quantize a trained VAE model and export weights for inference.

Supports three quantization modes:
  fp32  — original (no quantization), ~47 MB
  fp16  — float16, ~23 MB, negligible quality loss
  int8  — per-channel int8 with scale/zero-point, ~12 MB

The exported binary includes a 16-byte header so the loader knows the format.

Usage:
    python quantize_dog.py checkpoint.pth --mode fp16 -o weights_fp16.bin
    python quantize_dog.py checkpoint.pth --mode int8 -o weights_int8.bin
    python quantize_dog.py checkpoint.pth --mode fp32 -o weights_fp32.bin  # same as export
    python quantize_dog.py checkpoint.pth --mode fp16 --save-pth quantized.pth
"""

import argparse
import os
import struct
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: PyTorch not found.")
    exit(1)


# ── Minimal model definition (no heavy deps) ────────────────────────────────

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def fold_conv_bn(conv_weight, conv_bias, bn_rm, bn_rv, bn_w, bn_b, eps=1e-5):
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.size(0))
    factor = bn_w / torch.sqrt(bn_rv + eps)
    folded_weight = conv_weight * factor.view(-1, 1, 1, 1)
    folded_bias = (conv_bias - bn_rm) * factor + bn_b
    return folded_weight, folded_bias


def load_model(ckpt_path, latent_dim):
    model = VAE(latent_dim=latent_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        sd = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(sd)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def collect_decoder_tensors(state_dict):
    """Collect BN-folded decoder tensors in the canonical order."""
    tensors = []

    # 1. FC layer
    tensors.append(("fc.w", state_dict["decoder_fc.0.weight"]))
    tensors.append(("fc.b", state_dict["decoder_fc.0.bias"]))

    def get_folded(conv_prefix, bn_prefix):
        return fold_conv_bn(
            state_dict[f"{conv_prefix}.weight"],
            state_dict.get(f"{conv_prefix}.bias"),
            state_dict[f"{bn_prefix}.running_mean"],
            state_dict[f"{bn_prefix}.running_var"],
            state_dict[f"{bn_prefix}.weight"],
            state_dict[f"{bn_prefix}.bias"],
        )

    def add_resblock(prefix):
        w1, b1 = get_folded(f"{prefix}.block.0", f"{prefix}.block.1")
        tensors.append(("rb.conv1.w", w1))
        tensors.append(("rb.conv1.b", b1))
        w2, b2 = get_folded(f"{prefix}.block.3", f"{prefix}.block.4")
        tensors.append(("rb.conv2.w", w2))
        tensors.append(("rb.conv2.b", b2))

    # 2. Stages 1-4
    for ci, bi, ri in [(1, 2, 4), (6, 7, 9), (11, 12, 14), (16, 17, 19)]:
        w, b = get_folded(f"decoder.net.{ci}", f"decoder.net.{bi}")
        tensors.append(("conv.w", w))
        tensors.append(("conv.b", b))
        add_resblock(f"decoder.net.{ri}")

    # 3. Stage 5
    w5, b5 = get_folded("decoder.net.21", "decoder.net.22")
    tensors.append(("s5.conv1.w", w5))
    tensors.append(("s5.conv1.b", b5))
    tensors.append(("s5.conv2.w", state_dict["decoder.net.24.weight"]))
    tensors.append(("s5.conv2.b", state_dict.get("decoder.net.24.bias", torch.zeros(3))))

    return tensors


# ── Binary Format ────────────────────────────────────────────────────────────
#
# Header (16 bytes):
#   magic:   u32 = 0x56414551  ('VAEQ')
#   version: u32 = 1
#   mode:    u32 = 0 (fp32) | 1 (fp16) | 2 (int8)
#   count:   u32 = number of tensors
#
# Then for each tensor:
#   fp32: raw float32 bytes
#   fp16: raw float16 bytes
#   int8: [num_elements u32] [out_channels u32] [scale f32 × out_ch] [zero_point i8 × out_ch (padded to 4)] [int8 data (padded to 4)]

MAGIC = 0x56414551
VERSION = 1
MODE_FP32 = 0
MODE_FP16 = 1
MODE_INT8 = 2


def write_header(f, mode, count):
    f.write(struct.pack("<IIII", MAGIC, VERSION, mode, count))


def export_fp32(f, tensors):
    for name, t in tensors:
        f.write(t.detach().contiguous().cpu().float().numpy().tobytes())


def export_fp16(f, tensors):
    for name, t in tensors:
        arr = t.detach().contiguous().cpu().half().numpy()
        f.write(arr.tobytes())


def quantize_tensor_int8(tensor):
    """Per-output-channel symmetric int8 quantization.

    For a weight tensor with shape [out_ch, ...], compute a scale per out_ch
    so that the full range maps to [-127, 127]. Zero-point is always 0
    (symmetric).

    Returns: (int8_data, scales) where scales is [out_ch] float32.
    """
    t = tensor.detach().contiguous().cpu().float()
    if t.dim() >= 2:
        # Per output-channel: flatten all dims except 0
        flat = t.view(t.shape[0], -1)
        max_abs = flat.abs().amax(dim=1).clamp(min=1e-8)  # [out_ch]
    else:
        # 1-D (bias): single scale
        max_abs = t.abs().clamp(min=1e-8).unsqueeze(0) if t.dim() == 0 else t.abs().amax().clamp(min=1e-8).unsqueeze(0)
        t = t.unsqueeze(0) if t.dim() == 0 else t.unsqueeze(0)

    scales = max_abs / 127.0  # [out_ch] or [1]
    # Quantize
    if t.dim() >= 2:
        quantized = torch.clamp(torch.round(t / scales.view(-1, *([1] * (t.dim() - 1)))), -127, 127).to(torch.int8)
    else:
        quantized = torch.clamp(torch.round(t / scales.view(-1, 1)), -127, 127).to(torch.int8)

    return quantized.view(tensor.shape), scales.squeeze()


def export_int8(f, tensors):
    for name, t in tensors:
        quantized, scales = quantize_tensor_int8(t)
        q_np = quantized.numpy().flatten()
        s_np = scales.numpy().flatten() if scales.dim() > 0 else np.array([scales.item()], dtype=np.float32)

        num_elements = q_np.size
        out_channels = s_np.size

        # Write: num_elements(u32), out_channels(u32), scales(f32×out_ch), int8_data
        f.write(struct.pack("<II", num_elements, out_channels))
        f.write(s_np.astype(np.float32).tobytes())
        f.write(q_np.astype(np.int8).tobytes())
        # Pad to 4-byte alignment
        pad = (4 - (q_np.size % 4)) % 4
        if pad:
            f.write(b'\x00' * pad)


# ── Quality comparison ───────────────────────────────────────────────────────

def compute_mse(tensors_a, tensors_b):
    """MSE between two lists of tensors (same order)."""
    total_mse = 0
    total_count = 0
    for (_, a), (_, b) in zip(tensors_a, tensors_b):
        diff = (a.float() - b.float()).pow(2).sum().item()
        total_mse += diff
        total_count += a.numel()
    return total_mse / total_count if total_count > 0 else 0


def dequantize_int8(tensors):
    """Dequantize int8 tensors back to float for quality comparison."""
    result = []
    for name, t in tensors:
        quantized, scales = quantize_tensor_int8(t)
        if t.dim() >= 2 and scales.dim() > 0:
            deq = quantized.float() * scales.view(-1, *([1] * (t.dim() - 1)))
        else:
            s = scales.item() if scales.dim() == 0 else scales[0].item()
            deq = quantized.float() * s
        result.append((name, deq))
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize VAE decoder weights")
    parser.add_argument("checkpoint", help="Path to .pth or .ckpt")
    parser.add_argument("--mode", choices=["fp32", "fp16", "int8"], default="fp16",
                        help="Quantization mode (default: fp16)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output binary path (default: weights_{mode}.bin)")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--save-pth", default=None,
                        help="Also save a quantized PyTorch state_dict .pth file")
    parser.add_argument("--compare", action="store_true",
                        help="Run quality comparison between fp32 and quantized")
    args = parser.parse_args()

    out_path = args.output or f"weights_{args.mode}.bin"

    print(f"Loading {args.checkpoint} (latent_dim={args.latent_dim})...")
    model = load_model(args.checkpoint, args.latent_dim)
    state_dict = model.state_dict()
    tensors = collect_decoder_tensors(state_dict)

    total_params = sum(t.numel() for _, t in tensors)
    print(f"Decoder: {len(tensors)} tensors, {total_params:,} parameters")

    # Export binary
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mode_code = {"fp32": MODE_FP32, "fp16": MODE_FP16, "int8": MODE_INT8}[args.mode]

    with open(out_path, "wb") as f:
        write_header(f, mode_code, len(tensors))
        if args.mode == "fp32":
            export_fp32(f, tensors)
        elif args.mode == "fp16":
            export_fp16(f, tensors)
        elif args.mode == "int8":
            export_int8(f, tensors)

    file_size = os.path.getsize(out_path)
    fp32_size = total_params * 4
    ratio = file_size / fp32_size * 100
    print(f"\nExported to {out_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  Compression: {ratio:.1f}% of fp32 ({fp32_size / 1024 / 1024:.1f} MB)")

    # Quality comparison
    if args.compare or args.mode != "fp32":
        print(f"\nQuality comparison ({args.mode} vs fp32):")
        if args.mode == "fp16":
            fp16_tensors = [(n, t.half().float()) for n, t in tensors]
            mse = compute_mse(tensors, fp16_tensors)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            print(f"  Weight MSE: {mse:.2e}")
            print(f"  Weight PSNR: {psnr:.1f} dB")
        elif args.mode == "int8":
            deq_tensors = dequantize_int8(tensors)
            mse = compute_mse(tensors, deq_tensors)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            print(f"  Weight MSE: {mse:.2e}")
            print(f"  Weight PSNR: {psnr:.1f} dB")

        # Run actual inference comparison
        print("\nInference comparison (10 random latents):")
        with torch.no_grad():
            z_test = torch.randn(10, args.latent_dim)
            ref_out = model.decode(z_test)

            if args.mode == "fp16":
                model_q = VAE(latent_dim=args.latent_dim)
                sd_q = {k: v.half().float() for k, v in state_dict.items()}
                model_q.load_state_dict(sd_q)
                model_q.eval()
                q_out = model_q.decode(z_test)
            elif args.mode == "int8":
                # For int8, dequantize decoder weights back and run inference
                # (This is an approximation — real int8 inference would use
                #  quantized ops, but for quality comparison this is equivalent)
                model_q = VAE(latent_dim=args.latent_dim)
                model_q.load_state_dict(state_dict)
                model_q.eval()
                # Quantize and dequantize all decoder conv/linear weights
                for name, param in model_q.named_parameters():
                    if "decoder" in name and "weight" in name:
                        q, s = quantize_tensor_int8(param.data)
                        if param.dim() >= 2 and s.dim() > 0:
                            param.data = q.float() * s.view(-1, *([1] * (param.dim() - 1)))
                        else:
                            sv = s.item() if s.dim() == 0 else s[0].item()
                            param.data = q.float() * sv
                q_out = model_q.decode(z_test)
            else:
                q_out = ref_out

            pixel_mse = (ref_out - q_out).pow(2).mean().item()
            pixel_psnr = 10 * np.log10(1.0 / pixel_mse) if pixel_mse > 0 else float("inf")
            max_diff = (ref_out - q_out).abs().max().item()
            print(f"  Pixel MSE:  {pixel_mse:.2e}")
            print(f"  Pixel PSNR: {pixel_psnr:.1f} dB")
            print(f"  Max |diff|: {max_diff:.6f}")

    # Optionally save quantized PyTorch checkpoint
    if args.save_pth:
        if args.mode == "fp16":
            sd_save = {k: v.half() for k, v in state_dict.items()}
        elif args.mode == "int8":
            # Save dequantized (int8 PyTorch native quantization is complex)
            sd_save = {}
            for k, v in state_dict.items():
                if "decoder" in k and "weight" in k and v.dim() >= 2:
                    q, s = quantize_tensor_int8(v)
                    sd_save[k] = q.float() * s.view(-1, *([1] * (v.dim() - 1)))
                    sd_save[k + "._quant_scale"] = s
                else:
                    sd_save[k] = v
        else:
            sd_save = state_dict
        torch.save(sd_save, args.save_pth)
        print(f"\nSaved PyTorch checkpoint: {args.save_pth}")

    print("\nDone.")


if __name__ == "__main__":
    main()
