# hit_generator.py (inference-only)
# Minimal subset of your original hit_generator used by inference.py:
#   - SingleVoiceTCN (model definition)
#   - sample_variation (retrieve-then-generate style local edit sampler)
#
# Drop this file next to inference.py (so `from hit_generator import ...` works).

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Model: TCN with circular padding
# ----------------------------

def circular_pad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Circular padding for 1D conv.
    x: (B, C, T)
    """
    if pad <= 0:
        return x
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


class CircularConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Use odd kernel_size for 'same' length.")
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,  # manual circular padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.dilation * (self.kernel_size - 1) // 2
        x = circular_pad_1d(x, pad)
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CircularConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CircularConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(float(dropout))
        groups = 8 if channels >= 8 else 1
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        return x + residual


class SingleVoiceTCN(nn.Module):
    """
    Input per step:
      - hit_in (0/1, possibly corrupted)
      - vel_in (0..1, possibly corrupted)
      - mask_in (1 if masked/unknown, 0 if known)
    x: (B, 3, T)

    Output:
      - hit_logits: (B, T)
      - vel_mu: (B, T) in [0,1]
      - vel_log_sigma: (B, T) if predict_sigma=True
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 64,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 1, 2, 4),
        dropout: float = 0.1,
        predict_sigma: bool = True,
    ):
        super().__init__()
        self.predict_sigma = bool(predict_sigma)

        self.in_proj = nn.Conv1d(int(in_channels), int(hidden), kernel_size=1)
        self.blocks = nn.ModuleList(
            [TCNBlock(int(hidden), kernel_size=int(kernel_size), dilation=int(d), dropout=float(dropout)) for d in dilations]
        )

        self.hit_head = nn.Conv1d(int(hidden), 1, kernel_size=1)  # logits
        self.vel_head = nn.Conv1d(int(hidden), 1, kernel_size=1)  # mu -> sigmoid
        self.sigma_head = nn.Conv1d(int(hidden), 1, kernel_size=1) if self.predict_sigma else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)

        hit_logits = self.hit_head(h).squeeze(1)            # (B, T)
        vel_mu = torch.sigmoid(self.vel_head(h).squeeze(1)) # (B, T)

        out: Dict[str, torch.Tensor] = {"hit_logits": hit_logits, "vel_mu": vel_mu}
        if self.predict_sigma and self.sigma_head is not None:
            out["vel_log_sigma"] = self.sigma_head(h).squeeze(1)  # (B, T)
        return out


# ----------------------------
# Inference helper: variation sampling
# ----------------------------

@torch.no_grad()
def sample_variation(
    model: nn.Module,
    ref_hit: torch.Tensor,          # (T,) float 0/1
    ref_vel: torch.Tensor,          # (T,) float 0..1 (0 where ref_hit==0)
    *,
    n_iters: int = 8,
    device: str | torch.device = "cpu",
    edit_fraction: float = 0.25,    # fraction of steps to allow the model to rewrite per iteration
    temperature_hit: float = 0.85,  # higher => more random hit sampling
    sigma_floor: float = 0.08,      # min std for vel sampling when predict_sigma=True
    keep_ref_outside_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a "version" of a reference pattern by repeatedly masking + resampling
    a random subset of steps.

    - Starts from the reference pattern.
    - Each iteration picks a fresh mask of steps (size ~= edit_fraction*T).
    - Model predicts hits/vels for masked steps, leaving others unchanged.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    if ref_hit.dim() != 1 or ref_vel.dim() != 1:
        raise ValueError("ref_hit and ref_vel must be 1D tensors of shape (T,)")
    if ref_hit.shape != ref_vel.shape:
        raise ValueError("ref_hit and ref_vel must have the same shape")

    T = int(ref_hit.numel())

    # current state (start from reference)
    hit = ref_hit.to(dev, dtype=torch.float32).clamp(0.0, 1.0).clone()
    vel = ref_vel.to(dev, dtype=torch.float32).clamp(0.0, 1.0).clone()
    vel = vel * hit  # enforce 0 where hit==0

    # clamp params
    n_iters = max(1, int(n_iters))
    edit_fraction = float(max(0.0, min(1.0, edit_fraction)))
    temperature_hit = float(max(1e-3, temperature_hit))
    sigma_floor = float(max(0.0, sigma_floor))

    for _ in range(n_iters):
        # mask: 1 where we ask the model to rewrite
        if edit_fraction <= 0.0:
            break
        mask = (torch.rand((T,), device=dev) < edit_fraction).float()

        # build inputs (B=1, C=3, T)
        hit_in = hit * (1.0 - mask)
        vel_in = vel * (1.0 - mask)
        x = torch.stack([hit_in, vel_in, mask], dim=0).unsqueeze(0)

        out = model(x)
        hit_logits = out["hit_logits"].squeeze(0)  # (T,)
        vel_mu = out["vel_mu"].squeeze(0)          # (T,)

        # sample hits only on masked steps
        p = torch.sigmoid(hit_logits / temperature_hit)
        new_hit = torch.bernoulli(p).float()

        # sample velocities
        if "vel_log_sigma" in out:
            log_sigma = out["vel_log_sigma"].squeeze(0)  # (T,)
            sigma = torch.exp(log_sigma).clamp_min(sigma_floor)
            new_vel = (vel_mu + sigma * torch.randn_like(vel_mu)).clamp(0.0, 1.0)
        else:
            new_vel = vel_mu

        new_vel = new_vel * new_hit  # enforce vel=0 when hit=0

        # update only masked steps
        hit = hit * (1.0 - mask) + new_hit * mask
        vel = vel * (1.0 - mask) + new_vel * mask

        if keep_ref_outside_mask:
            # Re-anchor non-masked steps to the original reference each iter,
            # so the "identity" stays close to the example.
            hit = hit * mask + ref_hit.to(dev, dtype=torch.float32).clamp(0.0, 1.0) * (1.0 - mask)
            vel_ref = (ref_vel.to(dev, dtype=torch.float32).clamp(0.0, 1.0) *
                       ref_hit.to(dev, dtype=torch.float32).clamp(0.0, 1.0))
            vel = vel * mask + vel_ref * (1.0 - mask)

    # final safety clamp
    hit = hit.clamp(0.0, 1.0)
    vel = (vel.clamp(0.0, 1.0) * (hit > 0.5).float())

    return hit, vel
