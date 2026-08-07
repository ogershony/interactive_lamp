#!/usr/bin/env python3
"""
Masked transformer denoiser for conditional flow matching over whole
variable-length lamp clips.

Input:  x_t (B, T, 9) noisy frames, padding mask (B, T)
Cond:   16-d unit-L2 affect vector + log-duration + flow time t
Output: velocity field v(x_t, t | cond) of the same shape

Conditioning enters through AdaLN-Zero (DiT-style): each block's
LayerNorms are modulated by (shift, scale, gate) regressed from the
summed condition embedding; gates start at zero so the network begins as
an identity and learns conditioning gradually. Classifier-free guidance
uses a LEARNED null embedding (`null_cond`) -- not a uniform affect
vector, which is a legitimate in-distribution "ambiguous" label.

A caption-conditioning slot is reserved: pass `extra_cond` (B, D_extra)
to Denoiser.forward and set extra_cond_dim > 0 at construction. Unused
in Phase 5.
"""

import math

import torch
import torch.nn as nn

from dataset import N_CHANNELS


def timestep_embedding(t, dim, max_period=1000.0):
    """Sinusoidal embedding of flow time t in [0, 1] (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32,
                                     device=t.device) / half)
    args = t[:, None].float() * freqs[None] * max_period
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    """Pre-LN transformer block with AdaLN-Zero modulation."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(),
                                 nn.Linear(ff, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond, pad_mask):
        # cond: (B, d); pad_mask: (B, T) True on REAL frames
        s1, b1, g1, s2, b2, g2 = self.mod(cond)[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        h, _ = self.attn(h, h, h, key_padding_mask=~pad_mask,
                         need_weights=False)
        x = x + g1 * h
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.mlp(h)
        return x


class Denoiser(nn.Module):
    """Velocity-field transformer: (noisy clip, mask, t, affect,
    log-duration) -> per-frame velocity toward the data. See the module
    docstring for how conditioning enters (AdaLN-Zero + learned null)."""

    def __init__(self, d=160, heads=4, ff=640, layers=5, t_max=240,
                 n_affect=16, extra_cond_dim=0):
        super().__init__()
        self.t_max = t_max
        # +1 input channel: normalized phase (frame index / clip length),
        # helps the model plan endings on variable-length clips
        self.in_proj = nn.Linear(N_CHANNELS + 1, d)
        self.pos = nn.Parameter(torch.randn(1, t_max, d) * 0.02)
        self.affect_mlp = nn.Sequential(nn.Linear(n_affect, d), nn.SiLU(),
                                        nn.Linear(d, d))
        self.null_cond = nn.Parameter(torch.zeros(d))
        self.dur_mlp = nn.Sequential(nn.Linear(1, d), nn.SiLU(),
                                     nn.Linear(d, d))
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(),
                                      nn.Linear(d, d))
        self.time_dim = d
        self.extra_proj = (nn.Linear(extra_cond_dim, d)
                           if extra_cond_dim else None)
        self.blocks = nn.ModuleList(Block(d, heads, ff)
                                    for _ in range(layers))
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_mod = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.out_mod.weight)
        nn.init.zeros_(self.out_mod.bias)
        self.out_proj = nn.Linear(d, N_CHANNELS)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def cond_vector(self, label, log_dur, t, drop=None, extra_cond=None):
        """label (B,16) unit-L2 or None (=> null); drop: bool (B,) mask
        replacing the affect embedding with the learned null."""
        B = t.shape[0]
        if label is None:
            aff = self.null_cond[None].expand(B, -1)
        else:
            aff = self.affect_mlp(label)
            if drop is not None:
                aff = torch.where(drop[:, None],
                                  self.null_cond[None].expand(B, -1), aff)
        c = (aff + self.dur_mlp(log_dur[:, None])
             + self.time_mlp(timestep_embedding(t, self.time_dim)))
        if extra_cond is not None and self.extra_proj is not None:
            c = c + self.extra_proj(extra_cond)
        return c

    def forward(self, x, mask, t, label, log_dur, drop=None,
                extra_cond=None):
        B, T, _ = x.shape
        phase = (torch.arange(T, device=x.device).float()[None]
                 / mask.sum(dim=1, keepdim=True).clamp(min=1))
        h = self.in_proj(torch.cat([x, phase[..., None]], dim=-1))
        h = h + self.pos[:, :T]
        cond = self.cond_vector(label, log_dur, t, drop, extra_cond)
        for blk in self.blocks:
            h = blk(h, cond, mask)
        s, b = self.out_mod(cond)[:, None].chunk(2, dim=-1)
        return self.out_proj(self.out_norm(h) * (1 + s) + b)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
