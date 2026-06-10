import math
from typing import Optional

import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

try:
    # Optional: real Vision Mamba (state-space model)
    # pip install mamba-ssm causal-conv1d
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:  # pragma: no cover
    Mamba = None
    _HAS_MAMBA = False

from .climate_swinir_arch import SwinIRClimate


class _GRU1DStub(nn.Module):
    """A lightweight fallback replacing Mamba when mamba-ssm is unavailable.

    Operates on sequence [B, N, C], where N=H*W. It uses a GRU to model
    long-range dependencies with O(N) complexity. This is NOT equivalent to
    Mamba but enables code to run without extra deps.
    """

    def __init__(self, dim: int, hidden_ratio: float = 1.0):
        super().__init__()
        hidden = max(dim, int(dim * hidden_ratio))
        self.proj_in = nn.Linear(dim, hidden)
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.proj_out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        out, _ = self.gru(x)
        return self.proj_out(out)


class VMamba1D(nn.Module):
    """1D Mamba wrapper with graceful fallback to GRU.

    Inputs/Outputs are sequences [B, N, C].
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if _HAS_MAMBA:
            self.core = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.core = _GRU1DStub(dim, hidden_ratio=float(expand))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class VMamba2D(nn.Module):
    """Directional 2D VMamba using 1D Mamba over flattened scans.

    For each direction (→, ←, ↓, ↑), we flatten HxW into sequence, run Mamba,
    and aggregate results. This mimics multi-directional selective scan in
    vision Mamba variants with minimal complexity.
    """

    def __init__(self, dim: int, directions: int = 4, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.directions = directions
        self.branches = nn.ModuleList([
            VMamba1D(dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(directions)
        ])
        self.proj = nn.Conv2d(dim * directions, dim, kernel_size=1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def _seq_scan(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: [B, C, H, W] → [B, N, C]
        if mode == 'lr':  # left->right
            seq = x.flatten(2).transpose(1, 2)
        elif mode == 'rl':  # right->left
            seq = torch.flip(x, dims=[-1]).flatten(2).transpose(1, 2)
        elif mode == 'tb':  # top->bottom (scan along H)
            seq = x.transpose(2, 3).flatten(2).transpose(1, 2)
        elif mode == 'bt':  # bottom->top
            seq = torch.flip(x.transpose(2, 3), dims=[-1]).flatten(2).transpose(1, 2)
        else:
            raise ValueError(f'Unknown scan mode: {mode}')
        return seq

    def _seq_to_map(self, y: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
        # y: [B, N, C] → [B, C, H, W]
        B, C, H, W = ref.shape
        fmap = y.transpose(1, 2).reshape(B, C, H * W)
        if mode == 'lr':
            fmap = fmap.view(B, C, H, W)
        elif mode == 'rl':
            fmap = fmap.view(B, C, H, W)
            fmap = torch.flip(fmap, dims=[-1])
        elif mode == 'tb':
            fmap = fmap.view(B, C, W, H).transpose(2, 3)
        elif mode == 'bt':
            fmap = fmap.view(B, C, W, H).transpose(2, 3)
            fmap = torch.flip(fmap, dims=[-2])
        else:
            raise ValueError(f'Unknown scan mode: {mode}')
        return fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        modes = ['lr', 'rl', 'tb', 'bt'][: self.directions]
        outs = []
        for mode, branch in zip(modes, self.branches):
            seq = self._seq_scan(x, mode=mode)
            seq = self.norm(seq)
            out = branch(seq)
            fmap = self._seq_to_map(out, ref=x, mode=mode)
            outs.append(fmap)
        y = torch.cat(outs, dim=1)
        y = self.proj(y)
        return x + self.drop_path(y)


class DropPath(nn.Module):
    """Stochastic Depth: per-sample drop-path.
    Copied minimal implementation to avoid extra deps.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


@ARCH_REGISTRY.register()
class SwinIRVMambaClimate(SwinIRClimate):
    """SwinIR + Spatial-Temporal VMamba hybrid for climate downscaling.

    - Keeps terrain FiLM injection from parent class
    - Inserts directional VMamba blocks after early stem before SwinIR body
    - Optional: can be extended to temporal VMamba by flattening T*H*W
    """

    def __init__(self, vmamba_depth: int = 2, vmamba_d_state: int = 16, vmamba_expand: int = 2,
                 vmamba_drop_path: float = 0.0, vmamba_directions: int = 4, **kwargs):
        super().__init__(**kwargs)
        dim = self.embed_dim if hasattr(self, 'embed_dim') else 96
        blocks = []
        for i in range(vmamba_depth):
            blocks.append(VMamba2D(dim=dim, directions=vmamba_directions,
                                   d_state=vmamba_d_state, expand=vmamba_expand,
                                   drop_path=vmamba_drop_path))
        self.vmamba = nn.Sequential(*blocks)

    def forward(self, x):
        # Reuse parent logic but insert vmamba after stem and FiLM topo injection
        if isinstance(x, dict):
            hgt = x.get('hgt', None)
            x = x['lq']
        else:
            hgt = None

        x = self.conv_first(x)

        if getattr(self, 'add_hgt', False) and hgt is not None:
            # identical to parent: per-sample z-score and FiLM
            if hgt.dim() == 3:
                hgt = hgt.unsqueeze(0)
                if hgt.size(0) != x.size(0):
                    hgt = hgt.repeat(x.size(0), 1, 1, 1)
            if hgt.dim() == 4 and hgt.size(1) > 1:
                hgt = hgt.mean(dim=1, keepdim=True)
            hgt_mean = hgt.mean(dim=(-2, -1), keepdim=True)
            hgt_std = hgt.std(dim=(-2, -1), keepdim=True)
            hgt = (hgt - hgt_mean) / (hgt_std + 1e-6)

            topo_feat = self.hgt_net(hgt)
            gamma = 1.0 + self.hgt_alpha * self.film_gamma(topo_feat)
            beta = self.hgt_alpha * self.film_beta(topo_feat)
            x = x * gamma + beta

        # Insert VMamba hybrid module
        x = self.vmamba(x)

        if self.upsampler == 'pixelshuffle':
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'bilinear+conv':
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=5, mode='bilinear', aligh_corners=False)))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', aligh_corners=False)))
            x = self.conv_last(self.lrelu(self.conv_hr(torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', aligh_corners=False))))
        else:
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_last(x)
        return self.act(x)


