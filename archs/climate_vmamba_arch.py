"""
Pure VMamba + HGT architecture for climate downscaling
Removes SwinIR dependency and uses pure VMamba blocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:
    Mamba = None
    _HAS_MAMBA = False

from .utils import HGTNet


class _GRU1DStub(nn.Module):
    """GRU fallback when mamba-ssm is unavailable."""
    
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
    """1D Mamba wrapper with graceful fallback to GRU."""
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if _HAS_MAMBA:
            self.core = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.core = _GRU1DStub(dim, hidden_ratio=float(expand))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class DropPath(nn.Module):
    """Stochastic Depth implementation."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class VMambaBlock(nn.Module):
    """Pure VMamba block for 2D spatial processing."""
    
    def __init__(self, dim: int, directions: int = 4, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, drop_path: float = 0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.directions = directions
        
        # Pre-norm
        self.norm1 = norm_layer(dim)
        
        # Multi-directional VMamba branches
        self.branches = nn.ModuleList([
            VMamba1D(dim, d_state=d_state, d_conv=d_conv, expand=expand) 
            for _ in range(directions)
        ])
        
        # Fusion projection
        self.proj = nn.Conv2d(dim * directions, dim, kernel_size=1)
        
        # Feed-forward network
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _seq_scan(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Convert 2D feature map to sequence for different scan directions."""
        if mode == 'lr':  # left->right
            seq = x.flatten(2).transpose(1, 2)
        elif mode == 'rl':  # right->left
            seq = torch.flip(x, dims=[-1]).flatten(2).transpose(1, 2)
        elif mode == 'tb':  # top->bottom
            seq = x.transpose(2, 3).flatten(2).transpose(1, 2)
        elif mode == 'bt':  # bottom->top
            seq = torch.flip(x.transpose(2, 3), dims=[-1]).flatten(2).transpose(1, 2)
        else:
            raise ValueError(f'Unknown scan mode: {mode}')
        return seq

    def _seq_to_map(self, y: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
        """Convert sequence back to 2D feature map."""
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
        shortcut = x
        
        # VMamba processing
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        modes = ['lr', 'rl', 'tb', 'bt'][:self.directions]
        outs = []
        
        for mode, branch in zip(modes, self.branches):
            seq = self._seq_scan(x, mode=mode)
            out = branch(seq)
            fmap = self._seq_to_map(out, ref=x, mode=mode)
            outs.append(fmap)
        
        y = torch.cat(outs, dim=1)
        y = self.proj(y)
        x = shortcut + self.drop_path(y)
        
        # Feed-forward
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = shortcut + self.drop_path(x)
        
        return x


@ARCH_REGISTRY.register()
class VMambaClimate(nn.Module):
    """Pure VMamba architecture with HGT gating for climate downscaling."""
    
    def __init__(self, 
                 img_size=16,
                 in_chans=3, 
                 num_out_ch=1,
                 embed_dim=96,
                 depths=[6, 6, 6, 6],
                 directions=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 drop_path_rate=0.1,
                 upscale=4,
                 img_range=1.0,
                 upsampler='pixelshuffle',
                 add_hgt=True,
                 hgt_alpha_init=0.01,
                 **kwargs):
        
        super().__init__()
        
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_out_ch = num_out_ch
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.img_range = img_range
        self.upsampler = upsampler
        self.add_hgt = add_hgt
        
        # Stem: convert input to embedding
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # HGT (topographic) gating
        if add_hgt:
            self.hgt_net = HGTNet(upscale=upscale, embed_dim=embed_dim)
            self.film_gamma = nn.Conv2d(embed_dim, embed_dim, 1)
            self.film_beta = nn.Conv2d(embed_dim, embed_dim, 1)
            self.hgt_alpha = nn.Parameter(torch.tensor(hgt_alpha_init))
        
        # Build VMamba layers
        num_layers = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        self.layers = nn.ModuleList()
        layer_idx = 0
        
        for stage_idx, depth in enumerate(depths):
            stage_layers = []
            for _ in range(depth):
                stage_layers.append(
                    VMambaBlock(
                        dim=embed_dim,
                        directions=directions,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        drop_path=dpr[layer_idx]
                    )
                )
                layer_idx += 1
            self.layers.append(nn.Sequential(*stage_layers))
        
        # Output projection
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # Upsampling
        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        elif upsampler == 'bilinear+conv':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.conv_up1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_hr = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.act = nn.Identity()

    def forward(self, x):
        # Handle dict input (with HGT)
        if isinstance(x, dict):
            hgt = x.get('hgt', None)
            x = x['lq']
        else:
            hgt = None
        
        # Stem
        x = self.conv_first(x)
        
        # HGT gating (FiLM modulation) - always apply if HGT is enabled
        if self.add_hgt:
            if hgt is not None:
                # Normalize HGT per sample
                if hgt.dim() == 3:
                    hgt = hgt.unsqueeze(0)
                    if hgt.size(0) != x.size(0):
                        hgt = hgt.repeat(x.size(0), 1, 1, 1)
                if hgt.dim() == 4 and hgt.size(1) > 1:
                    hgt = hgt.mean(dim=1, keepdim=True)
                
                hgt_mean = hgt.mean(dim=(-2, -1), keepdim=True)
                hgt_std = hgt.std(dim=(-2, -1), keepdim=True)
                hgt = (hgt - hgt_mean) / (hgt_std + 1e-6)
                
                # FiLM gating
                topo_feat = self.hgt_net(hgt)
                gamma = 1.0 + self.hgt_alpha * self.film_gamma(topo_feat)
                beta = self.hgt_alpha * self.film_beta(topo_feat)
                x = x * gamma + beta
            else:
                # Ensure HGT parameters are used even when hgt is None
                dummy_hgt = torch.zeros(x.size(0), 1, x.size(2)*4, x.size(3)*4, 
                                      device=x.device, dtype=x.dtype)
                topo_feat = self.hgt_net(dummy_hgt)
                # Apply minimal modulation to preserve gradients
                gamma = 1.0 + 0.0 * self.hgt_alpha * self.film_gamma(topo_feat)
                beta = 0.0 * self.hgt_alpha * self.film_beta(topo_feat)
                x = x * gamma + beta
        
        # VMamba processing
        shortcut = x
        for stage in self.layers:
            x = stage(x)
        
        x = self.conv_after_body(x) + shortcut
        
        # Upsampling
        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'bilinear+conv':
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        
        return self.act(x)


class Upsample(nn.Sequential):
    """Upsample module using pixel shuffle."""
    
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported')
        super().__init__(*m)
