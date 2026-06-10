import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from .utils import Activation


@ARCH_REGISTRY.register()
class SRCNNClimate(nn.Module):
    """Simple SRCNN for climate downscaling.

    - Upsample low-resolution input to target size using bilinear interpolation
      and then apply three conv layers (9x9, 1x1, 5x5) as in SRCNN.
    - Supports dict or tensor input; when dict, expects key 'lq'.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 1,
        upscale: int = 4,
        feat1: int = 64,
        feat2: int = 32,
        activation: str = 'none',
    ) -> None:
        super().__init__()
        self.upscale = upscale

        # SRCNN backbone
        self.conv1 = nn.Conv2d(num_in_ch, feat1, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feat1, feat2, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(feat2, num_out_ch, kernel_size=5, stride=1, padding=2)

        self.act = Activation(activation)

    def forward(self, x):
        # accept dict or tensor
        if isinstance(x, dict):
            x = x['lq']

        # upsample first as in classical SRCNN
        x = F.interpolate(
            x, scale_factor=self.upscale, mode='bilinear', align_corners=False
        )

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return self.act(x)


