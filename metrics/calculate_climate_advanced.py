import torch
import torch.nn.functional as F
from basicsr.utils.registry import METRIC_REGISTRY
import torch.nn.functional as F


def _per_channel_flat(x: torch.Tensor) -> torch.Tensor:
    # 支持 [B,C,H,W] 或 [C,H,W]；统一到 [C, N]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    b, c, h, w = x.shape
    return x.reshape(b, c, -1).permute(1, 0, 2).reshape(c, -1)


@METRIC_REGISTRY.register()
def calculate_climate_acc(img: torch.Tensor, img2: torch.Tensor, crop_border: int):
    """Anomaly Correlation Coefficient (逐通道)。
    逐样本去空间均值并计算相关系数，然后对 batch 取平均。
    返回 list[Tensor]，与现有 metrics 风格一致。
    """
    if crop_border:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    accs = []
    eps = 1e-8
    channels = img.shape[1] if img.dim() == 4 else 1
    for c in range(channels):
        x = img[:, [c]] if img.dim() == 4 else img.unsqueeze(1)
        y = img2[:, [c]] if img2.dim() == 4 else img2.unsqueeze(1)
        # 去除每个样本自己的空间均值
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        y = y - y.mean(dim=(-2, -1), keepdim=True)
        B = x.shape[0]
        rs = []
        for b in range(B):
            xb = x[b].reshape(1, -1).squeeze(0)
            yb = y[b].reshape(1, -1).squeeze(0)
            xb = xb - xb.mean()
            yb = yb - yb.mean()
            denom = (xb.std() * yb.std()) + eps
            rs.append(((xb * yb).mean()) / denom)
        accs.append(torch.stack(rs).mean())
    return accs


def _radial_psd(a: torch.Tensor) -> torch.Tensor:
    # a: [B,1,H,W] 或 [B,H,W]，输出径向平均功率谱 1D Tensor
    if a.dim() == 4:
        a = a.mean(dim=0)  # [1,H,W]
    if a.dim() == 3:
        a = a[0]
    A = torch.fft.rfft2(a)
    P = (A.real ** 2 + A.imag ** 2)
    H, Wh = P.shape
    yy = torch.arange(H, device=P.device).unsqueeze(1).float()
    xx = torch.arange(Wh, device=P.device).unsqueeze(0).float()
    rr = torch.sqrt((yy - H / 2) ** 2 + (xx - 0) ** 2)
    r = rr.round().long().clamp(min=0)
    rmax = int(r.max().item())
    psd = []
    for k in range(rmax + 1):
        mask = (r == k)
        if mask.any():
            psd.append(P[mask].mean())
        else:
            psd.append(torch.tensor(0., device=P.device))
    return torch.stack(psd)


@METRIC_REGISTRY.register()
def calculate_climate_psd(img: torch.Tensor, img2: torch.Tensor, crop_border: int):
    """PSD 差异（L1），逐通道。
    返回 list[Tensor]，值越小越好。
    """
    if crop_border:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    diffs = []
    channels = img.shape[1] if img.dim() == 4 else 1
    for c in range(channels):
        x = img[:, [c]] if img.dim() == 4 else img.unsqueeze(1)
        y = img2[:, [c]] if img2.dim() == 4 else img2.unsqueeze(1)
        psd1 = _radial_psd(x)
        psd2 = _radial_psd(y)
        # 对齐长度（不同尺寸 pad 策略可能使长度略有差异）
        m = min(psd1.numel(), psd2.numel())
        diffs.append(F.l1_loss(psd1[:m], psd2[:m]))
    return diffs


@METRIC_REGISTRY.register()
def calculate_climate_psd_log(img: torch.Tensor, img2: torch.Tensor, crop_border: int):
    """对数功率谱差（逐通道）。更弱的幅度敏感度，适合跨场景比较。
    返回 list[Tensor]，值越小越好。
    """
    if crop_border:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    diffs = []
    eps = 1e-12
    channels = img.shape[1] if img.dim() == 4 else 1
    for c in range(channels):
        x = img[:, [c]] if img.dim() == 4 else img.unsqueeze(1)
        y = img2[:, [c]] if img2.dim() == 4 else img2.unsqueeze(1)
        psd1 = _radial_psd(x)
        psd2 = _radial_psd(y)
        m = min(psd1.numel(), psd2.numel())
        dlog = torch.abs(torch.log10(psd1[:m] + eps) - torch.log10(psd2[:m] + eps)).mean()
        diffs.append(dlog)
    return diffs


@METRIC_REGISTRY.register()
def calculate_climate_psd_rel(img: torch.Tensor, img2: torch.Tensor, crop_border: int):
    """相对功率谱误差（逐通道）：sum|P1-P2| / sum P2。
    返回 list[Tensor]，值越小越好。
    """
    if crop_border:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    diffs = []
    eps = 1e-12
    channels = img.shape[1] if img.dim() == 4 else 1
    for c in range(channels):
        x = img[:, [c]] if img.dim() == 4 else img.unsqueeze(1)
        y = img2[:, [c]] if img2.dim() == 4 else img2.unsqueeze(1)
        psd1 = _radial_psd(x)
        psd2 = _radial_psd(y)
        m = min(psd1.numel(), psd2.numel())
        num = torch.abs(psd1[:m] - psd2[:m]).sum()
        den = torch.abs(psd2[:m]).sum() + eps
        diffs.append(num / den)
    return diffs


# ------------------------- ACC variants (more conservative) ------------------------- #
def _corr_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.std() * y.std() + eps)


@METRIC_REGISTRY.register()
def calculate_climate_acc_tiles(img: torch.Tensor, img2: torch.Tensor, crop_border: int, tiles: int = 4):
    """Tile-ACC: 将图像分块后逐样本计算相关，再对所有块/样本取平均。
    更强调结构一致性，抑制均匀区域对相关性的“虚高”。
    """
    if crop_border:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    B, C, H, W = img.shape
    h = H // tiles
    w = W // tiles
    outs = []
    for c in range(C):
        vals = []
        for i in range(tiles):
            for j in range(tiles):
                xs = img[:, c, i * h:(i + 1) * h, j * w:(j + 1) * w]
                ys = img2[:, c, i * h:(i + 1) * h, j * w:(j + 1) * w]
                for b in range(B):
                    xb = xs[b].reshape(-1)
                    yb = ys[b].reshape(-1)
                    if xb.numel() >= 16 and xb.std() > 0 and yb.std() > 0:
                        vals.append(_corr_1d(xb, yb))
        outs.append(torch.stack(vals).mean() if len(vals) > 0 else torch.tensor(0.0, device=img.device))
    return outs


@METRIC_REGISTRY.register()
def calculate_climate_acc_highpass(img: torch.Tensor, img2: torch.Tensor, crop_border: int, k: int = 5):
    """Highpass-ACC: 先做均值滤波提取背景并高通，随后按逐样本计算相关再平均。
    用于强调中小尺度结构一致性。
    """
    if crop_border:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    pad = k // 2
    ker = torch.ones(1, 1, k, k, device=img.device) / (k * k)

    def hp(a: torch.Tensor) -> torch.Tensor:
        # 按通道分组卷积做平滑
        B, C, H, W = a.shape
        a_blur = F.conv2d(a, ker.expand(C, 1, k, k), padding=pad, groups=C)
        return a - a_blur

    xh = hp(img)
    yh = hp(img2)
    outs = []
    for c in range(img.shape[1]):
        vals = []
        for b in range(img.shape[0]):
            xb = xh[b, c].reshape(-1)
            yb = yh[b, c].reshape(-1)
            if xb.std() > 0 and yb.std() > 0:
                vals.append(_corr_1d(xb, yb))
        outs.append(torch.stack(vals).mean() if len(vals) > 0 else torch.tensor(0.0, device=img.device))
    return outs


