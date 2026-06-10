import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.colors import Normalize

try:
    import torch
except Exception:
    torch = None


def read_image(path: Path) -> np.ndarray:
    """Read an image file into a float32 numpy array in [0,1].
    Supports PNG/JPG. Returns HxWxC (3) if possible; converts L to RGB.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def resize_to_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resize image a to match b's spatial size using PIL bilinear."""
    if a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]:
        return a
    img = Image.fromarray((a * 255.0).clip(0, 255).astype(np.uint8))
    img = img.resize((b.shape[1], b.shape[0]), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def try_import_skimage_ssim():
    try:
        from skimage.metrics import structural_similarity as skimage_ssim
        return skimage_ssim
    except Exception:
        return None


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM in RGB; if skimage not available, fall back to negative MSE proxy."""
    ssim_fn = try_import_skimage_ssim()
    if ssim_fn is not None:
        # compute mean SSIM over channels
        ssim_vals = []
        for c in range(3):
            ssim_val = ssim_fn(img1[..., c], img2[..., c], data_range=1.0)
            ssim_vals.append(ssim_val)
        return float(np.mean(ssim_vals))
    # Fallback: use a bounded proxy based on MSE
    mse = float(np.mean((img1 - img2) ** 2))
    proxy = max(0.0, 1.0 - mse * 10.0)
    return proxy


def find_candidate_pairs(vis_dir: Path):
    """Find (pred, gt, lq) image triplets in a visualization directory.
    We use heuristic filename matching to pair files by a common stem or index.
    Returns a dict: key=index_str, value=dict(pred=Path, gt=Path or None, lq=Path or None)
    """
    all_files = [p for p in vis_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    # classify by tags in filename
    def tag(path: Path):
        name = path.name.lower()
        if "pred" in name or "sr" in name or "output" in name or "out" in name:
            return "pred"
        if "gt" in name or "target" in name or "hr" in name or "groundtruth" in name or "gt_" in name:
            return "gt"
        if "lq" in name or "input" in name or "lr" in name or "bicubic" in name:
            return "lq"
        # default unknown -> try to infer later; treat as pred by default
        return "pred"

    # extract index pattern like idx_0057 or 0057
    def extract_index(name: str) -> str:
        m = re.search(r"idx[_-]?(\d+)", name)
        if m:
            return m.group(1)
        m = re.search(r"(\d{4,})", name)
        if m:
            return m.group(1)
        # fallback to basename without suffix
        return Path(name).stem

    buckets = {}
    for p in all_files:
        role = tag(p)
        idx = extract_index(p.name)
        buckets.setdefault(idx, {}).setdefault(role, []).append(p)

    # choose the first for each role
    pairs = {}
    for idx, roles in buckets.items():
        pred = None
        for cand in roles.get("pred", []):
            pred = cand
            break
        if pred is None:
            continue
        gt = roles.get("gt", [None])[0]
        lq = roles.get("lq", [None])[0]
        pairs[idx] = {"pred": pred, "gt": gt, "lq": lq}
    return pairs


def pick_best_by_ssim(pairs: dict) -> str:
    """Return the index key with highest SSIM between pred and GT.
    If GT missing, return the first available index.
    """
    best_idx = None
    best_score = -1.0
    for idx, trio in pairs.items():
        pred_path = trio["pred"]
        gt_path = trio.get("gt")
        if gt_path is None or not gt_path.exists():
            if best_idx is None:
                best_idx = idx
            continue
        pred = read_image(pred_path)
        gt = read_image(gt_path)
        pred = resize_to_match(pred, gt)
        score = compute_ssim(pred, gt)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def scalar_to_rgb(arr: np.ndarray, vmin: float = None, vmax: float = None, cmap_name: str = "cividis") -> np.ndarray:
    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(arr, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            if lo == hi:
                hi = lo + 1e-6
        vmin, vmax = lo, hi
    normed = (arr - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = cmap(normed)[..., :3]
    return rgb.astype(np.float32)


def load_lr_gt_from_pt(index_int: int, hr_size=None) -> tuple:
    """Fallback: load LR/GT from dict_s2s_test.pt and colorize with jet.
    Returns (lr_rgb, gt_rgb) in [0,1].
    """
    dataset_dir = Path(os.environ.get("DATAROOT_GT", "/home/lyb/diffusion/main2 copy/dataset"))
    pt_path = dataset_dir / "dict_s2s_test.pt"
    if not pt_path.exists() or torch is None:
        return None, None
    try:
        data = torch.load(str(pt_path), map_location="cpu")
    except Exception:
        return None, None
    # Try common keys
    lr = None
    hr = None
    for k in ["LR_input", "LR", "lr"]:
        if k in data:
            lr = data[k]
            break
    for k in ["HR_target", "HR", "hr", "GT", "gt"]:
        if k in data:
            hr = data[k]
            break
    if lr is None or hr is None:
        return None, None
    # tensors -> numpy
    if hasattr(lr, "numpy"):
        lr = lr.numpy()
    if hasattr(hr, "numpy"):
        hr = hr.numpy()
    # Shapes expected: lr (C,N,h,w) or (N,C,h,w)
    def ensure_nchw(x):
        arr = np.array(x)
        if arr.ndim == 4 and arr.shape[0] in (1, 3):
            # likely (C,N,H,W) -> (N,C,H,W)
            arr = np.transpose(arr, (1, 0, 2, 3))
        return arr
    lr = ensure_nchw(lr)
    hr = ensure_nchw(hr)
    n = min(lr.shape[0], hr.shape[0])
    idx = index_int % n
    lr_sample = lr[idx]
    hr_sample = hr[idx]
    # lr may have 3 channels; pick channel 0 for display
    lr_2d = lr_sample[0]
    hr_2d = hr_sample[0]
    lr_rgb = scalar_to_rgb(lr_2d)
    hr_rgb = scalar_to_rgb(hr_2d)
    if hr_size is not None:
        lr_rgb = resize_to_match(lr_rgb, np.zeros((hr_size[0], hr_size[1], 3), dtype=np.float32))
    return lr_rgb, hr_rgb


def find_image_by_index(vis_dir: Path, index_key: str, prefer_roles=("pred", "gt", "lq")):
    pairs = find_candidate_pairs(vis_dir)
    trio = pairs.get(index_key)
    if trio:
        return trio
    # fallback: fuzzy search
    hits = {}
    for p in vis_dir.rglob("*"):
        if index_key in p.name:
            hits.setdefault("other", []).append(p)
    if not hits:
        return {"pred": None, "gt": None, "lq": None}
    # choose one as pred
    return {"pred": hits["other"][0], "gt": None, "lq": None}


def make_grid(tiles, titles, ncols=6, figsize=(10, 10), out_path: Path = None):
    nrows = len(tiles)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(tiles[r]):
                img = tiles[r][c]
                if img is not None:
                    ax.imshow(img)
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
    plt.tight_layout(pad=0.5)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        return fig


def main():
    results_root = Path(os.environ.get(
        "RESULTS_ROOT",
        "/home/lyb/diffusion/main2 copy/TransfomerDownscaling-main/results",
    ))

    dirs = {
        "SRCNN": results_root / "SRCNN_climate_infer_2019" / "visualization",
        "UNet": results_root / "UNet_climate_infer_2019" / "visualization",
        "Uformer": results_root / "Uformer_climate_infer_2019" / "visualization",
        "SwinIR": results_root / "SwinIR_climate_infer_2019" / "visualization",
        "SwinIR-HGT": results_root / "SwinIR_climate_infer_hgt_2019" / "visualization",
    }

    for k, v in dirs.items():
        if not v.exists():
            print(f"[Warn] Visualization dir missing: {k} -> {v}")

    # 1) Find best index in SwinIR-HGT
    hgt_pairs = find_candidate_pairs(dirs["SwinIR-HGT"])
    if not hgt_pairs:
        print("[Error] No visualization images found for SwinIR-HGT.")
        sys.exit(1)
    best_idx = pick_best_by_ssim(hgt_pairs)
    print(f"Best SwinIR-HGT index: {best_idx}")

    # 2) Gather images for grid by the same index across models
    columns = ["Input (LR↑)", "Ground Truth", "SRCNN", "UNet", "Uformer", "SwinIR", "SwinIR-HGT"]

    # Attempt to use LQ/GT from SwinIR-HGT folder as reference
    ref_trio = hgt_pairs.get(best_idx, {"lq": None, "gt": None})
    ref_lq = read_image(ref_trio["lq"]) if ref_trio.get("lq") else None
    ref_gt = read_image(ref_trio["gt"]) if ref_trio.get("gt") else None

    # Fallback: if either missing, reconstruct from dataset pt
    if ref_gt is None or ref_lq is None:
        try:
            idx_int = int(best_idx)
        except Exception:
            idx_int = 0
        # Also get scalar arrays to compute shared vmin/vmax
        dataset_dir = Path(os.environ.get("DATAROOT_GT", "/home/lyb/diffusion/main2 copy/dataset"))
        pt_path = dataset_dir / "dict_s2s_test.pt"
        lr_rgb, gt_rgb = load_lr_gt_from_pt(idx_int, hr_size=None)
        if ref_gt is None and gt_rgb is not None:
            ref_gt = gt_rgb
        if ref_lq is None and lr_rgb is not None:
            # if we have GT, resize LR to GT size
            if ref_gt is not None:
                ref_lq = resize_to_match(lr_rgb, ref_gt)
            else:
                ref_lq = lr_rgb

    # Enforce consistent color range for LR and GT using GT percentiles
    shared_vmin = None
    shared_vmax = None
    # Try to recover scalar GT to compute bounds
    if torch is not None:
        try:
            dataset_dir = Path(os.environ.get("DATAROOT_GT", "/home/lyb/diffusion/main2 copy/dataset"))
            data = torch.load(str(dataset_dir / "dict_s2s_test.pt"), map_location="cpu")
            hr = None
            for k in ["HR_target", "HR", "hr", "GT", "gt"]:
                if k in data:
                    hr = data[k]
                    break
            if hr is not None:
                arr = np.array(hr)
                if arr.ndim == 4 and arr.shape[0] in (1, 3):
                    arr = np.transpose(arr, (1, 0, 2, 3))
                idx_int = int(best_idx) if str(best_idx).isdigit() else 0
                gt_2d = arr[idx_int, 0]
                shared_vmin, shared_vmax = np.nanpercentile(gt_2d, [1.0, 99.0])
        except Exception:
            pass

    # If we have shared bounds and LR/GT already colorized, recolor them with consistent scale
    if shared_vmin is not None and shared_vmax is not None:
        # We need scalar LR/GT; if only RGB available, keep as is.
        # Attempt re-load scalars from pt
        try:
            dataset_dir = Path(os.environ.get("DATAROOT_GT", "/home/lyb/diffusion/main2 copy/dataset"))
            data = torch.load(str(dataset_dir / "dict_s2s_test.pt"), map_location="cpu") if torch is not None else None
            if data is not None:
                lr = None
                hr = None
                for k in ["LR_input", "LR", "lr"]:
                    if k in data:
                        lr = data[k]
                        break
                for k in ["HR_target", "HR", "hr", "GT", "gt"]:
                    if k in data:
                        hr = data[k]
                        break
                if lr is not None and hr is not None:
                    lr = np.array(lr)
                    hr = np.array(hr)
                    if lr.ndim == 4 and lr.shape[0] in (1, 3):
                        lr = np.transpose(lr, (1, 0, 2, 3))
                    if hr.ndim == 4 and hr.shape[0] in (1, 3):
                        hr = np.transpose(hr, (1, 0, 2, 3))
                    idx_int = int(best_idx) if str(best_idx).isdigit() else 0
                    lr_2d = lr[idx_int, 0]
                    hr_2d = hr[idx_int, 0]
                    ref_lq = scalar_to_rgb(lr_2d, shared_vmin, shared_vmax)
                    ref_gt = scalar_to_rgb(hr_2d, shared_vmin, shared_vmax)
        except Exception:
            pass

    tiles = []
    row_pred = []
    row_pred.append(ref_lq)
    row_pred.append(ref_gt)

    # helper to fetch model pred and resize to GT if needed
    def fetch_model_pred(model_key: str):
        vis_dir = dirs[model_key]
        trio = find_image_by_index(vis_dir, best_idx)
        pred_path = trio.get("pred")
        if pred_path is None or not Path(pred_path).exists():
            return None
        img = read_image(pred_path)
        if ref_gt is not None:
            img = resize_to_match(img, ref_gt)
        return img

    model_keys = ["SRCNN", "UNet", "Uformer", "SwinIR", "SwinIR-HGT"]
    preds = {}
    for m in model_keys:
        preds[m] = fetch_model_pred(m)
        row_pred.append(preds[m])

    tiles.append(row_pred)

    # Build error row: |pred - GT| (RGB), shown as scalar mean error heatmap with shared colorbar later
    error_scalars = {}
    for m in model_keys:
        if preds[m] is None or ref_gt is None:
            error_scalars[m] = None
        else:
            err = np.abs(preds[m] - ref_gt).mean(axis=2)
            error_scalars[m] = err
    # Determine error vmax by 99th percentile across models
    err_vals = np.concatenate([e.flatten() for e in error_scalars.values() if e is not None]) if any(e is not None for e in error_scalars.values()) else None
    err_vmax = np.nanpercentile(err_vals, 99.0) if err_vals is not None and err_vals.size > 0 else None
    def error_to_rgb(err2d):
        if err2d is None:
            return None
        vmax = err_vmax if err_vmax is not None and err_vmax > 0 else float(np.max(err2d) + 1e-6)
        norm = np.clip(err2d / vmax, 0.0, 1.0)
        return colormaps.get_cmap("magma")(norm)[..., :3]

    row_err = [None, None]
    for m in model_keys:
        row_err.append(error_to_rgb(error_scalars[m]))
    tiles.append(row_err)

    out_dir = results_root / "figures"
    out_path = out_dir / f"compare_best_swinhgt_idx_{best_idx}.png"
    titles = columns

    # Build figure with shared colorbar if we have bounds
    ncols = len(columns)
    # We will produce two rows: predictions and errors. Leftmost two columns of error row are empty placeholders
    fig = plt.figure(figsize=(16, 6.0))
    gs = fig.add_gridspec(2, ncols + 2, width_ratios=[1]*ncols + [0.04, 0.04], height_ratios=[1, 0.9], hspace=0.2, wspace=0.05)

    # Row 1: predictions with shared colorbar if available
    for c in range(ncols):
        ax = fig.add_subplot(gs[0, c])
        ax.axis("off")
        if c < len(tiles[0]) and tiles[0][c] is not None:
            ax.imshow(tiles[0][c])
        ax.set_title(titles[c], fontsize=10)
    if shared_vmin is not None and shared_vmax is not None:
        cax1 = fig.add_subplot(gs[0, -2])
        sm1 = plt.cm.ScalarMappable(norm=Normalize(vmin=shared_vmin, vmax=shared_vmax), cmap=colormaps.get_cmap("cividis"))
        sm1.set_array([])
        cb1 = fig.colorbar(sm1, cax=cax1)
        cb1.ax.set_title("Value", fontsize=9)

    # Row 2: error heatmaps with their own colorbar
    for c in range(ncols):
        ax = fig.add_subplot(gs[1, c])
        ax.axis("off")
        img = tiles[1][c] if len(tiles) > 1 else None
        if img is not None:
            ax.imshow(img)
        if c >= 2:
            # annotate simple MAE/SSIM for model columns
            model_idx = c - 2
            key = model_keys[model_idx] if 0 <= model_idx < len(model_keys) else None
            if key is not None and preds.get(key) is not None and ref_gt is not None:
                mae = float(np.mean(np.abs(preds[key] - ref_gt)))
                ssim_val = compute_ssim(preds[key], ref_gt)
                ax.set_title(f"|Err|  MAE={mae:.3f}  SSIM={ssim_val:.3f}", fontsize=8)
        else:
            ax.set_title("Error", fontsize=9)
    # error colorbar
    if err_vmax is not None and err_vmax > 0:
        cax2 = fig.add_subplot(gs[1, -1])
        sm2 = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=err_vmax), cmap=colormaps.get_cmap("magma"))
        sm2.set_array([])
        cb2 = fig.colorbar(sm2, cax=cax2)
        cb2.ax.set_title("|Err|", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


