import os
import re
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

try:
    import torch
except Exception:
    torch = None


def read_image(path: Path, extract_pred=False, extract_lr=False, extract_gt=False) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    
    h, w = arr.shape[:2]
    if w > h * 2:  # 宽度明显大于高度，可能是三联图 (LR|GT|Pred)
        part_w = w // 3
        if extract_lr:
            # 提取左侧1/3部分 (LR)
            arr = arr[:, :part_w]
        elif extract_gt:
            # 提取中间1/3部分 (GT)
            arr = arr[:, part_w:2*part_w]
        elif extract_pred:
            # 提取右侧1/3部分 (Pred)
            arr = arr[:, -part_w:]
    
    return arr


def resize_to(a: np.ndarray, size_hw) -> np.ndarray:
    h, w = size_hw
    if a.shape[0] == h and a.shape[1] == w:
        return a
    img = Image.fromarray((a * 255.0).clip(0, 255).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def find_candidate_pairs(vis_dir: Path, extract_pred_only=False):
    all_files = [p for p in vis_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    def tag(name: str):
        n = name.lower()
        if any(k in n for k in ["pred", "sr", "output", "out"]):
            return "pred"
        if any(k in n for k in ["gt", "target", "hr", "groundtruth"]):
            return "gt"
        if any(k in n for k in ["lq", "input", "lr", "bicubic"]):
            return "lq"
        return "pred"

    def extract_index(name: str) -> str:
        m = re.search(r"idx[_-]?(\d+)", name)
        if m:
            return m.group(1)
        m = re.search(r"(\d{4,})", name)
        if m:
            return m.group(1)
        return Path(name).stem

    buckets = {}
    for p in all_files:
        role = tag(p.name)
        idx = extract_index(p.name)
        buckets.setdefault(idx, {}).setdefault(role, []).append(p)

    pairs = {}
    for idx, roles in buckets.items():
        pred = roles.get("pred", [None])[0]
        gt = roles.get("gt", [None])[0]
        lq = roles.get("lq", [None])[0]
        if pred is not None:
            if extract_pred_only:
                # 如果是三联图，提取右侧预测部分
                pairs[idx] = {"pred": pred, "gt": gt, "lq": lq, "extract_pred": True}
            else:
                pairs[idx] = {"pred": pred, "gt": gt, "lq": lq, "extract_pred": False}
    return pairs


def scalar_to_rgb(arr: np.ndarray, vmin: float, vmax: float, cmap_name: str = "cividis") -> np.ndarray:
    cm = colormaps.get_cmap(cmap_name)
    norm = (arr - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    rgb = cm(norm)[..., :3]
    return rgb.astype(np.float32)


def load_lr_gt_scalar(idx_int: int, dataset_dir: Path):
    if torch is None:
        return None, None
    pt = dataset_dir / "dict_s2s_test.pt"
    if not pt.exists():
        return None, None
    data = torch.load(str(pt), map_location="cpu")
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
    lr = np.array(lr)
    hr = np.array(hr)
    if lr.ndim == 4 and lr.shape[0] in (1, 3):
        lr = np.transpose(lr, (1, 0, 2, 3))
    if hr.ndim == 4 and hr.shape[0] in (1, 3):
        hr = np.transpose(hr, (1, 0, 2, 3))
    n = min(lr.shape[0], hr.shape[0])
    idx = idx_int % n
    return lr[idx, 0], hr[idx, 0]


def main():
    parser = argparse.ArgumentParser(description="Multi-sample, multi-model comparison grid")
    parser.add_argument("--indices", type=str, default="", help="comma list of indices or keys; if empty, auto-pick common keys")
    parser.add_argument("--rows", type=int, default=6, help="number of rows when auto-picking keys")
    parser.add_argument("--out", type=str, default="multi_compare_auto.png")
    parser.add_argument("--search_roots", type=str, default="", help="optional extra search roots separated by |; defaults include results and experiments")
    parser.add_argument("--sources", type=str, default="results", help="which sources to use: results|experiments|both")
    parser.add_argument("--only_models", action="store_true", help="show only model predictions (hide LR/GT)")
    parser.add_argument("--lr_gt_left", action="store_true", help="show LR/GT on left, models on right")
    parser.add_argument("--extract_pred", action="store_true", help="extract prediction part from triple images (LR|GT|Pred)")
    parser.add_argument("--use_model_lr_gt", action="store_true", help="use LR/GT from model output images instead of raw data")
    args = parser.parse_args()

    results_root = Path(os.environ.get(
        "RESULTS_ROOT",
        "/home/lyb/diffusion/main2 copy/TransfomerDownscaling-main/results",
    ))
    dataset_dir = Path(os.environ.get("DATAROOT_GT", "/home/lyb/diffusion/main2 copy/dataset"))

    # Discover available model visualization dirs (include only existing)
    candidate_dirs = []
    # fixed order for columns (only these four will be shown)
    ordered_names = ["SwinIR-VMamba-HGT", "SwinIR", "UNet", "SRCNN"]
    allowed_models = set(ordered_names)
    add_map = {
        "SRCNN": results_root / "SRCNN_climate_infer_2019" / "visualization",
        "UNet": results_root / "UNet_climate_infer_2019" / "visualization",
        "SwinIR": results_root / "SwinIR_climate_infer_2019" / "visualization",
        "SwinIR-VMamba-HGT": results_root / "SwinIR_VMamba_hgt_infer_20000" / "visualization",
    }
    if args.sources in ("results", "both"):
        for name in ordered_names:
            candidate_dirs.append((name, add_map[name]))

    # Also scan experiments/*/visualization for training-time outputs
    if args.sources in ("experiments", "both"):
        experiments_root = Path(os.environ.get("EXPER_ROOT", "/home/lyb/diffusion/main2 copy/TransfomerDownscaling-main/experiments"))
        if experiments_root.exists():
            for vis_dir in experiments_root.glob("*/visualization"):
                parent_name = vis_dir.parent.name
                name = None
                s_lower = parent_name.lower()
                if "srcnn" in s_lower:
                    name = "SRCNN"
                elif "unet" in s_lower:
                    name = "UNet"
                elif ("vmamba" in s_lower) and ("swinir" in s_lower):
                    name = "SwinIR-VMamba-HGT"
                elif "swinir" in s_lower and ("hgt" not in s_lower):
                    name = "SwinIR"
                # keep only allowed models
                if name is not None and name in allowed_models:
                    candidate_dirs.append((name, vis_dir))

    # Extra user-provided search roots
    if args.search_roots:
        for root in args.search_roots.split("|"):
            pr = Path(root.strip())
            if pr.exists():
                for vis_dir in pr.glob("*/visualization"):
                    parent_name = vis_dir.parent.name
                    s_lower = parent_name.lower()
                    name = None
                    if "srcnn" in s_lower:
                        name = "SRCNN"
                    elif "unet" in s_lower:
                        name = "UNet"
                    elif ("vmamba" in s_lower) and ("swinir" in s_lower):
                        name = "SwinIR-VMamba-HGT"
                    elif "swinir" in s_lower and ("hgt" not in s_lower):
                        name = "SwinIR"
                    if name is not None and name in allowed_models:
                        candidate_dirs.append((name, vis_dir))
    model_dirs = [(name, p) for name, p in candidate_dirs if p.exists()]

    # Map index->images for each model
    pairs_by_model = {}
    for name, p in model_dirs:
        pairs_by_model[name] = find_candidate_pairs(p, extract_pred_only=args.extract_pred)

    # Keep only models that actually have keys
    non_empty_models = [(name, p) for name, p in model_dirs if len(pairs_by_model.get(name, {})) > 0]
    if non_empty_models:
        model_dirs = non_empty_models

    # Prepare indices or auto-pick common keys
    indices = [s.strip() for s in args.indices.split(",") if s.strip()]

    # Build grid data: rows per index; columns: LR, GT, then models in fixed order
    if args.only_models:
        col_names = [name for name, _ in model_dirs]
    elif args.lr_gt_left:
        col_names = ["Input (LR↑)", "Ground Truth"] + [name for name, _ in model_dirs]
    else:
        col_names = ["Input (LR↑)", "Ground Truth"] + [name for name, _ in model_dirs]

    # Precompute size using GT scalar size if available
    # fallback target size 64x64
    target_size = (64, 64)
    try:
        lr0, gt0 = load_lr_gt_scalar(int(indices[0]), dataset_dir)
        if gt0 is not None:
            target_size = gt0.shape
    except Exception:
        pass

    rows_images = []
    row_labels = []
    # If no indices specified, get common keys across discovered models
    if not indices:
        model_names = [name for name, _ in model_dirs]
        if model_names:
            # intersect across models that have keys; if empty, fall back to union
            common = None
            for name in model_names:
                keys = set(pairs_by_model[name].keys())
                if not keys:
                    continue
                common = keys if common is None else (common & keys)
            if common and len(common) > 0:
                indices = sorted(list(common))[: args.rows]
            else:
                union = set()
                for name in model_names:
                    union |= set(pairs_by_model[name].keys())
                indices = sorted(list(union))[: args.rows]

    for idx in indices:
        # Get scalar LR/GT and colorize with per-row vmin/vmax from GT
        # Convert numeric index if possible; otherwise, skip LR/GT (only show predictions)
        if str(idx).isdigit():
            lr2d, gt2d = load_lr_gt_scalar(int(idx), dataset_dir)
        else:
            lr2d, gt2d = None, None
        if gt2d is not None:
            vmin, vmax = np.nanpercentile(gt2d, [1.0, 99.0])
        else:
            vmin, vmax = None, None
        lr_rgb = scalar_to_rgb(lr2d, vmin, vmax) if lr2d is not None else None
        gt_rgb = scalar_to_rgb(gt2d, vmin, vmax) if gt2d is not None else None
        if lr_rgb is not None:
            lr_rgb = resize_to(lr_rgb, target_size)
        if gt_rgb is not None:
            gt_rgb = resize_to(gt_rgb, target_size)

        row = []
        if not args.only_models:
            if args.use_model_lr_gt and model_dirs:
                # 从第一个可用模型的输出图片中提取LR和GT
                reference_model = model_dirs[0][0]
                trio = pairs_by_model[reference_model].get(idx, None)
                if trio and trio.get("pred") and Path(trio["pred"]).exists():
                    lr_from_model = read_image(trio["pred"], extract_lr=True)
                    gt_from_model = read_image(trio["pred"], extract_gt=True)
                    lr_from_model = resize_to(lr_from_model, target_size)
                    gt_from_model = resize_to(gt_from_model, target_size)
                    row = [lr_from_model, gt_from_model]
                else:
                    row = [None, None]
            else:
                row = [lr_rgb, gt_rgb]
        for name, _ in model_dirs:
            trio = pairs_by_model[name].get(idx, None)
            if trio and trio.get("pred") and Path(trio["pred"]).exists():
                extract_flag = trio.get("extract_pred", False)
                img = read_image(trio["pred"], extract_pred=extract_flag)
                img = resize_to(img, target_size)
            else:
                img = None
            row.append(img)
        rows_images.append(row)
        try:
            row_labels.append(f"idx {int(idx):04d}")
        except Exception:
            row_labels.append(str(idx))

    # Figure: rows x columns grid
    nrows = len(rows_images)
    ncols = len(col_names)
    fig_w = 2.0 * ncols
    fig_h = 2.0 * nrows
    if nrows == 0:
        raise SystemExit("No rows to plot: no overlapping indices found. Try --indices or adjust search paths.")
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            img = rows_images[r][c]
            if img is not None:
                ax.imshow(img)
            if r == 0:
                ax.set_title(col_names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)

    plt.tight_layout(pad=0.6)
    out_dir = Path(os.environ.get("FIG_ROOT", results_root / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


