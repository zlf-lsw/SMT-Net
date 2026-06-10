import os
import re
import argparse
import numpy as np


def find_common_indices(root_dir: str, model_dirs: list[str]) -> list[int]:
    pattern = re.compile(r"climate_test_it\d+_idx(\d+)\.npz$")
    idx_sets = []
    for sub in model_dirs:
        vis_dir = os.path.join(root_dir, "results", sub, "visualization")
        if not os.path.isdir(vis_dir):
            raise FileNotFoundError(f"Visualization dir not found: {vis_dir}")
        indices = set()
        for name in os.listdir(vis_dir):
            m = pattern.search(name)
            if m:
                indices.add(int(m.group(1)))
        if not indices:
            raise RuntimeError(f"No npz files in {vis_dir}")
        idx_sets.append(indices)
    common = set.intersection(*idx_sets) if idx_sets else set()
    return sorted(list(common))


def _to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, copy=False)
    if vmax - vmin < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    x = (arr - vmin) / (vmax - vmin)
    return (x.clip(0, 1) * 255.0).astype(np.uint8)


def _ensure_gray2d(x: np.ndarray) -> np.ndarray:
    # Accept [H,W] or [H,W,1]; if [1,H,W] -> [H,W]
    if x.ndim == 3:
        if x.shape[0] == 1 and x.shape[1] > 1:
            # [1,H,W] -> [H,W]
            return x[0]
        if x.shape[-1] == 1:
            return x[..., 0]
    return x


def build_grid_image(root_dir: str, indices: list[int], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Results subdirs
    unet = "UNet_climate_infer_2019"
    swin = "SwinIR_climate_infer_2019"
    uform = "Uformer_climate_infer_2019"
    srcnn = "SRCNN_climate_infer_2019"
    models = [("UNet", unet), ("SwinIR", swin), ("Uformer", uform), ("SRCNN", srcnn)]

    # Columns: LR | GT | models...
    num_cols = 2 + len(models)
    num_rows = len(indices)

    figsize = (num_cols * 2.4, num_rows * 2.4)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    for r, idx in enumerate(indices):
        # Load per-model npz for same idx
        per_model = {}
        vmins, vmaxs = [], []
        for name, sub in models:
            path = os.path.join(
                root_dir, "results", sub, "visualization", f"climate_test_it000000_idx{idx:04d}.npz"
            )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing npz: {path}")
            d = np.load(path)
            per_model[name] = d
            # Collect vmin/vmax for unified color range per-row
            vmins.append(float(d.get("vmin", np.nanmin([d["lr"], d["gt"], d["sr"]]))))
            vmaxs.append(float(d.get("vmax", np.nanmax([d["lr"], d["gt"], d["sr"]]))))

        vmin = float(np.min(vmins))
        vmax = float(np.max(vmaxs))

        # Column 0: LR (from UNet npz)
        lr = _ensure_gray2d(per_model["UNet"]["lr"])  # [H,W]
        axes[r, 0].imshow(_to_uint8(lr, vmin, vmax), cmap="turbo")
        axes[r, 0].set_title("LR (↑)", fontsize=9)
        axes[r, 0].axis("off")

        # Column 1: GT
        gt = _ensure_gray2d(per_model["UNet"]["gt"])  # [H,W]
        axes[r, 1].imshow(_to_uint8(gt, vmin, vmax), cmap="turbo")
        axes[r, 1].set_title("GT", fontsize=9)
        axes[r, 1].axis("off")

        # Remaining columns: each model's SR
        for c, (name, _) in enumerate(models, start=2):
            sr = _ensure_gray2d(per_model[name]["sr"])  # [H,W]
            axes[r, c].imshow(_to_uint8(sr, vmin, vmax), cmap="turbo")
            axes[r, c].set_title(name, fontsize=9)
            axes[r, c].axis("off")

        # Row label on the left margin
        axes[r, 0].text(
            -0.15,
            0.5,
            f"idx={idx:04d}",
            transform=axes[r, 0].transAxes,
            va="center",
            ha="right",
            fontsize=9,
        )

    fig.suptitle("Model Comparison (LR/GT vs UNet/SwinIR/Uformer/SRCNN)", fontsize=12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="indices to visualize; default: auto-pick")
    parser.add_argument("--rows", type=int, default=6, help="auto rows if indices not provided")
    parser.add_argument("--best_swinir", action="store_true", help="select top-K (rows) indices by SwinIR sr-vs-gt lowest MSE")
    parser.add_argument("--out", type=str, default=None, help="output png path")
    args = parser.parse_args()

    model_dirs = [
        "UNet_climate_infer_2019",
        "SwinIR_climate_infer_2019",
        "Uformer_climate_infer_2019",
        "SRCNN_climate_infer_2019",
    ]

    if args.indices:
        indices = sorted(set(args.indices))
    else:
        common = find_common_indices(args.root, model_dirs)
        if not common:
            raise RuntimeError("No common indices among models' visualization outputs")
        if args.best_swinir:
            # score each index by MSE(sr, gt) on SwinIR
            swin_dir = os.path.join(args.root, "results", "SwinIR_climate_infer_2019", "visualization")
            scored = []
            for idx in common:
                p = os.path.join(swin_dir, f"climate_test_it000000_idx{idx:04d}.npz")
                if os.path.isfile(p):
                    d = np.load(p)
                    sr = np.asarray(d["sr"], dtype=np.float32)
                    gt = np.asarray(d["gt"], dtype=np.float32)
                    # ensure [H,W]
                    if sr.ndim == 3 and sr.shape[-1] == 1:
                        sr = sr[..., 0]
                    if gt.ndim == 3 and gt.shape[-1] == 1:
                        gt = gt[..., 0]
                    mse = float(np.mean((sr - gt) ** 2))
                    scored.append((mse, idx))
            scored.sort(key=lambda x: x[0])
            indices = [idx for _, idx in scored[: args.rows]]
        else:
            indices = common[: args.rows]

    out_path = args.out or os.path.join(args.root, "results", "comparison", f"comparison_{len(indices)}rows.png")
    build_grid_image(args.root, indices, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------- 发现与解析 --------------------

def list_model_visual_dirs(root: Path) -> Dict[str, Path]:
    # 默认支持的模型可视化路径
    candidates = {
        'UNet': root / 'experiments' / 'UNet_climate_baseline_7to1_64x64' / 'visualization',
        'RCAN': root / 'experiments' / 'RCAN_climate_baseline_7to1_64x64' / 'visualization',
        'SwinIR': root / 'experiments' / 'SwinIR_climate_baseline_7to1_64x64' / 'visualization',
        'Uformer': root / 'experiments' / 'Uformer_climate_baseline_7to1_64x64' / 'visualization',
        'RRDB': root / 'experiments' / 'RRDB_climate_baseline_7to1_64x64' / 'visualization',
        'HAT': root / 'experiments' / 'HAT_climate_baseline_7to1_64x64' / 'visualization',
    }
    return {name: p for name, p in candidates.items() if p.exists()}


def parse_idx_from_name(name: str) -> Optional[int]:
    m = re.search(r"_idx(\d{4})\.(png|npz)$", name)
    if not m:
        return None
    return int(m.group(1))


def parse_iter_from_name(name: str) -> Optional[int]:
    m = re.search(r"_it(\d+)\b", name)
    if not m:
        return None
    return int(m.group(1))


def find_latest_npz_for_idx(vis_dir: Path, idx: int) -> Optional[Path]:
    pattern = re.compile(rf".*_idx{idx:04d}\.npz$")
    files = [p for p in vis_dir.glob('*.npz') if pattern.search(p.name)]
    if not files:
        return None
    files.sort(key=lambda p: (parse_iter_from_name(p.name) or -1, p.stat().st_mtime), reverse=True)
    return files[0]


def collect_common_indices(model_dirs: Dict[str, Path]) -> List[int]:
    indices_per_model: List[set] = []
    for _, d in model_dirs.items():
        idxs = set()
        for p in d.glob('*.npz'):
            idx = parse_idx_from_name(p.name)
            if idx is not None:
                idxs.add(idx)
        if idxs:
            indices_per_model.append(idxs)
    if not indices_per_model:
        return []
    common = set.intersection(*indices_per_model) if len(indices_per_model) > 1 else indices_per_model[0]
    return sorted(common)


def load_fields(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    with np.load(npz_path) as data:
        lr = data['lr']
        gt = data['gt']
        sr = data['sr']
        vmin = float(data.get('vmin', np.nanmin([lr.min(), gt.min(), sr.min()])))
        vmax = float(data.get('vmax', np.nanmax([lr.max(), gt.max(), sr.max()])))
    return lr, gt, sr, vmin, vmax

# -------------------- 指标 --------------------

def calc_mae(sr: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(sr - gt)))

def calc_mse(sr: np.ndarray, gt: np.ndarray) -> float:
    diff = sr - gt
    return float(np.mean(diff * diff))

# -------------------- 画图：专业单行 --------------------

def professional_compare(
    model_dirs: Dict[str, Path],
    idx: int,
    order: List[str],
    out_dir: Path,
    cmap: str,
    dpi: int,
    figsize: Tuple[float, float],
    filetype: str,
):
    # 收集每个模型的最新npz
    entries: List[Tuple[str, Path]] = []
    for name in order:
        d = model_dirs.get(name)
        if d is None:
            continue
        p = find_latest_npz_for_idx(d, idx)
        if p is not None:
            entries.append((name, p))
    if not entries:
        raise RuntimeError('未找到任何模型的 npz，请先完成验证。')

    # 选第一项作为参考，统一LR/GT与基础色标
    ref_lr, ref_gt, _, ref_vmin, ref_vmax = load_fields(entries[0][1])
    # 同步全模型的全局vmin/vmax
    vmins = [ref_vmin]
    vmaxs = [ref_vmax]
    for _, p in entries:
        _, gt, sr, vmin, vmax = load_fields(p)
        vmins.extend([vmin, float(sr.min()), float(gt.min())])
        vmaxs.extend([vmax, float(sr.max()), float(gt.max())])
    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    # 单行布局：LR | GT | 模型1 | 模型2 | ...
    ncols = 2 + len(entries)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes[0]

    def show(ax, img, title: str):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    im0 = show(axes[0], ref_lr, 'Input (LR↑)')
    show(axes[1], ref_gt, 'GT')

    # 模型列 + 指标（PSNR不展示以简洁，此模式专注结构对比）
    for i, (name, p) in enumerate(entries, start=2):
        _, _, sr, _, _ = load_fields(p)
        show(axes[i], sr, name)

    # 右侧共享色条
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel('Value (shared scale)', rotation=90)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'pro_compare_idx{idx:04d}.{filetype}'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

# -------------------- 画图：旧模式（行=模型，列=字段） --------------------

def legacy_compare(
    model_dirs: Dict[str, Path],
    idx: int,
    which: List[str],
    out_dir: Path,
    cmap: str,
    dpi: int,
    figsize: Tuple[float, float],
    filetype: str,
):
    names = []
    lr_list, gt_list, sr_list = [], [], []
    vmins, vmaxs = [], []

    for name, d in model_dirs.items():
        p = find_latest_npz_for_idx(d, idx)
        if p is None:
            continue
        lr, gt, sr, vmin, vmax = load_fields(p)
        names.append(name)
        lr_list.append(lr)
        gt_list.append(gt)
        sr_list.append(sr)
        vmins.extend([vmin, float(lr.min()), float(gt.min()), float(sr.min())])
        vmaxs.extend([vmax, float(lr.max()), float(gt.max()), float(sr.max())])

    if not names:
        raise RuntimeError('未找到匹配的npz数据，请先完成验证以生成npz。')

    vmin = float(np.min(vmins))
    vmax = float(np.max(vmaxs))

    nrows = len(names)
    ncols = len(which)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, squeeze=False)

    for r, name in enumerate(names):
        fields = {'lr': lr_list[r], 'gt': gt_list[r], 'sr': sr_list[r]}
        for c, key in enumerate(which):
            ax = axes[r, c]
            im = ax.imshow(fields[key], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(key.upper())
        axes[r, 0].set_ylabel(name, rotation=90, fontsize=10)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel('Value (shared scale)', rotation=90)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'legacy_compare_idx{idx:04d}_{"-".join(which)}.{filetype}'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

# -------------------- 画图：面板模式（行=样本，列=Input/Pred/GT/AbsError） --------------------

def panel_compare_single_model(
    model_dirs: Dict[str, Path],
    model_name: str,
    idxs: List[int],
    out_dir: Path,
    cmap_main: str,
    cmap_err: str,
    dpi: int,
    figsize: Tuple[float, float],
    filetype: str,
    title: str,
):
    vis_dir = model_dirs.get(model_name)
    if vis_dir is None:
        raise ValueError(f'未找到模型可视化目录: {model_name}')
    # 加载数据
    records = []
    for idx in idxs:
        p = find_latest_npz_for_idx(vis_dir, idx)
        if p is None:
            continue
        lr, gt, sr, vmin, vmax = load_fields(p)
        records.append((idx, lr, gt, sr))
    if not records:
        raise RuntimeError('所选样本没有npz，请先生成验证结果。')

    # 列范围：主图(LR/Pred/GT)统一色标；误差列独立(>=0)
    main_vals = []
    err_vals = []
    for _, lr, gt, sr in records:
        main_vals.extend([lr.min(), lr.max(), gt.min(), gt.max(), sr.min(), sr.max()])
        err = np.abs(sr - gt)
        err_vals.extend([err.min(), err.max()])
    vmin = float(np.min(main_vals))
    vmax = float(np.max(main_vals))
    err_min = 0.0
    err_max = float(np.max(err_vals))

    nrows = len(records)
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, squeeze=False)

    for r, (idx, lr, gt, sr) in enumerate(records):
        err = np.abs(sr - gt)
        mae = calc_mae(sr, gt)
        mse = calc_mse(sr, gt)
        # 逐列绘制
        ims = []
        ims.append(axes[r, 0].imshow(lr, cmap=cmap_main, vmin=vmin, vmax=vmax, interpolation='nearest'))
        ims.append(axes[r, 1].imshow(sr, cmap=cmap_main, vmin=vmin, vmax=vmax, interpolation='nearest'))
        ims.append(axes[r, 2].imshow(gt, cmap=cmap_main, vmin=vmin, vmax=vmax, interpolation='nearest'))
        ims.append(axes[r, 3].imshow(err, cmap=cmap_err, vmin=err_min, vmax=err_max, interpolation='nearest'))
        # 标题与色条
        axes[r, 0].set_title(f'Sample {r+1}: Input', fontsize=10)
        axes[r, 1].set_title(f'Sample {r+1}: Prediction', fontsize=10)
        axes[r, 2].set_title(f'Sample {r+1}: Ground Truth', fontsize=10)
        axes[r, 3].set_title(f'Sample {r+1}: Absolute Error\nMSE: {mse:.4f}, MAE: {mae:.4f}', fontsize=10)
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            fig.colorbar(ims[c], ax=axes[r, c], fraction=0.046, pad=0.02)
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'panel_{model_name}_idx{"-".join([f"{i:04d}" for i in idxs])}.{filetype}'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

# -------------------- 画图：矩阵模式（行=样本，列=GT+多模型SR，可选LR） --------------------

def matrix_compare_models(
    model_dirs: Dict[str, Path],
    models: List[str],
    idxs: List[int],
    out_dir: Path,
    cmap: str,
    dpi: int,
    figsize: Tuple[float, float],
    filetype: str,
    show_ticks: bool,
    title: str,
    include_lr: bool,
):
    # 选第一个可用模型提供GT/LR
    base_model = None
    for m in models:
        if m in model_dirs:
            base_model = m
            break
    if base_model is None:
        raise ValueError('未找到任何模型的可视化目录。')

    # 收集数据与全局色标
    grids = []  # 每行：[LR?, GT, SR(model1), SR(model2), ...]
    vals = []
    for idx in idxs:
        p_ref = find_latest_npz_for_idx(model_dirs[base_model], idx)
        if p_ref is None:
            continue
        lr_ref, gt_ref, _, _, _ = load_fields(p_ref)
        row = []
        if include_lr:
            row.append(lr_ref)
            vals.extend([lr_ref.min(), lr_ref.max()])
        row.append(gt_ref)
        vals.extend([gt_ref.min(), gt_ref.max()])
        for m in models:
            d = model_dirs.get(m)
            if d is None:
                row.append(None)
                continue
            p = find_latest_npz_for_idx(d, idx)
            if p is None:
                row.append(None)
                continue
            _, _, sr, _, _ = load_fields(p)
            row.append(sr)
            vals.extend([sr.min(), sr.max()])
        grids.append((idx, row))
    if not grids:
        raise RuntimeError('所选样本没有共同npz，请先生成验证结果。')

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    nrows = len(grids)
    ncols = (1 if include_lr else 0) + 1 + len(models)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, squeeze=False)

    # 列标题
    col_titles = (['Input (LR↑)'] if include_lr else []) + ['Ground Truth'] + models
    for c in range(ncols):
        axes[0, c].set_title(col_titles[c], fontsize=11)

    for r, (idx, row) in enumerate(grids):
        for c, img in enumerate(row):
            ax = axes[r, c]
            if img is None:
                ax.axis('off')
                continue
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            if show_ticks:
                pass
            else:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[r, 0].set_ylabel(f'idx {idx:04d}', fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'matrix_{"-".join(models)}_idx{"-".join([f"{i:04d}" for i in idxs])}.{filetype}'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

# -------------------- 主程序 --------------------

def main():
    parser = argparse.ArgumentParser(description='科研风格模型对比：pro=单行 LR|GT|各模型SR；panel=多行四列；matrix=多行(样本)×多列(LR可选+GT+多模型SR)')
    parser.add_argument('--root', type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--idx', type=int, default=None, help='验证样本idx，默认取共同idx中的最大值（pro/legacy）')
    parser.add_argument('--cmap', type=str, default='turbo', help='主图色图，如 turbo/viridis/coolwarm')
    parser.add_argument('--cmap_err', type=str, default='Reds', help='误差色图')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 4], help='图尺寸 inch（pro/legacy）')
    parser.add_argument('--out', type=str, default=None, help='输出目录，默认 experiments/compare_matplot')
    parser.add_argument('--filetype', type=str, default='png', choices=['png','pdf'])
    parser.add_argument('--order', type=str, nargs='*', default=['UNet','RCAN','SwinIR','Uformer','RRDB','HAT'], help='pro模式模型显示顺序（存在者会被使用）')
    parser.add_argument('--mode', type=str, default='pro', choices=['pro','legacy','panel','matrix'], help='选择输出模式')
    parser.add_argument('--show', type=str, default='gt,sr', help='legacy模式字段')
    # panel-specific
    parser.add_argument('--panel_model', type=str, default='UNet', help='panel模式：指定模型名')
    parser.add_argument('--panel_num', type=int, default=4, help='panel模式：采样的样本数量（从共同idx尾部取）')
    parser.add_argument('--panel_figsize', type=float, nargs=2, default=[12, 9], help='panel模式图尺寸')
    parser.add_argument('--title', type=str, default='Model Prediction Results Comparison', help='图标题（panel/matrix模式）')
    # matrix-specific（默认包含RRDB与HAT）
    parser.add_argument('--matrix_models', type=str, nargs='*', default=['UNet','RCAN','SwinIR','Uformer','RRDB','HAT'], help='matrix模式：列的模型列表')
    parser.add_argument('--matrix_num', type=int, default=6, help='matrix模式：显示样本数量（从共同idx尾部取）')
    parser.add_argument('--matrix_figsize', type=float, nargs=2, default=[14, 12], help='matrix模式图尺寸')
    parser.add_argument('--ticks', action='store_true', help='matrix模式显示坐标刻度')
    parser.add_argument('--no_lr', action='store_true', help='matrix模式禁用输入(LR)列')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_dirs = list_model_visual_dirs(root)
    if not model_dirs:
        raise FileNotFoundError('未在 experiments/ 下找到可视化目录（请先运行验证生成npz）。')

    out_dir = Path(args.out).resolve() if args.out else (root / 'experiments' / 'compare_matplot')

    # 引入局部函数（避免上面重复粘贴所有函数体）
    from inspect import currentframe, getouterframes
    g = globals()
    professional_compare = g['professional_compare']
    legacy_compare = g['legacy_compare']
    panel_compare_single_model = g['panel_compare_single_model']
    matrix_compare_models = g['matrix_compare_models']

    if args.mode in ['pro', 'legacy']:
        common_idxs = collect_common_indices(model_dirs)
        if not common_idxs:
            raise RuntimeError('不同模型之间没有共同的样本idx，请确保它们都生成了相同idx的npz。')
        target_idx = args.idx if args.idx is not None else common_idxs[-1]
        if args.mode == 'pro':
            professional_compare(model_dirs, target_idx, args.order, out_dir, args.cmap, args.dpi, tuple(args.figsize), args.filetype)
        else:
            which = [s.strip().lower() for s in args.show.split(',') if s.strip()]
            legacy_compare(model_dirs, target_idx, which, out_dir, args.cmap, args.dpi, tuple(args.figsize), args.filetype)
    elif args.mode == 'panel':
        vis_dir = model_dirs.get(args.panel_model)
        if vis_dir is None:
            raise ValueError(f'未找到模型可视化目录: {args.panel_model}')
        idxs = sorted({parse_idx_from_name(p.name) for p in vis_dir.glob("*.npz") if parse_idx_from_name(p.name) is not None})
        if not idxs:
            raise RuntimeError('该模型没有npz文件，请先运行验证。')
        chosen = idxs[-args.panel_num:]
        panel_compare_single_model(model_dirs, args.panel_model, chosen, out_dir, args.cmap, args.cmap_err, args.dpi, tuple(args.panel_figsize), args.filetype, args.title)
    else:
        # matrix 模式：行=多个样本，列=LR(可选)+GT+多模型；若交集为空则逐步剔除样本最少的模型
        # 收集各模型的idx集合
        model_to_idxs = {}
        available_models = [m for m in args.matrix_models if m in model_dirs]
        for m in available_models:
            idxs = sorted({parse_idx_from_name(p.name) for p in model_dirs[m].glob('*.npz') if parse_idx_from_name(p.name) is not None})
            if idxs:
                model_to_idxs[m] = set(idxs)
        if not model_to_idxs:
            raise RuntimeError('所选模型均无npz，请先运行验证。')
        kept = list(model_to_idxs.keys())
        common = set.intersection(*[model_to_idxs[m] for m in kept])
        dropped: List[str] = []
        while not common and len(kept) > 1:
            # 移除样本数量最少的模型
            worst = min(kept, key=lambda m: len(model_to_idxs[m]))
            kept.remove(worst)
            dropped.append(worst)
            common = set.intersection(*[model_to_idxs[m] for m in kept])
        if not common:
            raise RuntimeError('无法找到共同样本idx，请确保至少两个模型生成过npz。')
        chosen = sorted(common)[-args.matrix_num:]
        if dropped:
            print(f"[Info] 为得到共同样本，已剔除模型: {', '.join(dropped)}")
        matrix_compare_models(model_dirs, kept, chosen, out_dir, args.cmap, args.dpi, tuple(args.matrix_figsize), args.filetype, args.ticks, args.title, include_lr=(not args.no_lr))


if __name__ == '__main__':
    main()
