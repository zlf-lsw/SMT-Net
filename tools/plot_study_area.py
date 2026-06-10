import argparse
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.colors import LightSource


def load_dem_from_nc(nc_path: str, bbox=None):
    ds = xr.open_dataset(nc_path)
    # 拿第一个变量作为 DEM
    var = list(ds.data_vars)[0]
    da = ds[var]

    # 统一为 2D 数组
    arr = da.values
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float32)

    # 推断经纬度坐标与范围
    lon = None
    lat = None
    for n in ["lon", "longitude", "x"]:
        if n in da.coords:
            lon = da[n].values
            break
    for n in ["lat", "latitude", "y"]:
        if n in da.coords:
            lat = da[n].values
            break

    # 裁剪到 bbox（lon_min lat_min lon_max lat_max）
    if bbox and {"lon", "longitude", "x"}.intersection(da.coords) and {"lat", "latitude", "y"}.intersection(da.coords):
        lon_min, lat_min, lon_max, lat_max = bbox
        lon_name = next((n for n in ["lon", "longitude", "x"] if n in da.coords), None)
        lat_name = next((n for n in ["lat", "latitude", "y"] if n in da.coords), None)
        if lon_name and lat_name:
            lon_vals = da[lon_name]
            lat_vals = da[lat_name]
            # 处理升降序
            lon_slice = slice(lon_min, lon_max) if lon_vals[0] <= lon_vals[-1] else slice(lon_max, lon_min)
            lat_slice = slice(lat_min, lat_max) if lat_vals[0] <= lat_vals[-1] else slice(lat_max, lat_min)
            da = da.sel({lon_name: lon_slice, lat_name: lat_slice})
            arr = da.values
            if arr.ndim == 3:
                arr = arr[0]
            arr = arr.astype(np.float32)

    if lon is not None and lat is not None and lon.ndim == 1 and lat.ndim == 1:
        extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    else:
        # 无坐标时用像素坐标
        extent = [0, arr.shape[1], 0, arr.shape[0]]
    return arr, extent


def load_dem_from_tif(tif_path: str, bbox=None):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(tif_path) as ds:
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            win = from_bounds(lon_min, lat_min, lon_max, lat_max, ds.transform)
            arr = ds.read(1, window=win, out_dtype='float32')
            # extent from window
            left, top = rasterio.transform.xy(ds.transform, win.row_off, win.col_off, offset='ul')
            right, bottom = rasterio.transform.xy(ds.transform, win.row_off+win.height, win.col_off+win.width, offset='lr')
            extent = [left, right, bottom, top]
        else:
            arr = ds.read(1).astype('float32')
            left, top = ds.bounds.left, ds.bounds.top
            right, bottom = ds.bounds.right, ds.bounds.bottom
            extent = [left, right, bottom, top]
    return arr, extent


def resample_grid(arr: np.ndarray, factor: int, order: int = 3) -> np.ndarray:
    if factor <= 1:
        return arr
    try:
        from scipy.ndimage import zoom
        return zoom(arr, zoom=factor, order=order)
    except Exception:
        # 退化为重复，上采样但无平滑
        return arr.repeat(factor, axis=0).repeat(factor, axis=1)


def draw_main(ax, dem, extent, stations=None, nearest=False, vmin=None, vmax=None,
              resample=1, hillshade=False, shade_azdeg=315, shade_altdeg=45):
    dem_plot = dem.copy()
    # NaN 处理
    nan_mask = ~np.isfinite(dem_plot)
    if np.all(nan_mask):  # 全是空
        dem_plot = np.zeros_like(dem_plot)
        nan_mask[:] = False
    # 上采样以避免“马赛克”（视觉层面）
    if resample > 1:
        dem_plot = resample_grid(dem_plot, resample, order=3)
        if np.any(nan_mask):
            nan_mask = resample_grid(nan_mask.astype(float), resample, order=0) > 0.5

    if vmin is None or vmax is None:
        vmin = float(np.nanpercentile(dem, 1))
        vmax = float(np.nanpercentile(dem, 99))

    if hillshade:
        ls = LightSource(azdeg=shade_azdeg, altdeg=shade_altdeg)
        rgb = ls.shade(dem_plot, cmap=plt.get_cmap('terrain'), vmin=vmin, vmax=vmax, vert_exag=1.0, blend_mode='soft')
        im = ax.imshow(rgb, origin='upper', extent=extent,
                       interpolation='nearest' if nearest else None)
    else:
        im = ax.imshow(dem_plot, origin='upper', extent=extent, cmap='terrain',
                       interpolation='nearest' if nearest else None,
                       vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("DEM (m)")
    # 站点
    if stations is not None and len(stations) > 0:
        ax.scatter(stations[:, 0], stations[:, 1], s=9, c="deepskyblue", edgecolor="k", lw=0.3)
    # 边界框
    x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.0)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")


def draw_inset(ax_parent, extent):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax_in = inset_axes(ax_parent, width="28%", height="28%", loc="upper left", borderpad=1.1,
                           axes_class=plt.Axes, axes_kwargs=dict(projection=ccrs.PlateCarree()))
        ax_in.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4)
        ax_in.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
        ax_in.set_extent([73, 135, 18, 54])
        # 研究区框
        x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
        ax_in.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="crimson", lw=1.2)
        ax_in.set_axis_off()
    except Exception:
        # 无 cartopy 时，绘制一个简易定位图
        ax_in = inset_axes(ax_parent, width="28%", height="28%", loc="upper left", borderpad=1.1)
        ax_in.set_xlim(70, 140)
        ax_in.set_ylim(15, 55)
        x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
        ax_in.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="crimson", lw=1.2)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title("Locator", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="", help="HR_topo.nc")
    ap.add_argument("--dem_tif", default="", help="高分辨率DEM GeoTIFF（优先使用）")
    ap.add_argument("--stations", default="", help="CSV with lon,lat (optional)")
    ap.add_argument("--out", default="results/comparison/study_area_full.png")
    ap.add_argument("--bbox", default="", help="lon_min lat_min lon_max lat_max (optional)")
    ap.add_argument("--nearest", action="store_true", help="use nearest interpolation to避免模糊")
    ap.add_argument("--res", type=int, default=1, help="视觉上采样倍数(>=1)，缓解块状感")
    ap.add_argument("--hillshade", action="store_true", help="叠加山体阴影增强地形层次")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--paper_style", action="store_true", help="生成论文风格：左侧全国定位+右侧研究区，连线、图例、罗盘/比例尺简化版")
    args = ap.parse_args()

    bbox = None
    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = map(float, args.bbox.split())
        bbox = (lon_min, lat_min, lon_max, lat_max)
    if args.dem_tif:
        dem, extent = load_dem_from_tif(args.dem_tif, bbox=bbox)
    elif args.nc:
        dem, extent = load_dem_from_nc(args.nc, bbox=bbox)
    else:
        raise SystemExit("请提供 --dem_tif 或 --nc")

    stations_np = None
    if args.stations and os.path.exists(args.stations):
        import pandas as pd
        df = pd.read_csv(args.stations)
        if {"lon", "lat"}.issubset(df.columns):
            stations_np = df[["lon", "lat"]].values.astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not args.paper_style:
        fig = plt.figure(figsize=(9, 6), dpi=200)
        ax = fig.add_subplot(111)
        draw_main(ax, dem, extent, stations_np, nearest=args.nearest,
                  vmin=args.vmin, vmax=args.vmax, resample=args.res,
                  hillshade=args.hillshade)
        draw_inset(ax, extent)
        ax.set_title("Study Area (DEM background + stations)")
        fig.savefig(args.out, bbox_inches="tight")
    else:
        # 论文风格：两列布局
        fig = plt.figure(figsize=(10.5, 6.2), dpi=200)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 0.15, 1.6])
        # 左：全国定位图
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            axL = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            axL.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
            axL.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
            axL.set_extent([73, 135, 18, 54])
            # 研究区 bbox 红框
            x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
            axL.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="crimson", lw=1.4))
            axL.set_axis_off()
        except Exception:
            axL = fig.add_subplot(gs[0, 0])
            axL.set_xlim(70, 140); axL.set_ylim(15, 55)
            x0, x1, y0, y1 = extent[0], extent[1], extent[2], extent[3]
            axL.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="crimson", lw=1.4))
            axL.set_xticks([]); axL.set_yticks([])
            axL.set_title("Locator", fontsize=9)

        # 右：研究区DEM
        axR = fig.add_subplot(gs[0, 2])
        draw_main(axR, dem, extent, stations_np, nearest=True,
                  vmin=args.vmin, vmax=args.vmax, resample=max(args.res, 4),
                  hillshade=True)
        axR.set_title("")

        # 中：连接线（用 ConnectionPatch）
        con1 = ConnectionPatch(xyA=(x1, y1), coordsA=axL.transData,
                               xyB=(extent[0], extent[3]), coordsB=axR.transData,
                               color='k', lw=0.8)
        con2 = ConnectionPatch(xyA=(x0, y0), coordsA=axL.transData,
                               xyB=(extent[1], extent[2]), coordsB=axR.transData,
                               color='k', lw=0.8)
        fig.add_artist(con1); fig.add_artist(con2)

        # 简易罗盘与比例尺（右下角）
        axR.annotate('N', xy=(0.96, 0.18), xytext=(0.96, 0.25), xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='-|>', lw=1.0, color='k'), ha='center', va='bottom', fontsize=8)
        # 比例尺（近似）：按中心纬度换算
        lat_c = 0.5*(extent[2]+extent[3])
        km_per_deg = 111.32*np.cos(np.deg2rad(lat_c))
        bar_km = 200
        bar_deg = bar_km / km_per_deg if km_per_deg>1e-6 else 2
        x_bar0 = extent[0] + 0.05*(extent[1]-extent[0])
        y_bar = extent[2] + 0.06*(extent[3]-extent[2])
        axR.plot([x_bar0, x_bar0+bar_deg], [y_bar, y_bar], 'k-', lw=2)
        axR.text(x_bar0+bar_deg/2, y_bar+0.01*(extent[3]-extent[2]), f"{bar_km:.0f} km", ha='center', va='bottom', fontsize=8)

        fig.suptitle("研究区域位置（DEM 背景 + 站点）", fontsize=11)
        fig.savefig(args.out, bbox_inches='tight')

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


