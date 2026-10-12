import os
import json
import numpy as np
import pandas as pd
from scipy import ndimage
from rasterio.features import geometry_mask
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from .raster_ops import align_to_reference, smooth_coarse_layer, points_to_rowcol, pixel_size_m
from .io_utils import boundary_geoms

def build_reference_grid(ref_raster_path):
    import rasterio

    with rasterio.open(ref_raster_path) as r:
        ref = {
            "crs": r.crs,
            "transform": r.transform,
            "width": r.width,
            "height": r.height,
            "profile": r.profile.copy(),
        }
    return ref

def build_feature_stack(cfg, include_coping=False):
    layers = dict(cfg.FACTORS)
    if include_coping:
        layers.update(cfg.COPING_LAYERS)

    if cfg.DIST_TO_WATER_MODE == "drop":
        layers.pop("dist_to_water", None)

    ref_path = os.path.join(cfg.data_dir, cfg.factor_dir, cfg.FACTORS[cfg.REFERENCE_FACTOR])
    ref = build_reference_grid(ref_path)

    geoms = boundary_geoms(os.path.join(cfg.data_dir, cfg.boundary_file))
    inside = geometry_mask(
        geoms,
        out_shape=(ref["height"], ref["width"]),
        transform=ref["transform"],
        invert=True
    )

    stack = {}
    for name, fn in layers.items():
        arr = align_to_reference(os.path.join(cfg.data_dir, cfg.factor_dir, fn), ref)
        if name == "landcover":
            arr[arr == cfg.LANDCOVER_NODATA] = np.nan
        if name == "rainfall" and cfg.SMOOTH_RAINFALL:
            arr = smooth_coarse_layer(arr, ref, cfg.RAINFALL_CELL_DEG)
        arr[~inside] = np.nan
        stack[name] = arr

    return ref, inside, stack

def prepare_training_samples(cfg, ref, inside, stack):
    from .io_utils import load_points
    pts, skipped = load_points(os.path.join(cfg.data_dir, cfg.points_file), cfg.label_field)
    rows, cols = points_to_rowcol(pts["lon"].values, pts["lat"].values, ref)
    in_grid = (rows >= 0) & (rows < ref["height"]) & (cols >= 0) & (cols < ref["width"])
    rows_c = np.clip(rows, 0, ref["height"] - 1)
    cols_c = np.clip(cols, 0, ref["width"] - 1)

    pts["in_fct"] = in_grid & inside[rows_c, cols_c]
    for f in stack.keys():
        pts[f] = np.where(in_grid, stack[f][rows_c, cols_c], np.nan)

    pts["dup_pixel"] = pd.Series(rows_c * ref["width"] + cols_c).duplicated(keep="first").values

    feature_names = list(stack.keys())
    data = pts[pts["in_fct"] & ~pts[feature_names].isna().any(axis=1) & ~pts["dup_pixel"]].reset_index(drop=True)
    for c in cfg.CATEGORICAL:
        if c in data:
            data[c] = data[c].round().astype(int)

    X = data[feature_names].astype(float)
    y = data["label"].values
    return data, X, y

def vif_table(X):
    Z = (X - X.mean()) / X.std(ddof=0)
    out = []
    for c in Z.columns:
        y = Z[c].values
        A = np.column_stack([np.ones(len(Z)), Z.drop(columns=c).values])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - (y - A @ beta).var() / y.var() if y.var() > 0 else 0.0
        out.append({
            "factor": c,
            "R2": r2,
            "tolerance": 1 - r2,
            "VIF": np.inf if r2 >= 1 else 1 / (1 - r2),
        })
    return pd.DataFrame(out).sort_values("VIF", ascending=False)

def spatial_blocks(df, ref, block_km):
    dx, dy = pixel_size_m(ref)
    rows, cols = points_to_rowcol(df["lon"].values, df["lat"].values, ref)
    return (
        np.floor(rows * dy / (block_km * 1000)).astype(int) * 10_000
        + np.floor(cols * dx / (block_km * 1000)).astype(int)
    )
