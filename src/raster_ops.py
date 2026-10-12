import math
from scipy import ndimage
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform


def pixel_size_m(ref):
    t = ref["transform"]
    if ref["crs"].is_geographic:
        lat_mid = t.f + t.e * ref["height"] / 2.0
        return abs(t.a) * 111_320.0 * math.cos(math.radians(lat_mid)), abs(t.e) * 110_574.0
    return abs(t.a), abs(t.e)

def pixel_area_km2(ref):
    t = ref["transform"]
    if ref["crs"].is_geographic:
        lats = t.f + (np.arange(ref["height"]) + 0.5) * t.e
        return abs(t.a) * 111_320.0 * np.cos(np.radians(lats)) * abs(t.e) * 110_574.0 / 1e6
    return np.full(ref["height"], abs(t.a * t.e) / 1e6)

def points_to_rowcol(lon, lat, ref):
    t = ref["transform"]
    x, y = np.asarray(lon), np.asarray(lat)
    if not ref["crs"].is_geographic:
        x, y = map(np.asarray, warp_transform("EPSG:4326", ref["crs"], x.tolist(), y.tolist()))
    return (
        np.floor((y - t.f) / t.e).astype(np.int64),
        np.floor((x - t.c) / t.a).astype(np.int64),
    )

def is_same_grid(src, ref):
    return (
        src.crs == ref["crs"]
        and src.width == ref["width"]
        and src.height == ref["height"]
        and np.allclose(tuple(src.transform)[:6], tuple(ref["transform"])[:6], atol=1e-9)
    )

def align_to_reference(path, ref, pred_chunk_rows=256):
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        st = src.transform
        out = np.full((ref["height"], ref["width"]), np.nan, dtype=np.float32)
        rt = ref["transform"]

        if is_same_grid(src, ref):
            out[:] = data.astype(np.float32)
        elif src.crs == ref["crs"]:
            xs = rt.c + (np.arange(ref["width"]) + 0.5) * rt.a
            ys = rt.f + (np.arange(ref["height"]) + 0.5) * rt.e
            cols = np.floor((xs - st.c) / st.a).astype(np.int64)
            rows = np.floor((ys - st.f) / st.e).astype(np.int64)
            vc, vr = (cols >= 0) & (cols < src.width), (rows >= 0) & (rows < src.height)
            out[np.ix_(np.where(vr)[0], np.where(vc)[0])] = data[np.ix_(rows[vr], cols[vc])]
        else:
            xs = rt.c + (np.arange(ref["width"]) + 0.5) * rt.a
            for r0 in range(0, ref["height"], pred_chunk_rows):
                r1 = min(r0 + pred_chunk_rows, ref["height"])
                ys = rt.f + (np.arange(r0, r1) + 0.5) * rt.e
                X, Y = np.meshgrid(xs, ys)
                tx, ty = warp_transform(ref["crs"], src.crs, X.ravel().tolist(), Y.ravel().tolist())
                cols = np.floor((np.asarray(tx) - st.c) / st.a).astype(np.int64)
                rows = np.floor((np.asarray(ty) - st.f) / st.e).astype(np.int64)
                ok = (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height)
                blk = np.full(X.size, np.nan, dtype=np.float32)
                blk[ok] = data[rows[ok], cols[ok]]
                out[r0:r1] = blk.reshape(X.shape)

    if nodata is not None:
        out[np.isclose(out, nodata)] = np.nan
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 1e30] = np.nan
    return out

def smooth_coarse_layer(arr, ref, cell_deg, pred_chunk_rows=256):
    from scipy.ndimage import map_coordinates, distance_transform_edt
    t, H, W = ref["transform"], ref["height"], ref["width"]
    xs = t.c + (np.arange(W) + 0.5) * t.a
    ys = t.f + (np.arange(H) + 0.5) * t.e

    x0 = math.floor(xs.min() / cell_deg) * cell_deg
    y0 = math.ceil(ys.max() / cell_deg) * cell_deg
    ci = np.floor((xs - x0) / cell_deg).astype(int)
    ri = np.floor((y0 - ys) / cell_deg).astype(int)
    nr, nc = ri.max() + 1, ci.max() + 1

    coarse = np.full((nr, nc), np.nan)
    for r in range(nr):
        rows_r = np.where(ri == r)[0]
        if rows_r.size == 0:
            continue
        band = arr[rows_r]
        for c in range(nc):
            v = band[:, ci == c]
            v = v[np.isfinite(v)]
            if v.size:
                coarse[r, c] = np.median(v)

    miss = ~np.isfinite(coarse)
    if miss.any():
        idx = distance_transform_edt(miss, return_distances=False, return_indices=True)
        coarse = coarse[tuple(idx)]

    fc = (xs - x0) / cell_deg - 0.5
    out = np.empty((H, W), dtype=np.float32)
    for r0 in range(0, H, pred_chunk_rows):
        r1 = min(r0 + pred_chunk_rows, H)
        fr = (y0 - ys[r0:r1]) / cell_deg - 0.5
        FR, FC = np.meshgrid(fr, fc, indexing="ij")
        out[r0:r1] = map_coordinates(
            coarse,
            [FR.ravel(), FC.ravel()],
            order=1,
            mode="nearest",
        ).reshape(FR.shape)

    out[~np.isfinite(arr)] = np.nan
    return out
