import numpy as np

def compute_cci(cfg, ref, inside, layers):
    COPING_INDICATORS = {
        "emergency_access": ("fct_Emergency.tif", -1),
        "health_access": ("health.tif", -1),
        "night_light": ("fctlight.tif", 1),
        "road_density": ("fct_road_density.tif", 1),
    }

    norm_layers = {}
    for name, (fn, direction) in COPING_INDICATORS.items():
        arr = layers[name]
        arr[~inside] = np.nan
        v = arr[np.isfinite(arr)]
        lo, hi = np.percentile(v, [2, 98])
        n = np.clip((arr - lo) / (hi - lo), 0, 1)
        norm_layers[name] = (1 - n) if direction < 0 else n

    num = np.zeros((ref["height"], ref["width"]), np.float32)
    den = np.zeros_like(num)
    cnt = np.zeros(num.shape, np.int8)

    for name, n in norm_layers.items():
        ok = np.isfinite(n)
        num[ok] += n[ok]
        den[ok] += 1
        cnt[ok] += 1

    cci = np.where((cnt >= 3) & inside, num / np.maximum(den, 1e-9), np.nan).astype(np.float32)
    return cci
