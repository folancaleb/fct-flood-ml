import os
import numpy as np
import pandas as pd

def generate_prob_maps(cfg, fitted, ref, valid, stack, feature_names):
    prob_maps = {}
    class_breaks = {}

    for name, est in fitted.items():
        prob = np.full((ref["height"], ref["width"]), np.nan, dtype=np.float32)
        for r0 in range(0, ref["height"], cfg.PRED_CHUNK_ROWS):
            r1 = min(r0 + cfg.PRED_CHUNK_ROWS, ref["height"])
            v = valid[r0:r1]
            if not v.any():
                continue
            Xc = pd.DataFrame({f: stack[f][r0:r1][v] for f in feature_names})
            for c in cfg.CATEGORICAL:
                if c in Xc:
                    Xc[c] = Xc[c].round().astype(int)
            blk = prob[r0:r1]
            blk[v] = est.predict_proba(Xc)[:, 1]

        fin = np.isfinite(prob)
        if cfg.CLASS_METHOD == "quantile":
            inner = np.quantile(prob[fin], [0.2, 0.4, 0.6, 0.8])
        else:
            inner = np.array([0.2, 0.4, 0.6, 0.8])

        cls = np.zeros(prob.shape, dtype=np.uint8)
        cls[fin] = np.digitize(prob[fin], inner) + 1

        prob_maps[name] = (prob, cls)
        class_breaks[name] = inner.tolist()

    return prob_maps, class_breaks
