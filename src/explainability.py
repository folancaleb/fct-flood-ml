import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence

def shap_for_model(model_name, fitted, X, feature_names):
    est = fitted[model_name]
    if model_name == "XGBoost":
        booster = est.get_booster()
        dm = xgb.DMatrix(X)
        contrib = booster.predict(dm, pred_contribs=True)
        sv, base = contrib[:, :-1], float(contrib[0, -1])
        err = np.abs(contrib.sum(axis=1) - booster.predict(dm, output_margin=True)).max()
    else:
        ex = shap.TreeExplainer(est)
        sv = ex.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = float(np.atleast_1d(ex.expected_value)[-1])
        err = np.abs(sv.sum(axis=1) + base - est.predict_proba(X)[:, 1]).max()

    return pd.DataFrame(sv, columns=feature_names), base, err

def sign_changes(x, sv, n_bins=10):
    q = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(q) < 3:
        return [], None, None

    idx = np.clip(np.digitize(x, q[1:-1]), 0, len(q) - 2)
    centres = np.array([np.median(x[idx == k]) for k in range(len(q) - 1) if (idx == k).any()])
    means = np.array([sv[idx == k].mean() for k in range(len(q) - 1) if (idx == k).any()])

    out = []
    for k in range(len(means) - 1):
        if np.sign(means[k]) != np.sign(means[k + 1]) and means[k] != 0:
            x0, x1, y0, y1 = centres[k], centres[k + 1], means[k], means[k + 1]
            out.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    return out, centres, means
