import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, cohen_kappa_score, confusion_matrix
)
from scipy.stats import norm

def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall (sensitivity)": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_true, pred, zero_division=0),
        "Kappa": cohen_kappa_score(y_true, pred),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }

def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out

def delong(y_true, preds):
    y_true = np.asarray(y_true)
    preds = np.atleast_2d(preds)
    pos, neg = preds[:, y_true == 1], preds[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov
