import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

def build_model_registry(cfg):
    return {
        "RF": (
            RandomForestClassifier(random_state=cfg.seed, n_jobs=cfg.n_jobs),
            {
                "n_estimators": [300, 500],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 3, 5],
                "max_features": ["sqrt", 0.5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=cfg.seed,
                n_jobs=cfg.n_jobs,
                tree_method="hist",
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.7, 1.0],
            },
        ),
        "CART": (
            DecisionTreeClassifier(random_state=cfg.seed),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [1, 5, 10],
                "ccp_alpha": [0.0, 0.005, 0.01],
            },
        ),
    }

def train_models(cfg, X, y):
    models = build_model_registry(cfg)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    fitted = {}
    best_params = {}
    inner_cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    for name, (estimator, grid) in models.items():
        gs = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner_cv, n_jobs=cfg.n_jobs, refit=True)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        fitted[name] = best
        best_params[name] = gs.best_params_
        joblib.dump(best, os.path.join(cfg.out_dir, f"model_{name}.joblib"))

    with open(os.path.join(cfg.out_dir, "best_hyperparameters.json"), "w") as f:
        json.dump({k: {kk: (vv.item() if isinstance(vv, np.generic) else vv) for kk, vv in v.items()} for k, v in best_params.items()}, f, indent=2)

    return fitted, X_tr, X_te, y_tr, y_te
