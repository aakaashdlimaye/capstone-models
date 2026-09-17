"""Machine-learning baselines on the flattened 8 x 29 window.

Input is the window flattened to 232 features plus the two structural
indicators at t-0, so these models see exactly the information the deep models
see — only without the time axis being modelled explicitly.

Tuning budget is identical to the deep models: 30 random-search trials scored
on validation PR-AUC.  Nothing is ever fitted on val or test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import metrics as M
from . import utils as U

TUNING_SEED = 4321
MODELS = ("logreg", "svm_rbf", "random_forest", "xgboost", "stacking")


# --------------------------------------------------------------------------
# Search spaces
# --------------------------------------------------------------------------
def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_params(model: str, rng: np.random.Generator) -> dict:
    if model == "logreg":
        return {"C": _loguniform(rng, 1e-4, 1e2)}
    if model == "svm_rbf":
        return {"n_components": int(rng.choice([128, 256, 512])),
                "gamma": _loguniform(rng, 1e-4, 1e-1),
                "alpha": _loguniform(rng, 1e-7, 1e-3)}
    if model == "random_forest":
        return {"n_estimators": int(rng.choice([100, 200, 300])),
                "max_depth": [6, 10, 16, None][int(rng.integers(4))],
                "min_samples_leaf": int(rng.choice([1, 5, 20, 50])),
                "max_features": ["sqrt", 0.1, 0.3][int(rng.integers(3))]}
    if model == "xgboost":
        return {"n_estimators": int(rng.choice([200, 400, 600])),
                "max_depth": int(rng.choice([3, 5, 7])),
                "learning_rate": _loguniform(rng, 0.01, 0.3),
                "subsample": float(rng.choice([0.6, 0.8, 1.0])),
                "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
                "min_child_weight": float(rng.choice([1, 5, 20]))}
    raise KeyError(model)


DEFAULTS = {
    "logreg": {"C": 1.0},
    "svm_rbf": {"n_components": 256, "gamma": 1e-3, "alpha": 1e-5},
    "random_forest": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
                      "max_features": "sqrt"},
    "xgboost": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
}


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def make_estimator(model: str, params: dict, pos_weight: float, seed: int = 0, n_jobs: int = -1):
    if model == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params["C"], max_iter=1000, solver="lbfgs",
                                       class_weight="balanced", random_state=seed)),
        ])
    if model == "svm_rbf":
        # An exact RBF SVC is O(n^2) in the 86,592 training rows and is not
        # tractable here, so the RBF kernel is approximated with a Nystroem
        # feature map and a linear smoothed-hinge (modified Huber) classifier,
        # which is in the SVM loss family and emits probabilities as the spec
        # requires.  The exact SVC is used on the small external UCI sets.
        return Pipeline([
            ("scale", StandardScaler()),
            ("rbf", Nystroem(gamma=params["gamma"], n_components=params["n_components"],
                             random_state=seed)),
            ("clf", SGDClassifier(loss="modified_huber", alpha=params["alpha"],
                                  class_weight="balanced", max_iter=300, tol=1e-4,
                                  random_state=seed)),
        ])
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
            class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs)
    if model == "xgboost":
        from xgboost import XGBClassifier
        import torch

        return XGBClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            scale_pos_weight=pos_weight, tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_metric="aucpr", random_state=seed, n_jobs=n_jobs, verbosity=0)
    raise KeyError(model)


def predict_proba(est, X: np.ndarray) -> np.ndarray:
    p = est.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p


# --------------------------------------------------------------------------
# Tuning — 30 trials on val PR-AUC, same budget as the deep models
# --------------------------------------------------------------------------
def tune(model: str, Xtr, ytr, Xval, yval, n_trials: int = C.TUNING_TRIALS,
         seed: int = 0, force: bool = False, path: Path | None = None) -> dict:
    path = path or (C.RESULTS / "tuning" / f"ml_{model}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    rng = np.random.default_rng(TUNING_SEED)
    cand = [dict(DEFAULTS[model])]
    seen = {json.dumps(cand[0], sort_keys=True, default=str)}
    while len(cand) < n_trials:
        p = sample_params(model, rng)
        k = json.dumps(p, sort_keys=True, default=str)
        if k in seen:
            continue
        seen.add(k)
        cand.append(p)

    rows, best = [], None
    for i, p in enumerate(cand):
        t0 = time.perf_counter()
        est = make_estimator(model, p, pos_weight, seed=seed)
        est.fit(Xtr, ytr)
        vpr = M.pr_auc(yval, predict_proba(est, Xval))
        row = {"trial": i, "model": model, **p, "val_pr_auc": vpr,
               "seconds": time.perf_counter() - t0, "is_default": i == 0}
        rows.append(row)
        if best is None or (np.isfinite(vpr) and vpr > best["val_pr_auc"]):
            best = row
        print(f"  [tune {model}] trial {i + 1}/{n_trials} -> val PR-AUC {vpr:.4f} "
              f"({row['seconds']:.0f}s)", flush=True)

    out = {"model": model, "n_trials": n_trials,
           "best": {k: best[k] for k in DEFAULTS[model]},
           "best_val_pr_auc": best["val_pr_auc"],
           "default_val_pr_auc": rows[0]["val_pr_auc"],
           "trials": rows, **U.provenance()}
    U.write_json(path, out)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    return out


# --------------------------------------------------------------------------
# Stacking — out-of-fold train predictions, meta-learner fitted on train only
# --------------------------------------------------------------------------
def fit_stacking(base_params: dict[str, dict], Xtr, ytr, seed: int = 0,
                 n_folds: int = 3) -> tuple[dict, LogisticRegression, list[str]]:
    """Level-0 models refit on full train, level-1 LR on out-of-fold columns.

    The meta-learner never sees val or test: its training matrix is built from
    cross-validated predictions computed inside the train split.
    """
    names = [m for m in ("logreg", "svm_rbf", "random_forest", "xgboost") if m in base_params]
    pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    oof = np.zeros((len(ytr), len(names)), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(Xtr, ytr):
        for j, m in enumerate(names):
            est = make_estimator(m, base_params[m], pos_weight, seed=seed)
            est.fit(Xtr[tr_i], ytr[tr_i])
            oof[te_i, j] = predict_proba(est, Xtr[te_i])
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    meta.fit(oof, ytr)
    fitted = {m: make_estimator(m, base_params[m], pos_weight, seed=seed).fit(Xtr, ytr)
              for m in names}
    return fitted, meta, names


def stacking_predict(fitted: dict, meta, names: list[str], X: np.ndarray) -> np.ndarray:
    Z = np.column_stack([predict_proba(fitted[m], X) for m in names])
    return meta.predict_proba(Z)[:, 1]
