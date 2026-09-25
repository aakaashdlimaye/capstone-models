"""Cached tabular models on arbitrary feature matrices.

The decomposition's new rungs and Phase H's controls both need the same thing:
fit a tuned tabular learner to some slice of the tensor, on the row-set
contract, and leave its predictions on disk in the same layout every other model
uses.  That is all this module does.

Predictions land in `results/preds/{name}_{h}.parquet` and
`results/preds_val/{name}_{h}.parquet`, so the leakage audit sees them, the
cluster bootstrap can read them, and nothing downstream needs to know whether a
score came from a neural network or a gradient-boosted tree.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import experiment as E
from . import metrics as M
from . import ml_baselines as ML
from . import utils as U


def slice_features(splits, split: str, feature_cols: list[int] | None = None,
                   t0_only: bool = False, with_indicators: bool = False) -> np.ndarray:
    """A 2-D design matrix cut out of the (n, 8, 29) tensor."""
    d = splits[split]
    x = d.features(feature_cols)
    if with_indicators:
        x = np.concatenate([x, d.indicators], axis=2)
    if t0_only:
        x = x[:, -1, :]
    else:
        x = x.reshape(len(x), -1)
    return np.ascontiguousarray(x.astype(np.float32))


def feature_labels(feature_cols: list[int] | None, t0_only: bool,
                   with_indicators: bool) -> list[str]:
    names = [C.FEATURE_NAMES[i] for i in (feature_cols or range(C.N_FEATURES))]
    if with_indicators:
        names = names + list(C.INDICATOR_NAMES)
    if t0_only:
        return [f"{n}@t-0" for n in names]
    return [f"{n}@t-{C.WINDOW_LEN - 1 - t}" for t in range(C.WINDOW_LEN) for n in names]


def is_cached(name: str, h: int) -> bool:
    return ((C.PREDS / f"{name}_{h}.parquet").exists()
            and (E.VAL_PREDS / f"{name}_{h}.parquet").exists())


def seed_key(name: str, h: int, seed: int) -> str:
    return f"{name}_s{seed}_{h}"


def run_tabular(name: str, h: int, splits, model: str = "xgboost",
                feature_cols: list[int] | None = None, t0_only: bool = False,
                with_indicators: bool = False, n_trials: int = C.TUNING_TRIALS,
                seeds=C.SEEDS, force: bool = False, note: str = "") -> dict:
    """Tune on val, fit one model per seed, and cache both the per-seed and the
    seed-averaged predictions.

    The tabular learners are run over the same five seeds as the neural models.
    XGBoost and the random forest are stochastic (subsampling, feature sampling,
    bootstrap), so a single fit is one draw; comparing one draw against a
    five-seed ensemble biases the comparison against the tabular model.  The
    deterministic learners simply return five identical fits, and their reported
    standard deviation is zero, which is the honest answer rather than a special
    case.

    `{name}_{h}` is the seed-averaged prediction, so every existing consumer of
    that key now reads an ensemble; `{name}_s{seed}_{h}` holds each draw.
    """
    Xtr = slice_features(splits, "train", feature_cols, t0_only, with_indicators)
    Xva = slice_features(splits, "val", feature_cols, t0_only, with_indicators)
    Xte = slice_features(splits, "test", feature_cols, t0_only, with_indicators)
    ytr = splits["train"].target(h).astype(int)
    yva = splits["val"].target(h).astype(int)
    yte = splits["test"].target(h).astype(int)

    seeds = tuple(seeds)
    key = f"{name}_{h}"
    cfg_path = C.RESULTS / "tuning" / f"tab_{key}.json"
    have_all = (is_cached(name, h)
                and all(is_cached(f"{name}_s{s}", h) for s in seeds)
                and cfg_path.exists())

    if have_all and not force:
        best = json.loads(cfg_path.read_text(encoding="utf-8"))["best"]
    else:
        res = ML.tune(model, Xtr, ytr, Xva, yva, n_trials=n_trials, seed=seeds[0],
                      force=force, path=cfg_path)
        best = res["best"]
        pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
        idx = {s: splits[s].frame()[["cik", "end_quarter"]].reset_index(drop=True)
               for s in ("val", "test")}
        acc = {"val": [], "test": []}
        for seed in seeds:
            est = ML.make_estimator(model, best, pos_weight, seed=seed)
            est.fit(Xtr, ytr)
            for split, y, p in (("val", yva, ML.predict_proba(est, Xva)),
                                ("test", yte, ML.predict_proba(est, Xte))):
                acc[split].append(p)
                _write(seed_key(name, h, seed), split, idx[split], y, p)
        for split, y in (("val", yva), ("test", yte)):
            _write(key, split, idx[split], y, np.mean(acc[split], axis=0))

        U.write_json(C.EXPERIMENTS / f"tab_{key}.json", {
            "key": key, "model": model, "horizon": h, "seeds": list(seeds),
            "feature_cols": feature_cols, "t0_only": t0_only,
            "with_indicators": with_indicators, "n_inputs": int(Xtr.shape[1]),
            "n_features": len(feature_labels(feature_cols, t0_only, with_indicators)),
            "n_trials": n_trials, "best": best, "note": note,
            "aggregation": "per-seed fits plus their seed-averaged ensemble",
            **U.provenance(),
        })
        print(f"[tabular] {key}: {model} on {Xtr.shape[1]} inputs, {len(seeds)} seeds "
              f"-> ensemble test PR-AUC "
              f"{M.pr_auc(yte, np.mean(acc['test'], axis=0)):.4f}", flush=True)

    # Both conventions, as everywhere else in the repository.
    per_seed = [E.score(seed_key(name, h, s)) for s in seeds]
    s = E.score(key)
    s.update({"name": name, "model": model, "horizon": h,
              "n_inputs": int(Xtr.shape[1]), "t0_only": t0_only,
              "n_seeds": len(seeds),
              "aggregation": f"ensemble of {len(seeds)} seeds", "note": note})
    for metric in ("roc_auc", "pr_auc", "f1", "recall"):
        vals = [r[metric] for r in per_seed]
        s[f"{metric}_ensemble"] = s[metric]
        s[f"{metric}_mean"] = float(np.mean(vals))
        s[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    s["aggregation_note"] = ("_mean/_std = per-seed metrics averaged; "
                             "_ensemble = metric of the seed-averaged prediction")
    return s


def _write(key: str, split: str, idx, y, p) -> None:
    out = idx.copy()
    out["y_true"] = np.asarray(y).astype(np.int8)
    out["p_hat"] = np.asarray(p).astype(np.float32)
    out.to_parquet((C.PREDS if split == "test" else E.VAL_PREDS) / f"{key}.parquet",
                   index=False)
