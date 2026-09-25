"""The run-and-cache layer every phase goes through.

One deep run = one `results/preds/{key}.parquet` (per-window test predictions)
plus one `results/runs/{key}.json` (metrics, hyperparameters, imbalance
statistics, provenance).  Downstream phases — DeLong, McNemar, cost curves,
figures — read those files and never re-train anything.

A run whose two files already exist is not repeated, so `python run_all.py` is
resumable and a crash costs only the run in flight.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import metrics as M
from . import train as T
from . import utils as U

VAL_PREDS = C.RESULTS / "preds_val"
VAL_PREDS.mkdir(parents=True, exist_ok=True)
C.TABLES.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
@dataclass
class RunSpec:
    """Everything that identifies a deep run."""

    arch: str
    horizon: int | str = 4
    treatment: str = "class_weight"
    seed: int = 0
    tag: str = ""                       # phase prefix, e.g. 'decompC', 'cv2017'
    feature_cols: list[int] | None = None
    hp: T.HParams | None = None
    want_attention: bool = False
    note: str = ""

    @property
    def key(self) -> str:
        stem = f"{self.arch}_{self.horizon}_{self.treatment}_{self.seed}"
        return f"{self.tag}_{stem}" if self.tag else stem


@dataclass
class DataBundle:
    """The arrays one run consumes, already restricted to a horizon/feature set."""

    Xtr: np.ndarray
    ytr: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    val_index: pd.DataFrame
    test_index: pd.DataFrame
    meta: dict = field(default_factory=dict)


def bundle(splits: dict[str, D.SplitData], horizon: int | str = 4,
           feature_cols: list[int] | None = None,
           row_mask: dict[str, np.ndarray] | None = None,
           with_indicators: bool = False, t0_only: bool = False) -> DataBundle:
    """Assemble a run's arrays from the loaded splits."""
    sel = {}
    for name in ("train", "val", "test"):
        d = splits[name]
        if row_mask and name in row_mask:
            d = d.subset(np.flatnonzero(row_mask[name]))
        sel[name] = d

    def Xof(d):
        x = d.features(feature_cols)
        if with_indicators:
            x = np.concatenate([x, d.indicators], axis=2)
        if t0_only:
            # Keep the (n, T, F) rank with T = 1 so every model and the whole
            # training loop work unchanged; only the time axis is removed.
            x = x[:, -1:, :]
        return np.ascontiguousarray(x)

    def yof(d):
        return d.all_targets() if horizon == "multi" else d.target(int(horizon))

    return DataBundle(
        Xtr=Xof(sel["train"]), ytr=yof(sel["train"]),
        Xval=Xof(sel["val"]), yval=yof(sel["val"]),
        Xte=Xof(sel["test"]), yte=yof(sel["test"]),
        val_index=sel["val"].frame()[["cik", "end_quarter"]].reset_index(drop=True),
        test_index=sel["test"].frame()[["cik", "end_quarter"]].reset_index(drop=True),
        meta={"n_train": sel["train"].n, "n_val": sel["val"].n, "n_test": sel["test"].n,
              "n_features": Xof(sel["train"]).shape[2],
              "n_steps": Xof(sel["train"]).shape[1],
              "with_indicators": with_indicators, "t0_only": t0_only},
    )


# --------------------------------------------------------------------------
def paths(key: str) -> tuple[Path, Path, Path, Path]:
    return (C.PREDS / f"{key}.parquet", VAL_PREDS / f"{key}.parquet",
            C.RUNS / f"{key}.json", C.ATTN / f"{key}.npz")


def weights_path(key: str) -> Path:
    return C.MODELS_DIR / f"{key}.pt"


def load_model(key: str, n_features: int | None = None, n_out: int = 1):
    """Rebuild a trained model from its saved weights (interpretability reads this)."""
    import torch

    from . import models as Models

    rec = load_run(key)
    n_features = n_features or rec["data"]["n_features"]
    hp = rec["hparams"]
    model = Models.build(rec["arch"], n_features=n_features, n_out=n_out,
                         n_steps=rec["data"].get("n_steps", C.WINDOW_LEN),
                         dropout=hp.get("dropout", 0.3), **hp.get("arch_kwargs", {}))
    model.load_state_dict(torch.load(weights_path(key), map_location="cpu"))
    model.eval()
    return model


def is_cached(key: str) -> bool:
    te, va, js, _ = paths(key)
    return te.exists() and va.exists() and js.exists()


def load_run(key: str) -> dict:
    _, _, js, _ = paths(key)
    return json.loads(js.read_text(encoding="utf-8"))


def load_preds(key: str, split: str = "test") -> pd.DataFrame:
    te, va, _, _ = paths(key)
    return pd.read_parquet(te if split == "test" else va)


def load_attention(key: str) -> np.ndarray | None:
    _, _, _, at = paths(key)
    if not at.exists():
        return None
    with np.load(at) as z:
        return z["attention"]


def write_config(spec: RunSpec, b: DataBundle) -> None:
    """The run's input configuration, written before it trains.

    `experiments/<key>.json` is what a run was asked to do; `results/runs/<key>.json`
    is what it did.  Keeping them apart means the matrix can be inspected (or
    re-executed) without reading outputs.
    """
    U.write_json(C.EXPERIMENTS / f"{spec.key}.json", {
        "key": spec.key, "arch": spec.arch, "horizon": spec.horizon,
        "treatment": spec.treatment, "seed": spec.seed, "tag": spec.tag,
        "note": spec.note, "feature_cols": spec.feature_cols,
        "feature_names": ([C.FEATURE_NAMES[i] for i in spec.feature_cols]
                          if spec.feature_cols else "all 29"),
        "want_attention": spec.want_attention,
        "hparams": (spec.hp or T.HParams()).as_dict(),
        "data": b.meta,
        "early_stopping": {"monitor": "validation PR-AUC",
                           "patience": (spec.hp or T.HParams()).patience},
        **U.provenance(),
    })


def run_deep(spec: RunSpec, b: DataBundle, force: bool = False,
             verbose: bool = False) -> dict:
    """Train (or reuse) one configuration and return its run record."""
    key = spec.key
    write_config(spec, b)
    if is_cached(key) and not force:
        return load_run(key)

    res = T.train_one(
        spec.arch, b.Xtr, b.ytr, b.Xval, b.yval, b.Xte, b.yte,
        treatment=spec.treatment, seed=spec.seed, hp=spec.hp, horizon=spec.horizon,
        want_attention=spec.want_attention, verbose=verbose,
    )

    multi = (spec.horizon == "multi")
    te_path, va_path, js_path, at_path = paths(key)

    def frame(index: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
        out = index.copy()
        if multi:
            for i, h in enumerate(C.HORIZONS):
                out[f"y_true_h{h}"] = y[:, i].astype(np.int8)
                out[f"p_hat_h{h}"] = p[:, i].astype(np.float32)
            out["y_true"] = y[:, -1].astype(np.int8)
            out["p_hat"] = p[:, -1].astype(np.float32)
        else:
            out["y_true"] = np.asarray(y).astype(np.int8)
            out["p_hat"] = np.asarray(p).astype(np.float32)
        return out

    import torch

    torch.save(res.state_dict, weights_path(key))
    frame(b.test_index, b.yte, res.test_pred).to_parquet(te_path, index=False)
    frame(b.val_index, b.yval, res.val_pred).to_parquet(va_path, index=False)
    if res.attention is not None:
        np.savez_compressed(at_path, attention=res.attention.astype(np.float32))

    record = {
        "key": key, "arch": spec.arch, "horizon": spec.horizon,
        "treatment": spec.treatment, "seed": spec.seed, "tag": spec.tag,
        "note": spec.note, "n_params": res.n_params,
        "epochs_run": res.epochs_run, "best_epoch": res.best_epoch,
        "best_val_pr_auc": res.best_val_pr_auc,
        "seconds": res.seconds, "device": res.device,
        "hparams": res.hparams, "imbalance_stats": res.imbalance_stats,
        "feature_cols": spec.feature_cols, "data": b.meta,
        "history": res.history, **U.provenance(),
    }
    U.write_json(js_path, record)
    return record


# --------------------------------------------------------------------------
# Scoring a cached run.  The threshold is always chosen on val and applied once
# to test (leakage rule 3).
# --------------------------------------------------------------------------
def score(key: str, horizon_col: str = "", thr_rule: str = "f1",
          fn_cost: float = 1.0, cost_ratios=C.COST_RATIOS) -> dict:
    ycol = f"y_true{horizon_col}"
    pcol = f"p_hat{horizon_col}"
    va = load_preds(key, "val")
    te = load_preds(key, "test")
    yv, pv = va[ycol].to_numpy(), va[pcol].to_numpy()
    yt, pt = te[ycol].to_numpy(), te[pcol].to_numpy()

    if thr_rule == "f1":
        thr, _ = M.best_f1_threshold(yv, pv)
    elif thr_rule == "cost":
        thr, _ = M.best_cost_threshold(yv, pv, fn_cost=fn_cost)
    else:
        thr = float(thr_rule)

    out = {"key": key, "threshold_rule": thr_rule}
    out.update({f"val_{k}": v for k, v in M.evaluate(yv, pv).items()
                if k in ("roc_auc", "pr_auc", "n_pos")})
    out.update(M.evaluate(yt, pt, thr=thr, cost_ratios=cost_ratios))
    return out


# --------------------------------------------------------------------------
# Two aggregation conventions, always reported side by side
# --------------------------------------------------------------------------
# A 5-seed family can be summarised two ways and they do not agree:
#   per_seed  — compute the metric for each seed, then average (mean +/- std)
#   ensemble  — average the five predicted probabilities, then compute once
# The ensemble is usually the better model and is what the decomposition's
# statistical tests need (they take one prediction vector); the per-seed mean is
# what shows seed sensitivity.  Reporting only one of them was what made the
# LSTM h=4 PR-AUC read as 0.039 in one table and 0.047 in another.  Every table
# now carries both, with the convention named in the column.
def ensemble_predictions(keys: list[str], split: str = "test", horizon_col: str = ""):
    """Seed-averaged probability for a family of runs, on one split."""
    frames = [load_preds(k, split) for k in keys]
    P = np.column_stack([f[f"p_hat{horizon_col}"].to_numpy() for f in frames])
    y = frames[0][f"y_true{horizon_col}"].to_numpy().astype(int)
    return y, P.mean(axis=1), frames[0]


def ensemble_score(keys: list[str], horizon_col: str = "", thr_rule: str = "f1",
                   fn_cost: float = 1.0, cost_ratios=C.COST_RATIOS) -> dict:
    """Score the seed-averaged prediction, threshold still chosen on val."""
    yv, pv, _ = ensemble_predictions(keys, "val", horizon_col)
    yt, pt, _ = ensemble_predictions(keys, "test", horizon_col)
    if thr_rule == "f1":
        thr, _ = M.best_f1_threshold(yv, pv)
    elif thr_rule == "cost":
        thr, _ = M.best_cost_threshold(yv, pv, fn_cost=fn_cost)
    else:
        thr = float(thr_rule)
    out = {"n_seeds_in_ensemble": len(keys), "aggregation": "ensemble"}
    out.update({f"val_{k}": v for k, v in M.evaluate(yv, pv).items()
                if k in ("roc_auc", "pr_auc")})
    out.update(M.evaluate(yt, pt, thr=thr, cost_ratios=cost_ratios))
    return out


def summarise(records: list[dict], scores: list[dict], by: list[str],
              metrics: tuple[str, ...] = ("roc_auc", "pr_auc", "f1", "recall",
                                          "specificity", "precision", "accuracy"),
              horizon_col: str = "") -> pd.DataFrame:
    """Per-seed mean +/- std **and** the seed-ensemble metric, in named columns.

    `<metric>_mean` / `<metric>_std` are the per-seed convention;
    `<metric>_ensemble` is the metric of the seed-averaged prediction.  See the
    note above for why both are always present.
    """
    df = pd.DataFrame(scores)
    meta = pd.DataFrame(records)[["key", *[c for c in by if c in pd.DataFrame(records).columns],
                                  "n_params", "epochs_run", "seconds"]]
    df = df.merge(meta, on="key", how="left")
    agg = {}
    for m in metrics:
        if m in df.columns:
            agg[f"{m}_mean"] = (m, "mean")
            agg[f"{m}_std"] = (m, "std")
    for c in df.columns:
        if c.startswith("cost_1to"):
            agg[f"{c}_mean"] = (c, "mean")
            agg[f"{c}_std"] = (c, "std")
    agg["n_seeds"] = ("key", "count")
    agg["n_params"] = ("n_params", "first")
    agg["epochs_mean"] = ("epochs_run", "mean")
    agg["seconds_total"] = ("seconds", "sum")
    out = df.groupby(by, dropna=False).agg(**agg).reset_index()

    # The ensemble convention, computed from the same cached predictions.
    ens_rows = []
    for gvals, g in df.groupby(by, dropna=False):
        keys = sorted(g["key"].tolist())
        row = dict(zip(by, gvals if isinstance(gvals, tuple) else (gvals,)))
        try:
            es = ensemble_score(keys, horizon_col=horizon_col)
        except Exception as exc:                       # a family missing a file
            row["ensemble_error"] = str(exc)
            ens_rows.append(row)
            continue
        for m in metrics:
            if m in es:
                row[f"{m}_ensemble"] = es[m]
        for c in es:
            if c.startswith("cost_1to"):
                row[f"{c}_ensemble"] = es[c]
        ens_rows.append(row)
    out = out.merge(pd.DataFrame(ens_rows), on=by, how="left")
    out["aggregation_note"] = ("_mean/_std = per-seed metrics averaged; "
                               "_ensemble = metric of the seed-averaged prediction")
    return out
