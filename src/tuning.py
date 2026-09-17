"""Hyperparameter search — identical budget for every architecture, val only.

30 random-search trials per architecture, scored on validation PR-AUC at h=4.
Trial 0 is always the specification's own defaults (Adam lr 1e-3, batch 32,
dropout 0.3), so the search can only improve on the spec, never silently
replace it with something unrelated.

Search trials are capped at 15 epochs with patience 5; the selected
configuration is then retrained to convergence (60 epochs, patience 10) for
every reported run.  That is the usual two-stage budget and it is logged here
so the comparison stays auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from . import train as T
from . import utils as U

SEARCH_EPOCHS = 15
SEARCH_PATIENCE = 5
TUNING_SEED = 1234
TUNE_HORIZON = 4

SPACE = {
    "lr": ("loguniform", 1e-4, 5e-3),
    "batch_size": ("choice", [32, 64, 128, 256]),
    "dropout": ("choice", [0.1, 0.2, 0.3, 0.4, 0.5]),
    "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
}

SPEC_DEFAULTS = {"lr": 1e-3, "batch_size": 32, "dropout": 0.3, "weight_decay": 0.0}


def sample(rng: np.random.Generator) -> dict:
    out = {}
    for name, spec in SPACE.items():
        kind = spec[0]
        if kind == "loguniform":
            lo, hi = spec[1], spec[2]
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            choices = spec[1]
            out[name] = choices[int(rng.integers(len(choices)))]
    return out


def trials(n: int = C.TUNING_TRIALS, seed: int = TUNING_SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [dict(SPEC_DEFAULTS)]
    seen = {json.dumps(out[0], sort_keys=True)}
    while len(out) < n:
        t = sample(rng)
        k = json.dumps(t, sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def tune(arch: str, b, n_trials: int = C.TUNING_TRIALS, force: bool = False,
         path: Path | None = None) -> dict:
    """Random search for one architecture.  Returns the winning hyperparameters."""
    path = path or (C.RESULTS / "tuning" / f"{arch}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    rows = []
    best = None
    for i, t in enumerate(trials(n_trials)):
        hp = T.HParams(max_epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE, **t)
        res = T.train_one(arch, b.Xtr, b.ytr, b.Xval, b.yval, b.Xte, b.yte,
                          treatment="class_weight", seed=TUNING_SEED, hp=hp,
                          horizon=TUNE_HORIZON)
        # The trial is judged on validation PR-AUC only; the test predictions
        # produced by the trial are discarded and never written anywhere.
        row = {"trial": i, "arch": arch, **t,
               "val_pr_auc": res.best_val_pr_auc,
               "epochs_run": res.epochs_run, "seconds": res.seconds,
               "is_spec_default": i == 0}
        rows.append(row)
        if best is None or (np.isfinite(row["val_pr_auc"]) and row["val_pr_auc"] > best["val_pr_auc"]):
            best = row
        print(f"  [tune {arch}] trial {i + 1}/{n_trials} "
              f"lr={t['lr']:.2e} bs={t['batch_size']:3d} do={t['dropout']:.1f} "
              f"wd={t['weight_decay']:.0e} -> val PR-AUC {row['val_pr_auc']:.4f} "
              f"({res.seconds:.0f}s)", flush=True)

    out = {
        "arch": arch,
        "n_trials": n_trials,
        "budget": {"search_epochs": SEARCH_EPOCHS, "search_patience": SEARCH_PATIENCE,
                   "final_epochs": C.MAX_EPOCHS, "final_patience": C.PATIENCE,
                   "selection_metric": "validation PR-AUC", "horizon": TUNE_HORIZON,
                   "treatment": "class_weight", "seed": TUNING_SEED},
        "space": {k: list(v) for k, v in SPACE.items()},
        "best": {k: best[k] for k in SPACE},
        "best_val_pr_auc": best["val_pr_auc"],
        "spec_default_val_pr_auc": rows[0]["val_pr_auc"],
        "trials": rows,
        **U.provenance(),
    }
    U.write_json(path, out)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    return out


def best_hparams(arch: str, max_epochs: int | None = None,
                 patience: int | None = None) -> T.HParams:
    """The tuned configuration, or the spec defaults if tuning has not run.

    The epoch budget is resolved at call time so a caller that lowers
    `config.MAX_EPOCHS` (the --quick wiring check) actually takes effect.
    """
    path = C.RESULTS / "tuning" / f"{arch}.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))["best"] if path.exists() else dict(SPEC_DEFAULTS)
    return T.HParams(max_epochs=max_epochs if max_epochs is not None else C.MAX_EPOCHS,
                     patience=patience if patience is not None else C.PATIENCE, **cfg)
