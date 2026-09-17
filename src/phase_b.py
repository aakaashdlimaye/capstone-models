"""Phase B — classical baselines, all scored on the row-set contract.

Two families:
  * published formulas from levels (Altman Z''/Z', Ohlson O, Zmijewski), each
    reported threshold-free *and* at its canonical cutoff;
  * machine-learning baselines on the flattened 8 x 29 window, tuned on val
    with the same 30-trial budget the deep models get.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import classical as K
from . import config as C
from . import data as D
from . import experiment as E
from . import metrics as M
from . import ml_baselines as ML
from . import utils as U

LEVELS_CACHE = C.RESULTS / "classical_levels.parquet"

FORMULAS = {
    # key -> (label, orientation) ; orientation +1 means higher = more distress
    "altman_zdp": "Altman Z'' (1995, four-variable)",
    "altman_zp": "Altman Z' (1983, five-variable)",
    "ohlson_o": "Ohlson O-score (1980)",
    "zmijewski": "Zmijewski (1984)",
}


# --------------------------------------------------------------------------
def build_levels_table(full: bool = True, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Join the panel levels onto the row set and impute with train medians."""
    notes_path = C.RESULTS / "classical_levels_notes.json"
    if LEVELS_CACHE.exists() and notes_path.exists() and not force:
        return pd.read_parquet(LEVELS_CACHE), json.loads(notes_path.read_text(encoding="utf-8"))

    lt = K.build_levels(full=full)
    rs = D.build_row_sets(full=full)
    j = K.join_to_row_set(lt.df, rs)

    train = j[j["split"] == "train"]
    med = K.fit_imputer(train)
    j, miss = K.apply_imputer(j, med)

    notes = {
        **lt.notes,
        "imputer_fitted_on": "train split rows only",
        "imputer_train_rows": int(len(train)),
        "train_medians": med,
        "missing_before_impute": miss.to_dict(orient="records"),
        "ttm_fallback_on_row_set": {
            c.replace("_ttm_fallback", ""): int(j[c].sum())
            for c in j.columns if c.endswith("_ttm_fallback")
        },
        **U.provenance(),
    }
    j.to_parquet(LEVELS_CACHE, index=False)
    U.write_json(notes_path, notes)
    U.write_table(miss, C.TABLES / "classical_missingness", floatfmt="%.5f")
    return j, notes


def formula_scores(j: pd.DataFrame) -> dict[str, np.ndarray]:
    """Distress scores, all oriented so that higher = more distressed."""
    zdp = K.altman_z_double_prime(j)
    zp = K.altman_z_prime(j)
    return {
        "altman_zdp": -zdp,     # Altman is a health score, so negate to rank distress
        "altman_zp": -zp,
        "ohlson_o": K.ohlson_o(j),
        "zmijewski": K.zmijewski(j),
        "_raw_zdp": zdp,
        "_raw_zp": zp,
    }


def save_classical_preds(key: str, j: pd.DataFrame, score: np.ndarray, h: int) -> None:
    """Write classical predictions in the same layout as a deep run."""
    for split, path in (("val", E.VAL_PREDS), ("test", C.PREDS)):
        m = (j["split"] == split).to_numpy()
        out = pd.DataFrame({
            "cik": j.loc[m, "cik"].to_numpy(),
            "end_quarter": j.loc[m, "end_quarter"].to_numpy(),
            "y_true": j.loc[m, f"y{h}"].to_numpy().astype(np.int8),
            "p_hat": np.asarray(score)[m].astype(np.float32),
        })
        out.to_parquet(path / f"{key}.parquet", index=False)


# --------------------------------------------------------------------------
def run_formulas(j: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Threshold-free and canonical-cutoff results for the four formulas."""
    sc = formula_scores(j)
    is_val = (j["split"] == "val").to_numpy()
    is_test = (j["split"] == "test").to_numpy()

    rows, cutoff_rows = [], []
    for h in C.HORIZONS:
        y = j[f"y{h}"].to_numpy().astype(int)
        for key, label in FORMULAS.items():
            s = sc[key]
            save_classical_preds(f"{key}_{h}", j, s, h)
            thr, _ = M.best_f1_threshold(y[is_val], s[is_val])
            r = {"model": key, "label": label, "family": "formula", "horizon": h,
                 "val_roc_auc": M.roc_auc(y[is_val], s[is_val]),
                 "val_pr_auc": M.pr_auc(y[is_val], s[is_val])}
            r.update(M.evaluate(y[is_test], s[is_test], thr=thr))
            rows.append(r)

        # Altman at its published cutoffs, which is how practitioners use it.
        for key, raw, cuts in (("altman_zdp", sc["_raw_zdp"], K.Z_DPRIME_CUTS),
                               ("altman_zp", sc["_raw_zp"], K.Z_PRIME_CUTS)):
            zone = K.altman_zone(raw, cuts)
            for rule, alarm in (("distress_only", zone == "distress"),
                                ("distress_or_grey", zone != "safe")):
                a = alarm[is_test]
                yt = y[is_test]
                tp = int(((a == 1) & (yt == 1)).sum()); fp = int(((a == 1) & (yt == 0)).sum())
                fn = int(((a == 0) & (yt == 1)).sum()); tn = int(((a == 0) & (yt == 0)).sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                cutoff_rows.append({
                    "model": key, "horizon": h, "rule": rule,
                    "cut_distress": cuts[0], "cut_safe": cuts[1],
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": prec, "recall": rec,
                    "specificity": tn / (tn + fp) if tn + fp else 0.0,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "accuracy": (tp + tn) / len(yt),
                    "alarm_rate": float(a.mean()),
                })
    return pd.DataFrame(rows), pd.DataFrame(cutoff_rows)


# --------------------------------------------------------------------------
def run_ml(splits: dict[str, D.SplitData], j: pd.DataFrame, horizons=C.HORIZONS,
           n_trials: int = C.TUNING_TRIALS, seed: int = 0, force: bool = False) -> pd.DataFrame:
    """Tune, fit and score the five ML baselines at each horizon."""
    Xtr = splits["train"].flat(); Xva = splits["val"].flat(); Xte = splits["test"].flat()
    idx = {s: splits[s].frame()[["cik", "end_quarter"]].reset_index(drop=True)
           for s in ("train", "val", "test")}

    rows = []
    for h in horizons:
        ytr = splits["train"].target(h).astype(int)
        yva = splits["val"].target(h).astype(int)
        yte = splits["test"].target(h).astype(int)
        pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))

        best_params = {}
        for m in ("logreg", "svm_rbf", "random_forest", "xgboost"):
            print(f"[phase B] tuning {m} at h={h}", flush=True)
            res = ML.tune(m, Xtr, ytr, Xva, yva, n_trials=n_trials, seed=seed,
                          force=force, path=C.RESULTS / "tuning" / f"ml_{m}_h{h}.json")
            best_params[m] = res["best"]

            t0 = time.perf_counter()
            est = ML.make_estimator(m, res["best"], pos_weight, seed=seed)
            est.fit(Xtr, ytr)
            pv, pt = ML.predict_proba(est, Xva), ML.predict_proba(est, Xte)
            rows.append(_score_ml(m, h, yva, pv, yte, pt, idx, time.perf_counter() - t0,
                                  res["best"], res["best_val_pr_auc"]))

        print(f"[phase B] stacking at h={h}", flush=True)
        t0 = time.perf_counter()
        fitted, meta, names = ML.fit_stacking(best_params, Xtr, ytr, seed=seed)
        pv = ML.stacking_predict(fitted, meta, names, Xva)
        pt = ML.stacking_predict(fitted, meta, names, Xte)
        rows.append(_score_ml("stacking", h, yva, pv, yte, pt, idx, time.perf_counter() - t0,
                              {"base_models": names,
                               "meta_coef": meta.coef_[0].tolist(),
                               "meta_intercept": float(meta.intercept_[0])}, float("nan")))
    return pd.DataFrame(rows)


def _score_ml(model, h, yva, pv, yte, pt, idx, seconds, params, tuned_val_pr):
    key = f"{model}_{h}"
    pd.DataFrame({"cik": idx["val"]["cik"], "end_quarter": idx["val"]["end_quarter"],
                  "y_true": yva.astype(np.int8), "p_hat": pv.astype(np.float32)}
                 ).to_parquet(E.VAL_PREDS / f"{key}.parquet", index=False)
    pd.DataFrame({"cik": idx["test"]["cik"], "end_quarter": idx["test"]["end_quarter"],
                  "y_true": yte.astype(np.int8), "p_hat": pt.astype(np.float32)}
                 ).to_parquet(C.PREDS / f"{key}.parquet", index=False)
    thr, _ = M.best_f1_threshold(yva, pv)
    r = {"model": model, "label": model, "family": "ml", "horizon": h,
         "val_roc_auc": M.roc_auc(yva, pv), "val_pr_auc": M.pr_auc(yva, pv),
         "tuned_val_pr_auc": tuned_val_pr, "seconds": seconds,
         "params": json.dumps(params, default=str)}
    r.update(M.evaluate(yte, pt, thr=thr))
    return r


# --------------------------------------------------------------------------
def main(full: bool = True, n_trials: int = C.TUNING_TRIALS, force: bool = False) -> dict:
    print("=" * 70)
    print("PHASE B — CLASSICAL AND MACHINE-LEARNING BASELINES")
    print("=" * 70)
    splits = D.load_all(full=full)
    j, notes = build_levels_table(full=full, force=force)
    print(f"[phase B] levels joined to {len(j):,} windows; "
          f"TTM x4 fallback used on {notes['ttm_fallback_on_row_set']}")

    formulas, cutoffs = run_formulas(j)
    ml = run_ml(splits, j, n_trials=n_trials, force=force)

    allrows = pd.concat([formulas, ml], ignore_index=True)
    for h in C.HORIZONS:
        sub = allrows[allrows["horizon"] == h].drop(columns=["params"], errors="ignore")
        U.write_table(sub, C.RESULTS / f"baselines_{h}")
    U.write_table(cutoffs, C.RESULTS / "altman_canonical_cutoffs", floatfmt="%.4f")
    allrows.to_csv(C.RESULTS / "baselines_all.csv", index=False)

    # Gate: Altman Z'' must beat chance and lose to the tuned ML baselines.
    gate = []
    for h in C.HORIZONS:
        z = float(allrows.query("model == 'altman_zdp' and horizon == @h")["roc_auc"].iloc[0])
        best_ml = float(allrows.query("family == 'ml' and horizon == @h")["roc_auc"].max())
        gate.append({"horizon": h, "altman_zdp_test_roc_auc": z,
                     "best_ml_test_roc_auc": best_ml,
                     "above_chance": z > 0.5, "below_best_ml": z < best_ml})
    gate_df = pd.DataFrame(gate)
    print(gate_df.to_string(index=False))
    U.write_table(gate_df, C.TABLES / "phase_b_gate")
    ok = bool(gate_df["above_chance"].all() and gate_df["below_best_ml"].all())
    print(f"[phase B] GATE {'PASS' if ok else 'FAIL'}")
    return {"gate_pass": ok, "rows": len(allrows)}
