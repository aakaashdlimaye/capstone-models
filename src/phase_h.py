"""Phase H — the controls the architecture comparison was missing, and the
metrics a practitioner would actually ask for.

Six questions:

  H1  Does the eight-quarter window help the *winning* model, or only the LSTM?
      XGBoost at t-0 against XGBoost on the flattened window, same tuning budget.
  H2  Is the LSTM's gain over logistic regression the time axis, or just the
      nonlinearity?  A static MLP on the same 29 ratios at t-0 separates them.
  H3  Are the architecture differences distinguishable from noise?  Pairwise
      firm-clustered intervals, with DeLong and McNemar beside them.
  H4  Of the k riskiest firms, how many actually fail?  Precision@k and capture
      curves, at window level and at firm level.
  H5  Are the scores calibrated?  Reliability curves, Brier, and ECE.
  H6  Does any of this hold for small firms, or only for large ones?
      Total-assets terciles, and sectors with enough positives to say anything.

Nothing here retunes or rescores an existing model: H3 to H6 read cached
prediction files.  H1 and H2 add new models under the usual run-and-cache rules,
tuning on validation only and scoring test once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import experiment as E
from . import metrics as M
from . import tabular as TAB
from . import tuning as Tune
from . import utils as U

N_BOOT = 2000
MLP_29_TUNING = C.RESULTS / "tuning" / "mlp_t0_all29.json"

# The four models the practitioner-facing sections report on: the field
# standard, the strongest baseline, the best neural model, and the
# five-ratio temporal model the decomposition singles out.
REPORT_MODELS = ("altman_zdp", "xgboost", "transformer", "C_lstm")


# --------------------------------------------------------------------------
def model_predictions(name: str, h: int, seeds=C.SEEDS) -> dict | None:
    """Test and validation predictions for one named model, plus its firm ids."""
    single = {"altman_zdp": f"altman_zdp_{h}", "altman_zp": f"altman_zp_{h}",
              "ohlson_o": f"ohlson_o_{h}", "zmijewski": f"zmijewski_{h}",
              "xgboost": f"xgboost_{h}", "random_forest": f"random_forest_{h}",
              "logreg": f"logreg_{h}", "stacking": f"stacking_{h}",
              "xgb_t0_29": f"xgb_t0_29_{h}"}
    ens = {"transformer": ("transformer", ""), "lstm": ("lstm", ""),
           "bilstm": ("bilstm", ""), "cnn_lstm_attn": ("cnn_lstm_attn", ""),
           "C_lstm": ("lstm", "decompC"), "mlp_t0_29": ("mlp", "h2mlp")}

    if name in single:
        key = single[name]
        if not (C.PREDS / f"{key}.parquet").exists():
            return None
        te, va = E.load_preds(key, "test"), E.load_preds(key, "val")
        return {"name": name, "aggregation": "single fit",
                "y": te["y_true"].to_numpy().astype(int), "p": te["p_hat"].to_numpy(),
                "yv": va["y_true"].to_numpy().astype(int), "pv": va["p_hat"].to_numpy(),
                "groups": te["cik"].astype(str).to_numpy()}

    arch, tag = ens[name]
    keys = [E.RunSpec(arch=arch, horizon=h, treatment="class_weight", seed=s, tag=tag).key
            for s in seeds]
    if not all((C.PREDS / f"{k}.parquet").exists() for k in keys):
        return None
    yt, pt, base = E.ensemble_predictions(keys, "test")
    yv, pv, _ = E.ensemble_predictions(keys, "val")
    return {"name": name, "aggregation": f"ensemble of {len(keys)} seeds",
            "y": yt, "p": pt, "yv": yv, "pv": pv,
            "groups": base["cik"].astype(str).to_numpy()}


# --------------------------------------------------------------------------
# H1 / H2 — the two missing controls
# --------------------------------------------------------------------------
# r21..r24 are year-on-year growth ratios, so a single t-0 row already encodes a
# four-quarter comparison.  If that is why the eight-quarter window buys the tree
# nothing, removing them should hurt the t-0 model more than the window model and
# open up the gap between them.
GROWTH_IDX = [20, 21, 22, 23]                       # r21..r24
NON_GROWTH_IDX = [i for i in range(C.N_FEATURES) if i not in GROWTH_IDX]


def h1_temporal_control(splits, horizons=C.HORIZONS, n_trials: int = C.TUNING_TRIALS,
                        force: bool = False) -> pd.DataFrame:
    """Does the eight-quarter window buy the best model anything?

    The window model is the existing Phase B XGBoost (232 ratio inputs plus the
    two structural indicators).  The control is the same learner, same budget,
    on the same features at the end quarter only (29 + 2).  They differ in the
    time axis and nothing else.
    """
    rows = []
    for h in horizons:
        TAB.run_tabular("xgb_t0_29", h, splits, model="xgboost", t0_only=True,
                        with_indicators=True, n_trials=n_trials, force=force,
                        note="29 ratios at t-0 plus the two indicators; no time axis")
        win = model_predictions("xgboost", h)
        t0 = model_predictions("xgb_t0_29", h)
        if win is None or t0 is None:
            continue
        cmp = M.compare(win["y"], t0["p"], win["p"], win["groups"], n_boot=N_BOOT)
        rows.append({
            "horizon": h,
            "t0_pr_auc": M.pr_auc(t0["y"], t0["p"]), "window_pr_auc": M.pr_auc(win["y"], win["p"]),
            "t0_roc_auc": M.roc_auc(t0["y"], t0["p"]),
            "window_roc_auc": M.roc_auc(win["y"], win["p"]),
            "window_minus_t0_pr_auc": cmp["pr_auc_diff"],
            "pr_auc_cluster_ci_low": cmp["pr_auc_cluster_ci_low"],
            "pr_auc_cluster_ci_high": cmp["pr_auc_cluster_ci_high"],
            "pr_auc_cluster_p": cmp["pr_auc_cluster_p"],
            "window_minus_t0_roc_auc": cmp["roc_auc_diff"],
            "roc_auc_cluster_ci_low": cmp["roc_auc_cluster_ci_low"],
            "roc_auc_cluster_ci_high": cmp["roc_auc_cluster_ci_high"],
            "roc_auc_cluster_p": cmp["roc_auc_cluster_p"],
            "window_helps": bool(cmp["pr_auc_cluster_ci_low"] > 0),
            "n_clusters": cmp["n_clusters"],
        })
    return pd.DataFrame(rows)


def h1b_growth_ratio_explanation(splits, horizons=C.HORIZONS,
                                 n_trials: int = C.TUNING_TRIALS,
                                 seeds=C.SEEDS, force: bool = False) -> pd.DataFrame:
    """Why does the eight-quarter window buy the tree so little?

    A candidate explanation: four of the 29 ratios (r21 revenue growth, r22 net
    income growth, r23 assets growth, r24 equity growth) are year-on-year, so a
    single t-0 row already carries a four-quarter comparison.  The window would
    then be re-supplying history the t-0 model already has.

    The test is to drop those four and repeat the comparison.  If the
    explanation holds, removing them should cost the t-0 model more than the
    window model, and the window-minus-t0 gap should widen.
    """
    rows = []
    for h in horizons:
        TAB.run_tabular("xgb_t0_25", h, splits, model="xgboost",
                        feature_cols=NON_GROWTH_IDX, t0_only=True,
                        with_indicators=True, seeds=seeds, n_trials=n_trials,
                        force=force, note="25 ratios at t-0, year-on-year growth removed")
        TAB.run_tabular("xgb_win_25", h, splits, model="xgboost",
                        feature_cols=NON_GROWTH_IDX, t0_only=False,
                        with_indicators=True, seeds=seeds, n_trials=n_trials,
                        force=force, note="25 ratios x 8 quarters, growth removed")

        full_t0 = model_predictions("xgb_t0_29", h)
        full_win = model_predictions("xgboost", h, seeds)
        no_t0 = E.load_preds(f"xgb_t0_25_{h}", "test")
        no_win = E.load_preds(f"xgb_win_25_{h}", "test")
        if full_t0 is None or full_win is None:
            continue
        y, g = full_win["y"], full_win["groups"]

        gap_with = M.cluster_bootstrap_diff(y, full_t0["p"], full_win["p"], g,
                                            n_boot=N_BOOT)
        gap_without = M.cluster_bootstrap_diff(y, no_t0["p_hat"].to_numpy(),
                                               no_win["p_hat"].to_numpy(), g,
                                               n_boot=N_BOOT)
        t0_cost = M.cluster_bootstrap_diff(y, no_t0["p_hat"].to_numpy(), full_t0["p"], g,
                                           n_boot=N_BOOT)
        win_cost = M.cluster_bootstrap_diff(y, no_win["p_hat"].to_numpy(), full_win["p"], g,
                                            n_boot=N_BOOT)
        rows.append({
            "horizon": h,
            "t0_29_pr_auc": M.pr_auc(y, full_t0["p"]),
            "t0_25_pr_auc": M.pr_auc(y, no_t0["p_hat"].to_numpy()),
            "window_29_pr_auc": M.pr_auc(y, full_win["p"]),
            "window_25_pr_auc": M.pr_auc(y, no_win["p_hat"].to_numpy()),
            # What the growth ratios are worth to each model.
            "growth_worth_to_t0": t0_cost["diff"],
            "growth_worth_to_t0_ci_low": t0_cost["ci_low"],
            "growth_worth_to_t0_ci_high": t0_cost["ci_high"],
            "growth_worth_to_window": win_cost["diff"],
            "growth_worth_to_window_ci_low": win_cost["ci_low"],
            "growth_worth_to_window_ci_high": win_cost["ci_high"],
            # The window's value, with and without the growth ratios present.
            "window_gap_with_growth": gap_with["diff"],
            "window_gap_with_growth_ci_low": gap_with["ci_low"],
            "window_gap_with_growth_ci_high": gap_with["ci_high"],
            "window_gap_without_growth": gap_without["diff"],
            "window_gap_without_growth_ci_low": gap_without["ci_low"],
            "window_gap_without_growth_ci_high": gap_without["ci_high"],
            "gap_widens_without_growth": bool(gap_without["diff"] > gap_with["diff"]),
            "growth_worth_more_to_t0": bool(t0_cost["diff"] > win_cost["diff"]),
            "supports_explanation": bool(gap_without["diff"] > gap_with["diff"]
                                         and t0_cost["diff"] > win_cost["diff"]),
        })
    return pd.DataFrame(rows)


def h2_static_deep_control(splits, horizons=C.HORIZONS, seeds=C.SEEDS,
                           n_trials: int = C.TUNING_TRIALS, force: bool = False) -> pd.DataFrame:
    """A deep model with no time axis, to separate "deep" from "temporal"."""
    tune_bundle = E.bundle(splits, horizon=Tune.TUNE_HORIZON, t0_only=True)
    Tune.tune("mlp", tune_bundle, n_trials=n_trials, force=force, path=MLP_29_TUNING)
    hp = Tune.best_hparams_at(MLP_29_TUNING)

    rows = []
    for h in horizons:
        b = E.bundle(splits, horizon=h, t0_only=True)
        for seed in seeds:
            spec = E.RunSpec(arch="mlp", horizon=h, treatment="class_weight", seed=seed,
                             tag="h2mlp", hp=hp,
                             note="H2: all 29 ratios at t-0, static MLP")
            E.run_deep(spec, b, force=force)
        mlp = model_predictions("mlp_t0_29", h, seeds)
        lstm = model_predictions("lstm", h, seeds)
        logreg = model_predictions("logreg", h)
        if mlp is None or lstm is None:
            continue
        cmp = M.compare(lstm["y"], mlp["p"], lstm["p"], lstm["groups"], n_boot=N_BOOT)
        rows.append({
            "horizon": h,
            "logreg_window_pr_auc": M.pr_auc(logreg["y"], logreg["p"]) if logreg else np.nan,
            "mlp_t0_pr_auc": M.pr_auc(mlp["y"], mlp["p"]),
            "lstm_window_pr_auc": M.pr_auc(lstm["y"], lstm["p"]),
            "mlp_t0_roc_auc": M.roc_auc(mlp["y"], mlp["p"]),
            "lstm_window_roc_auc": M.roc_auc(lstm["y"], lstm["p"]),
            "lstm_minus_mlp_pr_auc": cmp["pr_auc_diff"],
            "pr_auc_cluster_ci_low": cmp["pr_auc_cluster_ci_low"],
            "pr_auc_cluster_ci_high": cmp["pr_auc_cluster_ci_high"],
            "time_axis_helps_beyond_nonlinearity": bool(cmp["pr_auc_cluster_ci_low"] > 0),
            "aggregation": "both are 5-seed ensembles",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# H3 — pairwise significance for the architecture comparison
# --------------------------------------------------------------------------
PAIRS = [("xgboost", "transformer"), ("xgboost", "C_lstm"),
         ("transformer", "lstm"), ("xgboost", "lstm"),
         ("transformer", "C_lstm")]


def h3_pairwise(horizons=C.HORIZONS, seeds=C.SEEDS, pairs=PAIRS) -> pd.DataFrame:
    """Firm-clustered intervals on every headline architecture comparison.

    The architecture table previously reported means with no test attached; the
    decomposition ladder was the only place anything was tested.
    """
    rows = []
    for h in horizons:
        cache = {n: model_predictions(n, h, seeds) for n in
                 {m for pair in pairs for m in pair}}
        for a, b in pairs:
            da, db = cache.get(a), cache.get(b)
            if da is None or db is None:
                continue
            thr_a, _ = M.best_f1_threshold(da["yv"], da["pv"])
            thr_b, _ = M.best_f1_threshold(db["yv"], db["pv"])
            cmp = M.compare(da["y"], da["p"], db["p"], da["groups"],
                            thr1=thr_a, thr2=thr_b, n_boot=N_BOOT)
            rows.append({
                "horizon": h, "model_a": a, "model_b": b,
                "a_pr_auc": M.pr_auc(da["y"], da["p"]),
                "b_pr_auc": M.pr_auc(db["y"], db["p"]),
                "a_aggregation": da["aggregation"], "b_aggregation": db["aggregation"],
                **cmp,
                "pr_auc_cluster_excludes_zero": bool(
                    np.isfinite(cmp["pr_auc_cluster_ci_low"])
                    and (cmp["pr_auc_cluster_ci_low"] > 0
                         or cmp["pr_auc_cluster_ci_high"] < 0)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# H4 — precision@k and capture
# --------------------------------------------------------------------------
def h4_precision_at_k(horizons=C.HORIZONS, seeds=C.SEEDS,
                      models=REPORT_MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How many of the k riskiest actually fail, at window and at firm level."""
    rows, curves = [], []
    for h in horizons:
        for name in models:
            d = model_predictions(name, h, seeds)
            if d is None:
                continue
            for level in ("window", "firm"):
                if level == "window":
                    y, p = d["y"], d["p"]
                else:
                    y, p = M.firm_level_scores(d["y"], d["p"], d["groups"])
                n = len(y)
                budgets = [("k=50", 50), ("k=100", 100), ("k=200", 200),
                           ("top 1%", max(1, round(0.01 * n))),
                           ("top 5%", max(1, round(0.05 * n)))]
                for label, k in budgets:
                    r = M.precision_at_k(y, p, k)
                    rows.append({"horizon": h, "model": name, "level": level,
                                 "budget": label, "n": n, "n_positive": int(y.sum()),
                                 "base_rate": float(y.mean()), **r})
                cc = M.capture_curve(y, p)
                cc["horizon"] = h; cc["model"] = name; cc["level"] = level
                curves.append(cc)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def capture_figure(curves: pd.DataFrame, out_stem, level: str = "firm") -> None:
    import matplotlib.pyplot as plt

    sub = curves[curves["level"] == level]
    horizons = sorted(sub["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.8),
                             squeeze=False, sharey=True)
    for i, h in enumerate(horizons):
        ax = axes[0][i]
        for name in sorted(sub["model"].unique()):
            c = sub[(sub["horizon"] == h) & (sub["model"] == name)]
            ax.plot(c["frac_flagged"], c["frac_captured"], lw=1.5, label=name)
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1,
                label="random" if i == 0 else None)
        ax.set_xscale("log")
        ax.set_title(f"h={h}", fontsize=10)
        ax.set_xlabel("fraction of firms flagged")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel(f"fraction of failures captured ({level} level)")
            ax.legend(fontsize=8)
    fig.suptitle("Capture curves — how much of the failure population a given "
                 "alarm budget reaches")
    fig.tight_layout()
    U.savefig(fig, out_stem)


# --------------------------------------------------------------------------
# H5 — calibration
# --------------------------------------------------------------------------
def h5_calibration(horizons=C.HORIZONS, seeds=C.SEEDS,
                   models=REPORT_MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, tables = [], []
    for h in horizons:
        for name in models:
            d = model_predictions(name, h, seeds)
            if d is None:
                continue
            p = d["p"]
            is_prob = bool(np.all((p >= 0) & (p <= 1)))
            if is_prob:
                pp, how = p, "native probability"
            else:
                # Altman, Ohlson and Zmijewski emit scores, not probabilities.
                # Asking whether a raw Z is "calibrated" is meaningless -- it
                # produced an ECE of 0.49, which says only that a Z-score is not
                # a probability.  The fair question is whether the score's
                # *ranking* can be turned into calibrated probabilities, so it
                # is Platt-scaled: a one-variable logistic fitted on validation
                # scores and applied once to test.  Monotone, so every ranking
                # metric is untouched, and fitted on val only.
                pp = M.platt_scale(d["yv"], d["pv"], p)
                how = "Platt-scaled on validation"
            tab, st = M.calibration(d["y"], pp)
            tab["horizon"] = h; tab["model"] = name
            tables.append(tab)
            summary.append({"horizon": h, "model": name, "is_probability": is_prob,
                            "scaling": how, "base_rate": float(d["y"].mean()),
                            "mean_predicted": float(pp.mean()), **st})
    return pd.DataFrame(summary), pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def calibration_figure(tables: pd.DataFrame, out_stem) -> None:
    import matplotlib.pyplot as plt

    horizons = sorted(tables["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.8),
                             squeeze=False)
    for i, h in enumerate(horizons):
        ax = axes[0][i]
        sub = tables[tables["horizon"] == h]
        for name in sorted(sub["model"].unique()):
            c = sub[sub["model"] == name].sort_values("mean_predicted")
            ax.plot(c["mean_predicted"], c["observed_rate"], "-o", ms=3, lw=1.3, label=name)
        lim = max(sub["mean_predicted"].max(), sub["observed_rate"].max())
        ax.plot([0, lim], [0, lim], ls=":", c="grey", lw=1)
        ax.set_title(f"h={h}", fontsize=10)
        ax.set_xlabel("mean predicted"); ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("observed failure rate"); ax.legend(fontsize=8)
    fig.suptitle("Reliability, quantile bins (a score below the diagonal is over-confident)")
    fig.tight_layout()
    U.savefig(fig, out_stem)


# --------------------------------------------------------------------------
# H6 — who does this work for?
# --------------------------------------------------------------------------
MIN_POSITIVES = 20


def _test_covariates(full: bool = True) -> pd.DataFrame:
    """Total assets and SIC for every test window, keyed like the predictions."""
    from . import phase_b as PB

    j, _ = PB.build_levels_table(full=full)
    te = j[j["split"] == "test"][["cik", "end_quarter", "Assets"]].copy()
    te["cik"] = te["cik"].astype(str)

    panel = pd.read_parquet(C.ratios_path(full=full), columns=["cik", "quarter", "sic"])
    panel["cik"] = panel["cik"].astype(str)
    sic = panel.drop_duplicates(["cik", "quarter"]).set_index(["cik", "quarter"])["sic"]
    te["sic"] = sic.reindex(
        pd.MultiIndex.from_arrays([te["cik"], te["end_quarter"]])).to_numpy()
    te["sic2"] = (te["sic"] // 100).astype("Int64")

    # Size terciles are cut on the TRAIN period, so the test split never informs
    # its own grouping.
    tr_assets = j.loc[j["split"] == "train", "Assets"].to_numpy(dtype=float)
    cuts = np.nanquantile(tr_assets, [1 / 3, 2 / 3])
    te["size_tercile"] = pd.cut(te["Assets"], [-np.inf, *cuts, np.inf],
                                labels=["small", "mid", "large"])
    return te, cuts


def h6_breakdowns(horizons=C.HORIZONS, seeds=C.SEEDS, models=REPORT_MODELS,
                  full: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cov, cuts = _test_covariates(full=full)
    size_rows, sector_rows = [], []
    for h in horizons:
        for name in models:
            d = model_predictions(name, h, seeds)
            if d is None:
                continue
            idx = pd.MultiIndex.from_arrays([d["groups"], _end_quarters(name, h, seeds)])
            c = cov.set_index(["cik", "end_quarter"]).reindex(idx)

            for tercile in ("small", "mid", "large"):
                m = (c["size_tercile"].to_numpy() == tercile)
                size_rows.append(_slice_row(d, m, {"horizon": h, "model": name,
                                                   "group_kind": "size tercile",
                                                   "group": tercile}))
            for s2 in sorted(pd.unique(c["sic2"].dropna())):
                m = (c["sic2"].to_numpy() == s2)
                sector_rows.append(_slice_row(d, m, {"horizon": h, "model": name,
                                                     "group_kind": "SIC-2",
                                                     "group": int(s2)}))
    return (pd.DataFrame(size_rows), pd.DataFrame(sector_rows),
            {"train_asset_tercile_cuts": [float(x) for x in cuts],
             "min_test_positives_to_report": MIN_POSITIVES})


def _end_quarters(name: str, h: int, seeds) -> np.ndarray:
    """The end quarters of a model's test rows, in prediction order."""
    single = {"altman_zdp": f"altman_zdp_{h}", "xgboost": f"xgboost_{h}",
              "xgb_t0_29": f"xgb_t0_29_{h}"}
    if name in single:
        key = single[name]
    else:
        arch, tag = {"transformer": ("transformer", ""), "lstm": ("lstm", ""),
                     "C_lstm": ("lstm", "decompC"), "mlp_t0_29": ("mlp", "h2mlp")}[name]
        key = E.RunSpec(arch=arch, horizon=h, treatment="class_weight",
                        seed=seeds[0], tag=tag).key
    return E.load_preds(key, "test")["end_quarter"].astype(str).to_numpy()


def _slice_row(d: dict, mask: np.ndarray, meta: dict) -> dict:
    """PR-AUC with a cluster CI on one slice, or an explicit 'too few'."""
    y, p, g = d["y"][mask], d["p"][mask], d["groups"][mask]
    n_pos = int(y.sum())
    row = {**meta, "n": int(mask.sum()), "n_positive": n_pos,
           "base_rate": float(y.mean()) if mask.sum() else np.nan,
           "n_firms": int(len(np.unique(g))) if mask.sum() else 0}
    if n_pos < MIN_POSITIVES:
        row.update({"pr_auc": np.nan, "roc_auc": np.nan, "pr_auc_cluster_ci_low": np.nan,
                    "pr_auc_cluster_ci_high": np.nan,
                    "status": f"too few ({n_pos} test positives, need {MIN_POSITIVES})"})
        return row
    ci = M.cluster_bootstrap_ci(y, p, g, stat=M.pr_auc, n_boot=1000)
    row.update({"pr_auc": M.pr_auc(y, p), "roc_auc": M.roc_auc(y, p),
                "pr_auc_cluster_ci_low": ci["ci_low"],
                "pr_auc_cluster_ci_high": ci["ci_high"], "status": "reported"})
    return row


# --------------------------------------------------------------------------
def main(full: bool = True, horizons=C.HORIZONS, seeds=C.SEEDS,
         n_trials: int = C.TUNING_TRIALS, force: bool = False) -> dict:
    print("=" * 70)
    print("PHASE H — CONTROLS AND PRACTITIONER METRICS")
    print("=" * 70)
    splits = D.load_all(full=full)

    h1 = h1_temporal_control(splits, horizons=horizons, n_trials=n_trials, force=force)
    U.write_table(h1, C.RESULTS / "h1_temporal_control")
    if len(h1):
        print(h1[["horizon", "t0_pr_auc", "window_pr_auc", "window_minus_t0_pr_auc",
                  "pr_auc_cluster_ci_low", "pr_auc_cluster_ci_high",
                  "window_helps"]].round(4).to_string(index=False))

    h1b = h1b_growth_ratio_explanation(splits, horizons=horizons, seeds=seeds,
                                       n_trials=n_trials, force=force)
    U.write_table(h1b, C.RESULTS / "h1b_growth_ratio_explanation")
    if len(h1b):
        print(h1b[["horizon", "window_gap_with_growth", "window_gap_without_growth",
                   "growth_worth_to_t0", "growth_worth_to_window",
                   "supports_explanation"]].round(4).to_string(index=False))

    h2 = h2_static_deep_control(splits, horizons=horizons, seeds=seeds,
                                n_trials=n_trials, force=force)
    U.write_table(h2, C.RESULTS / "h2_static_deep_control")

    h3 = h3_pairwise(horizons=horizons, seeds=seeds)
    U.write_table(h3, C.RESULTS / "h3_pairwise_significance")

    pk, curves = h4_precision_at_k(horizons=horizons, seeds=seeds)
    U.write_table(pk, C.RESULTS / "h4_precision_at_k")
    if len(curves):
        curves.to_csv(C.RESULTS / "h4_capture_curves.csv", index=False)
        capture_figure(curves, C.FIGURES / "capture_curves_firm", level="firm")
        capture_figure(curves, C.FIGURES / "capture_curves_window", level="window")

    cal, cal_tab = h5_calibration(horizons=horizons, seeds=seeds)
    U.write_table(cal, C.RESULTS / "h5_calibration", floatfmt="%.5f")
    if len(cal_tab):
        cal_tab.to_csv(C.RESULTS / "h5_calibration_bins.csv", index=False)
        calibration_figure(cal_tab, C.FIGURES / "calibration")

    size, sector, notes = h6_breakdowns(horizons=horizons, seeds=seeds, full=full)
    U.write_table(size, C.RESULTS / "h6_size_breakdown")
    U.write_table(sector, C.RESULTS / "h6_sector_breakdown")
    U.write_json(C.RESULTS / "h6_breakdown_notes.json", notes)

    n_reported = int((sector["status"] == "reported").sum()) if len(sector) else 0
    print(f"[phase H] sectors with >= {MIN_POSITIVES} test positives: {n_reported} "
          f"of {len(sector)} model-sector cells")
    return {"h1_rows": len(h1), "h1b_rows": len(h1b), "h2_rows": len(h2), "h3_rows": len(h3),
            "precision_at_k_rows": len(pk), "sector_cells_reported": n_reported}
