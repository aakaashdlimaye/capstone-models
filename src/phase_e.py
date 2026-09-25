"""Phase E — the decomposition experiment (Contribution 3).

Four models on the identical row set:

  A  Altman's variables, published coefficients, static t-0
  B  Altman's variables, coefficients re-fit on train, static t-0
  C  Altman's variables, coefficients learned, 8-quarter LSTM
  D  all 29 ratios, coefficients learned, 8-quarter LSTM

A->B is coefficient drift, B->C is the static formulation, C->D is the feature
set.  Model B is fitted twice: once on the annualised panel levels (so A and B
differ only in coefficients) and once on the tensor's own t-0 ratios (so B and C
differ only in the time axis).  Reporting both keeps each gap attributable
instead of silently absorbing a change of variable definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import classical as K
from . import config as C
from . import data as D
from . import experiment as E
from . import metrics as M
from . import phase_b as PB
from . import tuning as Tune
from . import utils as U

# B->C used to change four things at once -- linear to nonlinear, static to
# sequential, annualised panel levels to z-scored quarterly tensor ratios, and
# one model to a five-seed ensemble.  The ladder now walks those changes one at
# a time, so each rung differs from the one below it in exactly one respect.
PRIMARY_LADDER = ["A_zdp", "B_levels", "B_tensor", "B_mlp_t0", "C_lstm", "D_lstm"]
LADDER = PRIMARY_LADDER + ["D_transformer"]
OLD_LADDER = ["A_zdp", "B_levels", "C_lstm", "D_lstm"]   # kept for comparability

CAUSE = {
    "A_zdp->B_levels": "coefficient drift",
    "B_levels->B_tensor": "preprocessing (annualised levels to z-scored quarterly ratios)",
    "B_tensor->B_mlp_t0": "nonlinearity",
    "B_mlp_t0->C_lstm": "the time axis",
    "C_lstm->D_lstm": "the feature-set expansion",
    "D_lstm->D_transformer": "architecture",
}

# Every rung below B_mlp_t0 is a single deterministic fit; every rung from
# B_mlp_t0 up is the mean of five seeds' predicted probabilities.  Stated in the
# table so the aggregation convention is never implicit.
ENSEMBLE_RUNGS = {"B_mlp_t0", "C_lstm", "D_lstm", "D_transformer",
                  "B_xgb_t0", "C_xgb"}

# A refit two years closer to the test window has to recover at least this much
# of the ROC-AUC drop, with a firm-clustered interval clear of zero, before the
# drop is called distribution shift.  Without a floor the flag fires on noise.
MIN_RECOVERY_SHARE = 0.10


# --------------------------------------------------------------------------
def _save(key: str, j: pd.DataFrame, score: np.ndarray, h: int) -> None:
    PB.save_classical_preds(key, j, score, h)


def static_models(j: pd.DataFrame, splits, horizons=C.HORIZONS) -> pd.DataFrame:
    """Models A and B, both static, both on the end quarter of every window."""
    is_tr = (j["split"] == "train").to_numpy()
    rows = []

    # The tensor's own five Altman ratios at t-0, for the B-tensor variant.
    # `j` is built split-block-wise in tensor row order; the assertion below
    # proves that alignment instead of trusting it.
    Xt = np.zeros((len(j), len(C.ALTMAN_IDX)), dtype=np.float32)
    for s in ("train", "val", "test"):
        idx = np.flatnonzero((j["split"] == s).to_numpy())
        d = splits[s]
        assert len(idx) == d.n, f"{s}: row set has {len(idx)} rows, tensor has {d.n}"
        assert (j["cik"].to_numpy()[idx] == d.cik).all(), f"{s}: cik order mismatch"
        assert (j["end_quarter"].to_numpy()[idx] == d.end_quarter).all(),             f"{s}: end_quarter order mismatch"
        Xt[idx] = d.X[:, -1, C.ALTMAN_IDX]
    Xt_df = pd.DataFrame(Xt, columns=C.ALTMAN_NAMES)

    for h in horizons:
        y = j[f"y{h}"].to_numpy().astype(int)

        # --- A: published coefficients -----------------------------------
        for key, fn in (("A_zdp", K.altman_z_double_prime), ("A_zp", K.altman_z_prime)):
            s = -fn(j)                      # negate: low Z means distress
            _save(f"decomp{key}_{h}", j, s, h)
            rows.append({"model": key, "horizon": h, "variables": "Altman 5",
                         "coefficients": "published", "temporal": "static t-0"})

        # --- B: coefficients re-estimated on train -----------------------
        pipe, coefs = K.refit_altman(j[is_tr], y[is_tr], ["X1", "X2", "X3", "X4"])
        s = pipe.predict_proba(j[["X1", "X2", "X3", "X4"]].to_numpy(float))[:, 1]  # noqa: E501
        _save(f"decompB_levels_{h}", j, s, h)
        rows.append({"model": "B_levels", "horizon": h, "variables": "Altman 4 (Z'' set)",
                     "coefficients": "re-fit on train (levels)", "temporal": "static t-0",
                     **{f"coef_{k}": v for k, v in coefs.items()}})

        pipe_t, coefs_t = K.refit_altman(Xt_df[is_tr], y[is_tr], C.ALTMAN_NAMES)
        s_t = pipe_t.predict_proba(Xt_df[C.ALTMAN_NAMES].to_numpy(float))[:, 1]
        _save(f"decompB_tensor_{h}", j, s_t, h)
        rows.append({"model": "B_tensor", "horizon": h, "variables": "Altman 5 (tensor t-0)",
                     "coefficients": "re-fit on train (tensor ratios)",
                     "temporal": "static t-0",
                     **{f"coef_{k}": v for k, v in coefs_t.items()}})
    return pd.DataFrame(rows)


def temporal_models(splits, horizons=C.HORIZONS, seeds=C.SEEDS, force: bool = False):
    """Model C (Altman five, LSTM) — D reuses the Phase C runs unchanged."""
    records = []
    for h in horizons:
        b = E.bundle(splits, horizon=h, feature_cols=C.ALTMAN_IDX)
        hp = Tune.best_hparams("lstm")
        for seed in seeds:
            spec = E.RunSpec(arch="lstm", horizon=h, treatment="class_weight", seed=seed,
                             tag="decompC", feature_cols=C.ALTMAN_IDX, hp=hp,
                             note="Model C: Altman's five ratios, 8-quarter LSTM")
            records.append(E.run_deep(spec, b, force=force))
            print(f"[phase E] {spec.key:42s} done", flush=True)
    return records


MLP_T0_TUNING = C.RESULTS / "tuning" / "mlp_t0_altman5.json"


def nonlinearity_rung(splits, horizons=C.HORIZONS, seeds=C.SEEDS,
                      n_trials: int = C.TUNING_TRIALS, force: bool = False):
    """B_mlp_t0 — the same five ratios, nonlinear, still with no time axis.

    Without this rung B->C credits the time axis with everything a nonlinear
    model buys.  Tuned on validation PR-AUC with the same 30-trial budget every
    other architecture gets, on its own study because the input shape differs.
    """
    records = []
    tune_bundle = E.bundle(splits, horizon=Tune.TUNE_HORIZON,
                           feature_cols=C.ALTMAN_IDX, t0_only=True)
    Tune.tune("mlp", tune_bundle, n_trials=n_trials, force=force, path=MLP_T0_TUNING)
    hp = Tune.best_hparams_at(MLP_T0_TUNING)
    for h in horizons:
        b = E.bundle(splits, horizon=h, feature_cols=C.ALTMAN_IDX, t0_only=True)
        for seed in seeds:
            spec = E.RunSpec(arch="mlp", horizon=h, treatment="class_weight", seed=seed,
                             tag="decompBmlp", feature_cols=C.ALTMAN_IDX, hp=hp,
                             note="B_mlp_t0: Altman's five ratios at t-0, static MLP")
            records.append(E.run_deep(spec, b, force=force))
    print(f"[phase E] B_mlp_t0: {len(records)} runs", flush=True)
    return records


def tree_rungs(splits, horizons=C.HORIZONS, seeds=C.SEEDS,
               n_trials: int = C.TUNING_TRIALS,
               force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same ladder walked by gradient boosting instead of a neural net.

    If the time axis is what pays, it should pay for a tree as well: B_xgb_t0
    (five ratios at t-0) against C_xgb (the same five across eight quarters)
    isolates it without any neural network in the comparison.  Both are five-seed
    ensembles and the difference carries a firm-clustered interval, because this
    is the cleanest evidence in the study and a single fit with no interval
    would not carry it.
    """
    from . import tabular as TAB

    rows, tests = [], []
    for h in horizons:
        rows.append(TAB.run_tabular(
            "decompB_xgb_t0", h, splits, model="xgboost", feature_cols=C.ALTMAN_IDX,
            t0_only=True, seeds=seeds, n_trials=n_trials, force=force,
            note="Altman's five ratios at t-0 only"))
        rows.append(TAB.run_tabular(
            "decompC_xgb", h, splits, model="xgboost", feature_cols=C.ALTMAN_IDX,
            t0_only=False, seeds=seeds, n_trials=n_trials, force=force,
            note="Altman's five ratios across all eight quarters (40 inputs)"))

        t0 = E.load_preds(f"decompB_xgb_t0_{h}", "test")
        win = E.load_preds(f"decompC_xgb_{h}", "test")
        y = win["y_true"].to_numpy().astype(int)
        g = win["cik"].astype(str).to_numpy()
        cmp = M.compare(y, t0["p_hat"].to_numpy(), win["p_hat"].to_numpy(), g,
                        n_boot=2000)
        tests.append({
            "horizon": h, "comparison": "C_xgb (5 ratios x 8 quarters) - B_xgb_t0 (5 at t-0)",
            "t0_pr_auc": M.pr_auc(y, t0["p_hat"].to_numpy()),
            "window_pr_auc": M.pr_auc(y, win["p_hat"].to_numpy()),
            "aggregation": f"both are {len(seeds)}-seed ensembles",
            **cmp,
            "window_helps_pr_auc": bool(cmp["pr_auc_cluster_ci_low"] > 0),
            "window_helps_roc_auc": bool(cmp["roc_auc_cluster_ci_low"] > 0),
        })
    return pd.DataFrame(rows), pd.DataFrame(tests)


def seed_mean_pred(key_fn, seeds, split: str, col: str = "p_hat") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Average predicted probability across seeds — the ensemble the tests use."""
    frames = [E.load_preds(key_fn(s), split) for s in seeds]
    base = frames[0][["cik", "end_quarter", "y_true"]].reset_index(drop=True)
    P = np.column_stack([f[col].to_numpy() for f in frames])
    return base["y_true"].to_numpy().astype(int), P.mean(axis=1), base


def collect_predictions(horizon: int, seeds=C.SEEDS) -> dict[str, dict]:
    """One prediction vector per ladder rung, on val and on test."""
    out = {}
    for name, key in (("A_zdp", f"decompA_zdp_{horizon}"),
                      ("A_zp", f"decompA_zp_{horizon}"),
                      ("B_levels", f"decompB_levels_{horizon}"),
                      ("B_tensor", f"decompB_tensor_{horizon}"),
                      ("B_xgb_t0", f"decompB_xgb_t0_{horizon}"),
                      ("C_xgb", f"decompC_xgb_{horizon}")):
        if not (C.PREDS / f"{key}.parquet").exists():
            continue
        te, va = E.load_preds(key, "test"), E.load_preds(key, "val")
        out[name] = {"y": te["y_true"].to_numpy().astype(int), "p": te["p_hat"].to_numpy(),
                     "yv": va["y_true"].to_numpy().astype(int), "pv": va["p_hat"].to_numpy(),
                     "groups": te["cik"].astype(str).to_numpy()}
    for name, kf in (
        ("B_mlp_t0", lambda s: E.RunSpec(arch="mlp", horizon=horizon,
                                         treatment="class_weight", seed=s,
                                         tag="decompBmlp").key),
        ("C_lstm", lambda s: E.RunSpec(arch="lstm", horizon=horizon,
                                       treatment="class_weight", seed=s, tag="decompC").key),
        ("D_lstm", lambda s: E.RunSpec(arch="lstm", horizon=horizon,
                                       treatment="class_weight", seed=s).key),
        ("D_transformer", lambda s: E.RunSpec(arch="transformer", horizon=horizon,
                                              treatment="class_weight", seed=s).key),
    ):
        if not (C.PREDS / f"{kf(seeds[0])}.parquet").exists():
            continue
        y, p, base = seed_mean_pred(kf, seeds, "test")
        yv, pv, _ = seed_mean_pred(kf, seeds, "val")
        out[name] = {"y": y, "p": p, "yv": yv, "pv": pv,
                     "groups": base["cik"].astype(str).to_numpy(),
                     "per_seed": [kf(s) for s in seeds]}
    return out


# --------------------------------------------------------------------------
def gaps(preds: dict, horizon: int, ladder=PRIMARY_LADDER, n_boot: int = 2000) -> pd.DataFrame:
    """Each rung's metrics, and the step from the rung below it.

    Intervals are **firm-clustered** bootstraps: stride-1 windows from one firm
    share seven of their eight quarters and one event, so a window-level
    interval understates the uncertainty.  DeLong and the window-level bootstrap
    are carried beside them under `window_level_` names.
    """
    ladder = [r for r in ladder if r in preds]
    rows = []
    for i, name in enumerate(ladder):
        d = preds[name]
        thr, _ = M.best_f1_threshold(d["yv"], d["pv"])
        g = d["groups"]
        r = {"model": name, "horizon": horizon,
             "aggregation": "ensemble of 5 seeds" if name in ENSEMBLE_RUNGS else "single fit",
             "roc_auc": M.roc_auc(d["y"], d["p"]), "pr_auc": M.pr_auc(d["y"], d["p"])}
        # The rung's own interval, named `_value_` so it cannot be confused with
        # (or overwritten by) the interval on the *difference* from the rung
        # below, which `M.compare` adds under `{metric}_cluster_ci_*`.
        for metric, stat in (("pr_auc", M.pr_auc), ("roc_auc", M.roc_auc)):
            ci = M.cluster_bootstrap_ci(d["y"], d["p"], g, stat=stat, n_boot=n_boot)
            r[f"{metric}_value_cluster_ci_low"] = ci["ci_low"]
            r[f"{metric}_value_cluster_ci_high"] = ci["ci_high"]
        r.update({k: v for k, v in M.threshold_metrics(d["y"], d["p"], thr).items()
                  if k in ("threshold", "f1", "recall", "specificity", "precision")})
        if i > 0:
            prev = preds[ladder[i - 1]]
            thr_prev, _ = M.best_f1_threshold(prev["yv"], prev["pv"])
            step = f"{ladder[i - 1]} -> {name}"
            r["step"] = step
            r["cause"] = CAUSE.get(step.replace(" -> ", "->"), "")
            cmp = M.compare(d["y"], prev["p"], d["p"], g, thr1=thr_prev, thr2=thr,
                            n_boot=n_boot)
            r.update(cmp)
            lo, hi = cmp["pr_auc_cluster_ci_low"], cmp["pr_auc_cluster_ci_high"]
            r["pr_auc_cluster_excludes_zero"] = bool(np.isfinite(lo) and (lo > 0 or hi < 0))
            lo, hi = cmp["roc_auc_cluster_ci_low"], cmp["roc_auc_cluster_ci_high"]
            r["roc_auc_cluster_excludes_zero"] = bool(np.isfinite(lo) and (lo > 0 or hi < 0))
        rows.append(r)
    return pd.DataFrame(rows)


def gap_shares(tab: pd.DataFrame, ladder=PRIMARY_LADDER) -> dict:
    """What share of the A->D movement does each step account for?

    Expressing a step as a share of the *net* A->D gap collapses as soon as one
    step moves backwards: a small net total in the denominator turns an ordinary
    step into several hundred percent.  Shares here are taken over the total
    **absolute** movement across the three steps, which is bounded, sums to one,
    and stays readable when a step degrades.  The signed change is reported
    beside it so the direction is never lost.
    """
    t = tab.set_index("model")
    out = {"horizon": int(tab["horizon"].iloc[0])}
    for metric in ("roc_auc", "pr_auc"):
        steps = {f"{ladder[i - 1]}->{ladder[i]}":
                 float(t.loc[ladder[i], metric] - t.loc[ladder[i - 1], metric])
                 for i in range(1, len(ladder))}
        total_abs = sum(abs(v) for v in steps.values())
        out[f"{metric}_A"] = float(t.loc[ladder[0], metric])
        out[f"{metric}_D"] = float(t.loc[ladder[-1], metric])
        out[f"{metric}_net_gap"] = float(t.loc[ladder[-1], metric] - t.loc[ladder[0], metric])
        out[f"{metric}_total_abs_movement"] = total_abs
        for step, g in steps.items():
            out[f"{metric}_gap_{step}"] = g
            out[f"{metric}_share_{step}"] = abs(g) / total_abs if total_abs else np.nan
            out[f"{metric}_direction_{step}"] = "improves" if g > 0 else "degrades"
    return out


def headline(shares: dict, tab: pd.DataFrame, ladder=PRIMARY_LADDER,
             metric: str = "pr_auc") -> str:
    """The paper's one-sentence finding, with its numbers and intervals."""
    parts = []
    for i in range(1, len(ladder)):
        step = f"{ladder[i - 1]}->{ladder[i]}"
        if f"{metric}_share_{step}" not in shares:
            continue
        parts.append(f"{shares[f'{metric}_share_{step}'] * 100:.0f}% is "
                     f"{CAUSE.get(step, step)} ({shares[f'{metric}_gap_{step}']:+.4f}, "
                     f"{shares[f'{metric}_direction_{step}']})")
    return (f"At h={shares['horizon']}, test PR-AUC moves from {shares[f'{metric}_A']:.4f} for "
            f"Altman Z″ to {shares[f'{metric}_D']:.4f} for the full temporal model, a net "
            f"{shares[f'{metric}_net_gap']:+.4f}.  Of the "
            f"{shares[f'{metric}_total_abs_movement']:.4f} of total movement across the "
            f"{len(parts)} steps, " + ", ".join(parts) + ".")


def significance_sentence(tab: pd.DataFrame, step: str) -> str:
    """Whether one step's firm-clustered interval excludes zero, in words."""
    t = tab.set_index("step")
    key = step.replace("->", " -> ")
    if key not in t.index:
        return f"{step}: not present at this horizon."
    r = t.loc[key]
    pr = (f"PR-AUC {r['pr_auc_diff']:+.4f} "
          f"[{r['pr_auc_cluster_ci_low']:+.4f}, {r['pr_auc_cluster_ci_high']:+.4f}]")
    roc = (f"ROC-AUC {r['roc_auc_diff']:+.4f} "
           f"[{r['roc_auc_cluster_ci_low']:+.4f}, {r['roc_auc_cluster_ci_high']:+.4f}]")
    verdict = []
    verdict.append("PR-AUC interval excludes zero" if r.get("pr_auc_cluster_excludes_zero")
                   else "PR-AUC interval contains zero")
    verdict.append("ROC-AUC interval excludes zero" if r.get("roc_auc_cluster_excludes_zero")
                   else "ROC-AUC interval contains zero")
    return f"{step}: {pr}, {roc} — {'; '.join(verdict)}."


def pr_curve_figure(preds: dict, horizon: int, out_stem, ladder=PRIMARY_LADDER) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(6.2, 5))
    for name in ladder:
        d = preds[name]
        pr, rc, _ = precision_recall_curve(d["y"], d["p"])
        ax.plot(rc, pr, label=f"{name} (PR-AUC {M.pr_auc(d['y'], d['p']):.4f})", lw=1.6)
    base = float(preds[ladder[0]]["y"].mean())
    ax.axhline(base, ls="--", c="grey", lw=1, label=f"base rate ({base:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_yscale("log")
    ax.set_title(f"Decomposition ladder, test set, h={horizon}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    U.savefig(fig, out_stem)


# --------------------------------------------------------------------------
# Missingness sanity check
# --------------------------------------------------------------------------
def complete_altman_mask(splits) -> dict[str, np.ndarray]:
    """Windows where all five Altman ratios are observed in all eight quarters."""
    return {s: (splits[s].mask[:, :, C.ALTMAN_IDX] == 1).all(axis=(1, 2))
            for s in ("train", "val", "test")}


def complete_subset_check(splits, horizons=(1, 4), seeds=C.SEEDS, force: bool = False):
    """Re-run C and D on the all-complete subset and show the gaps hold."""
    mask = complete_altman_mask(splits)
    frac = {s: float(m.mean()) for s, m in mask.items()}
    print(f"[phase E] all-five-complete windows: {frac}")

    rows = []
    for h in horizons:
        for tag, cols in (("cmplC", C.ALTMAN_IDX), ("cmplD", None)):
            b = E.bundle(splits, horizon=h, feature_cols=cols, row_mask=mask)
            hp = Tune.best_hparams("lstm")
            ps, ys = [], None
            for seed in seeds:
                spec = E.RunSpec(arch="lstm", horizon=h, treatment="class_weight",
                                 seed=seed, tag=tag, feature_cols=cols, hp=hp,
                                 note="all-five-Altman-complete subset")
                E.run_deep(spec, b, force=force)
                te = E.load_preds(spec.key, "test")
                ps.append(te["p_hat"].to_numpy()); ys = te["y_true"].to_numpy().astype(int)
            p = np.mean(ps, axis=0)
            rows.append({"horizon": h, "model": "C_lstm" if cols else "D_lstm",
                         "subset": "all-five-complete", "n_test": len(ys),
                         "n_pos": int(ys.sum()),
                         "roc_auc": M.roc_auc(ys, p), "pr_auc": M.pr_auc(ys, p)})
    out = pd.DataFrame(rows)
    out["train_frac_complete"] = frac["train"]
    out["test_frac_complete"] = frac["test"]
    return out


# --------------------------------------------------------------------------
def main(full: bool = True, horizons=C.HORIZONS, seeds=C.SEEDS,
         force: bool = False) -> dict:
    print("=" * 70)
    print("PHASE E — DECOMPOSITION (A -> B -> C -> D)")
    print("=" * 70)
    splits = D.load_all(full=full)
    j, _ = PB.build_levels_table(full=full)

    coefs = static_models(j, splits, horizons=horizons)
    U.write_table(coefs, C.RESULTS / "decomposition_coefficients", floatfmt="%.5f")

    temporal_models(splits, horizons=horizons, seeds=seeds, force=force)
    nonlinearity_rung(splits, horizons=horizons, seeds=seeds, force=force)
    trees, tree_tests = tree_rungs(splits, horizons=horizons, seeds=seeds, force=force)

    tables, primary, extended, shares, sig = [], {}, {}, [], []
    for h in horizons:
        preds = collect_predictions(h, seeds=seeds)
        tab = gaps(preds, h)
        full_tab = gaps(preds, h, ladder=LADDER)
        tables.append(full_tab)
        # The summary quotes intervals for the primary ladder's own steps, so it
        # needs that table; the extended one carries D_transformer.
        primary[h] = tab
        extended[h] = full_tab
        U.write_table(tab, C.RESULTS / f"decomposition_{h}")
        pr_curve_figure(preds, h, C.FIGURES / f"decomposition_pr_h{h}")
        shares.append(gap_shares(tab))
        for step in CAUSE:
            if step.split("->")[0] in preds and step.split("->")[1] in preds:
                sig.append({"horizon": h, "step": step, "cause": CAUSE[step],
                            "statement": significance_sentence(full_tab, step)})
        # The finding must not be LSTM-specific.
        alt_l = ["A_zdp", "B_levels", "C_lstm", "D_transformer"]
        if all(r in preds for r in alt_l):
            shares[-1].update({f"transformer_{k}": v for k, v in
                               gap_shares(gaps(preds, h, ladder=alt_l),
                                          ladder=alt_l).items()})
    pd.concat(tables, ignore_index=True).to_csv(C.RESULTS / "decomposition_all.csv", index=False)
    sh = pd.DataFrame(shares)
    U.write_table(sh, C.RESULTS / "decomposition_shares", floatfmt="%.4f")
    U.write_table(pd.DataFrame(sig), C.RESULTS / "decomposition_significance")
    U.write_table(trees.drop(columns=["key"], errors="ignore"),
                  C.RESULTS / "decomposition_tree_ladder")
    U.write_table(tree_tests, C.RESULTS / "decomposition_tree_ladder_tests")

    drift = coefficient_drift_diagnostic(j, horizons=horizons)
    U.write_table(drift, C.RESULTS / "decomposition_drift_diagnostic", floatfmt="%.5f")

    chk = complete_subset_check(splits, seeds=seeds, force=force)
    U.write_table(chk, C.RESULTS / "decomposition_complete_subset")

    write_summary(sh, chk, primary, extended, sig=pd.DataFrame(sig), drift=drift,
                  trees=trees, tree_tests=tree_tests)
    return {"horizons": list(horizons)}


def coefficient_drift_diagnostic(j: pd.DataFrame, horizons=C.HORIZONS) -> pd.DataFrame:
    """Is A->B's ROC-AUC drop stale coefficients, or distribution shift?

    Re-estimating Altman's coefficients leaves PR-AUC flat but lowers ROC-AUC at
    the longer horizons.  One explanation is that the training period is no
    longer representative of the test period, in which case fitting on
    train + val -- two years closer to the test window, and still strictly
    before it -- should recover some of the drop.  If it does not, the drop is
    about what the coefficients are fitted *to* rather than when.

    This is a diagnostic, not a reported model: nothing fitted on val scores any
    headline number, and its predictions are written under a `diag_` prefix.
    """
    rows = []
    is_tr = (j["split"] == "train").to_numpy()
    is_trval = j["split"].isin(["train", "val"]).to_numpy()
    is_te = (j["split"] == "test").to_numpy()
    Xall = j[["X1", "X2", "X3", "X4"]].to_numpy(float)
    groups = j.loc[is_te, "cik"].astype(str).to_numpy()

    for h in horizons:
        y = j[f"y{h}"].to_numpy().astype(int)
        out = {"horizon": h}
        for label, mask in (("train_only", is_tr), ("train_plus_val", is_trval)):
            pipe, _ = K.refit_altman(j[mask], y[mask], ["X1", "X2", "X3", "X4"])
            p = pipe.predict_proba(Xall)[:, 1]
            out[f"{label}_roc_auc"] = M.roc_auc(y[is_te], p[is_te])
            out[f"{label}_pr_auc"] = M.pr_auc(y[is_te], p[is_te])
        a = -K.altman_z_double_prime(j)
        out["published_roc_auc"] = M.roc_auc(y[is_te], a[is_te])
        out["published_pr_auc"] = M.pr_auc(y[is_te], a[is_te])
        out["roc_drop_train_only"] = out["train_only_roc_auc"] - out["published_roc_auc"]
        out["roc_drop_train_plus_val"] = out["train_plus_val_roc_auc"] - out["published_roc_auc"]

        p_tr = K.refit_altman(j[is_tr], y[is_tr], ["X1", "X2", "X3", "X4"])[0] \
            .predict_proba(Xall)[:, 1]
        p_trval = K.refit_altman(j[is_trval], y[is_trval], ["X1", "X2", "X3", "X4"])[0] \
            .predict_proba(Xall)[:, 1]

        # Is the drop real?  (published vs the train-only refit)
        cb = M.cluster_bootstrap_diff(y[is_te], a[is_te], p_tr[is_te], groups,
                                      stat=M.roc_auc, n_boot=1000)
        out["roc_drop_cluster_ci_low"] = cb["ci_low"]
        out["roc_drop_cluster_ci_high"] = cb["ci_high"]
        out["roc_drop_cluster_excludes_zero"] = bool(
            np.isfinite(cb["ci_low"]) and (cb["ci_low"] > 0 or cb["ci_high"] < 0))

        # Does moving the fit two years closer to test recover any of it?
        # The flag below used to fire on the sign of a difference of two drops
        # with no interval at all, so it read True on a 5e-5 change.  It now
        # requires the recovery to be both materially large and distinguishable
        # from zero under a firm-clustered interval.
        rec = M.cluster_bootstrap_diff(y[is_te], p_tr[is_te], p_trval[is_te], groups,
                                       stat=M.roc_auc, n_boot=1000)
        recovery = rec["diff"]                     # roc(train+val) - roc(train only)
        drop = abs(out["roc_drop_train_only"])
        out["roc_recovery_from_refitting_on_train_plus_val"] = recovery
        out["recovery_cluster_ci_low"] = rec["ci_low"]
        out["recovery_cluster_ci_high"] = rec["ci_high"]
        out["recovery_share_of_drop"] = recovery / drop if drop > 1e-9 else np.nan
        out["recovery_excludes_zero"] = bool(
            np.isfinite(rec["ci_low"]) and rec["ci_low"] > 0)
        out["shift_explains_drop"] = bool(
            out["recovery_excludes_zero"]
            and np.isfinite(out["recovery_share_of_drop"])
            and out["recovery_share_of_drop"] >= MIN_RECOVERY_SHARE)
        out["shift_verdict"] = (
            "shift explains part of the drop" if out["shift_explains_drop"]
            else f"shift does not explain the drop (recovers "
                 f"{out['recovery_share_of_drop'] * 100:.1f}% of it, "
                 f"95% CI [{rec['ci_low']:+.5f}, {rec['ci_high']:+.5f}])")
        rows.append(out)
    return pd.DataFrame(rows)


def write_summary(shares: pd.DataFrame, chk: pd.DataFrame,
                  tables: dict[int, pd.DataFrame],
                  extended: dict[int, pd.DataFrame] | None = None,
                  sig: pd.DataFrame | None = None,
                  drift: pd.DataFrame | None = None,
                  trees: pd.DataFrame | None = None,
                  tree_tests: pd.DataFrame | None = None) -> None:
    L = [
        "# Decomposition summary", "",
        "Every model here scores the identical row set.  **A** applies Altman's published",
        "Z″ coefficients to the window's end quarter; **B_levels** re-estimates those",
        "coefficients on train; **B_tensor** refits them on the tensor's own z-scored",
        "quarterly ratios; **B_mlp_t0** makes that nonlinear but still static;",
        "**C_lstm** adds the eight-quarter time axis; **D_lstm** adds the other 24 ratios.",
        "",
        "The ladder walks one change at a time on purpose.  The earlier B→C step moved",
        "four things at once — linear to nonlinear, static to sequential, annualised",
        "levels to z-scored quarterly ratios, and one fit to a five-seed ensemble — and",
        "credited all of it to the time axis.", "",
        "**Intervals are firm-clustered bootstraps.**  Stride-1 windowing gives one firm",
        "up to ~50 overlapping windows sharing seven of eight quarters and one event, so a",
        "window-level interval is too narrow.  DeLong and the window-level bootstrap are",
        "carried in `decomposition_all.csv` under `window_level_` names for comparison.",
        "",
        "Shares are of the **total absolute movement** across the steps, not of the net",
        "A→D gap: a step moves backwards, and dividing by a small net total would report",
        "an ordinary step as several hundred percent.", "",
    ]
    for _, r in shares.iterrows():
        h = int(r["horizon"])
        tab = tables[h]
        L += [f"## Horizon h = {h}", "", "> " + headline(r.to_dict(), tab), ""]
        t = tab.set_index("step")
        rows = []
        for step, cause in CAUSE.items():
            key = step.replace("->", " -> ")
            if key not in t.index or f"pr_auc_gap_{step}" not in r:
                continue
            rows.append({
                "step": step, "cause": cause,
                "PR-AUC change": r[f"pr_auc_gap_{step}"],
                "PR-AUC cluster 95% CI": (f"[{t.loc[key, 'pr_auc_cluster_ci_low']:+.4f}, "
                                          f"{t.loc[key, 'pr_auc_cluster_ci_high']:+.4f}]"),
                "excludes 0": bool(t.loc[key, "pr_auc_cluster_excludes_zero"]),
                "share of movement": r[f"pr_auc_share_{step}"],
                "ROC-AUC change": r[f"roc_auc_gap_{step}"],
                "ROC-AUC cluster 95% CI": (f"[{t.loc[key, 'roc_auc_cluster_ci_low']:+.4f}, "
                                           f"{t.loc[key, 'roc_auc_cluster_ci_high']:+.4f}]"),
                "ROC excludes 0": bool(t.loc[key, "roc_auc_cluster_excludes_zero"]),
                "DeLong p (window-level)": t.loc[key, "delong_window_level_p"],
                "McNemar p (window-level)": t.loc[key, "mcnemar_window_level_p"],
            })
        L += [U.to_markdown(pd.DataFrame(rows)), ""]

    if sig is not None and len(sig):
        L += ["## Significance, stated per horizon", "",
              "Each step is reported at each horizon separately.  A step that clears zero",
              "at h=4 need not clear it at h=1, where there are 60 test positives rather",
              "than 363, and the table below does not generalise across horizons.", ""]
        for h in sorted(sig["horizon"].unique()):
            L.append(f"**h = {h}**")
            L.append("")
            for _, row in sig[sig["horizon"] == h].iterrows():
                L.append(f"- {row['statement']}")
            L.append("")

    if drift is not None and len(drift):
        L += ["## Is the A→B ROC-AUC drop stale coefficients or distribution shift?", "",
              "Re-estimating Altman's coefficients leaves PR-AUC flat but *lowers* ROC-AUC",
              "at the longer horizons, so 'worth nothing' understates it: on the ranking",
              "metric the refit is actively worse.  One explanation is that the training",
              "period is no longer representative of the test period.  Refitting on",
              "train + val — two years closer to the test window, and still strictly before",
              "it — tests that: if shift were the cause, the drop should shrink.  This is a",
              "diagnostic only; nothing fitted on val scores a headline number.", "",
              U.to_markdown(drift[["horizon", "published_roc_auc", "train_only_roc_auc",
                                   "train_plus_val_roc_auc", "roc_drop_train_only",
                                   "roc_drop_cluster_ci_low", "roc_drop_cluster_ci_high",
                                   "roc_recovery_from_refitting_on_train_plus_val",
                                   "recovery_cluster_ci_low", "recovery_cluster_ci_high",
                                   "recovery_share_of_drop", "shift_explains_drop"]],
                           floatfmt="%.5f"), "",
              "Verdict per horizon:", ""]
        L += [f"- h={int(r.horizon)}: {r.shift_verdict}" for r in drift.itertuples()]
        L += [""]

    if trees is not None and len(trees):
        L += ["## The same question without a neural network", "",
              "If the time axis is what pays, it should pay for a gradient-boosted tree",
              "too.  B_xgb_t0 sees Altman's five ratios at t-0; C_xgb sees the same five",
              "across all eight quarters (40 inputs).  Same tuning budget, same row set.",
              "", U.to_markdown(trees[["name", "horizon", "n_inputs", "roc_auc", "pr_auc",
                                       "recall", "note"]]), ""]

    L += ["## Is the finding LSTM-specific?", "",
          "Model D is repeated with the Transformer.  Where the LSTM loses ground on the",
          "full 29-ratio feature set, the Transformer recovers part of it, so C→D is",
          "partly an LSTM capacity limitation rather than a pure statement about the",
          "feature set.  Both are in `decomposition_all.csv`.", ""]
    tr = [{"horizon": h,
           "C_lstm PR-AUC": tab.set_index("model").loc["C_lstm", "pr_auc"],
           "D_lstm PR-AUC": tab.set_index("model").loc["D_lstm", "pr_auc"],
           "D_transformer PR-AUC": tab.set_index("model").loc["D_transformer", "pr_auc"],
           "C_lstm ROC-AUC": tab.set_index("model").loc["C_lstm", "roc_auc"],
           "D_lstm ROC-AUC": tab.set_index("model").loc["D_lstm", "roc_auc"],
           "D_transformer ROC-AUC": tab.set_index("model").loc["D_transformer", "roc_auc"]}
          for h, tab in sorted((extended or {}).items())
          if "D_transformer" in set(tab["model"])]
    if tr:
        L += [U.to_markdown(pd.DataFrame(tr)), ""]

    L += ["## Missingness sanity check", "",
          "Models C and D retrained on the subset of windows where all five Altman ratios",
          "are observed in all eight quarters.  If the gaps were a missingness artefact",
          "they would close here.", "",
          U.to_markdown(chk), ""]
    (C.RESULTS / "decomposition_summary.md").write_text("\n".join(L), encoding="utf-8")
