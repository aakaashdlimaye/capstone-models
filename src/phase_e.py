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

LADDER = ["A_zdp", "B_levels", "B_tensor", "C_lstm", "D_lstm", "D_transformer"]
PRIMARY_LADDER = ["A_zdp", "B_levels", "C_lstm", "D_lstm"]


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
                      ("B_tensor", f"decompB_tensor_{horizon}")):
        te, va = E.load_preds(key, "test"), E.load_preds(key, "val")
        out[name] = {"y": te["y_true"].to_numpy().astype(int), "p": te["p_hat"].to_numpy(),
                     "yv": va["y_true"].to_numpy().astype(int), "pv": va["p_hat"].to_numpy()}
    for name, kf in (
        ("C_lstm", lambda s: E.RunSpec(arch="lstm", horizon=horizon,
                                       treatment="class_weight", seed=s, tag="decompC").key),
        ("D_lstm", lambda s: E.RunSpec(arch="lstm", horizon=horizon,
                                       treatment="class_weight", seed=s).key),
        ("D_transformer", lambda s: E.RunSpec(arch="transformer", horizon=horizon,
                                              treatment="class_weight", seed=s).key),
    ):
        y, p, _ = seed_mean_pred(kf, seeds, "test")
        yv, pv, _ = seed_mean_pred(kf, seeds, "val")
        out[name] = {"y": y, "p": p, "yv": yv, "pv": pv, "per_seed": [kf(s) for s in seeds]}
    return out


# --------------------------------------------------------------------------
def gaps(preds: dict, horizon: int, ladder=PRIMARY_LADDER) -> pd.DataFrame:
    """Each rung's metrics plus the DeLong/McNemar test against the rung below."""
    rows = []
    for i, name in enumerate(ladder):
        d = preds[name]
        thr, _ = M.best_f1_threshold(d["yv"], d["pv"])
        r = {"model": name, "horizon": horizon,
             "roc_auc": M.roc_auc(d["y"], d["p"]), "pr_auc": M.pr_auc(d["y"], d["p"])}
        r.update({k: v for k, v in M.threshold_metrics(d["y"], d["p"], thr).items()
                  if k in ("threshold", "f1", "recall", "specificity", "precision")})
        if i > 0:
            prev = preds[ladder[i - 1]]
            dl = M.delong_test(d["y"], prev["p"], d["p"])
            thr_prev, _ = M.best_f1_threshold(prev["yv"], prev["pv"])
            mc = M.mcnemar_test(d["y"], prev["p"], d["p"], thr_prev, thr)
            bs = M.bootstrap_diff(d["y"], prev["p"], d["p"], stat=M.pr_auc, n_boot=2000)
            r.update({
                "step": f"{ladder[i - 1]} -> {name}",
                "delong_roc_diff": dl["diff"], "delong_ci_low": dl["ci_low"],
                "delong_ci_high": dl["ci_high"], "delong_p": dl["p_value"],
                "pr_auc_diff": bs["diff"], "pr_ci_low": bs["ci_low"],
                "pr_ci_high": bs["ci_high"], "pr_boot_p": bs["p_value"],
                "mcnemar_b": mc["b"], "mcnemar_c": mc["c"],
                "mcnemar_p": mc["p_value"], "mcnemar_test": mc["test"],
            })
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


CAUSE = {
    "A_zdp->B_levels": "coefficient drift",
    "B_levels->C_lstm": "the static formulation",
    "C_lstm->D_lstm": "the feature-set expansion",
}


def headline(shares: dict, tab: pd.DataFrame, ladder=PRIMARY_LADDER,
             metric: str = "pr_auc") -> str:
    """The paper's one-sentence finding, with its numbers and intervals."""
    parts = []
    for i in range(1, len(ladder)):
        step = f"{ladder[i - 1]}->{ladder[i]}"
        parts.append(f"{shares[f'{metric}_share_{step}'] * 100:.0f}% is "
                     f"{CAUSE.get(step, step)} ({shares[f'{metric}_gap_{step}']:+.4f}, "
                     f"{shares[f'{metric}_direction_{step}']})")
    t = tab.set_index("step")
    drift_step = f"{ladder[0]} -> {ladder[1]}"
    drift = ""
    if drift_step in t.index:
        drift = (f"  Re-estimating Altman's coefficients on modern training data is worth "
                 f"{shares[f'{metric}_gap_' + ladder[0] + '->' + ladder[1]]:+.4f} PR-AUC, "
                 f"95% CI [{t.loc[drift_step, 'pr_ci_low']:+.4f}, "
                 f"{t.loc[drift_step, 'pr_ci_high']:+.4f}].")
    return (f"At h={shares['horizon']}, test PR-AUC moves from {shares[f'{metric}_A']:.4f} for "
            f"Altman Z″ to {shares[f'{metric}_D']:.4f} for the full temporal model, a net "
            f"{shares[f'{metric}_net_gap']:+.4f}.  Of the "
            f"{shares[f'{metric}_total_abs_movement']:.4f} of total movement across the three "
            f"steps, " + ", ".join(parts) + "." + drift)


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

    tables, primary, extended, shares = [], {}, {}, []
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
        # The finding must not be LSTM-specific.
        alt = gaps(preds, h, ladder=["A_zdp", "B_levels", "C_lstm", "D_transformer"])
        shares[-1].update({f"transformer_{k}": v for k, v in
                           gap_shares(alt, ladder=["A_zdp", "B_levels", "C_lstm",
                                                   "D_transformer"]).items()})
    pd.concat(tables, ignore_index=True).to_csv(C.RESULTS / "decomposition_all.csv", index=False)
    sh = pd.DataFrame(shares)
    U.write_table(sh, C.RESULTS / "decomposition_shares", floatfmt="%.4f")

    chk = complete_subset_check(splits, seeds=seeds, force=force)
    U.write_table(chk, C.RESULTS / "decomposition_complete_subset")

    write_summary(sh, chk, primary, extended)
    return {"horizons": list(horizons)}


def write_summary(shares: pd.DataFrame, chk: pd.DataFrame,
                  tables: dict[int, pd.DataFrame],
                  extended: dict[int, pd.DataFrame] | None = None) -> None:
    L = [
        "# Decomposition summary", "",
        "Four models on the identical row set.  **A** applies Altman's published Z″",
        "coefficients to the window's end quarter; **B** re-estimates those coefficients",
        "on train; **C** gives the same variables an 8-quarter LSTM; **D** gives the LSTM",
        "all 29 ratios.  A→B isolates coefficient drift, B→C the static formulation,",
        "C→D the feature set.", "",
        "Shares are of the **total absolute movement** across the three steps, not of the",
        "net A→D gap.  One step moves backwards, and dividing by a small net total would",
        "report an ordinary step as several hundred percent.  Each step's signed change",
        "and its 95% interval sit beside its share.", "",
    ]
    for _, r in shares.iterrows():
        h = int(r["horizon"])
        tab = tables[h]
        L += [f"## Horizon h = {h}", "", "> " + headline(r.to_dict(), tab), ""]
        t = tab.set_index("step")
        rows = []
        for step in ("A_zdp->B_levels", "B_levels->C_lstm", "C_lstm->D_lstm"):
            key = step.replace("->", " -> ")
            has = key in t.index
            rows.append({
                "step": step, "cause": CAUSE.get(step, ""),
                "PR-AUC change": r[f"pr_auc_gap_{step}"],
                "PR-AUC 95% CI": (f"[{t.loc[key, 'pr_ci_low']:+.4f}, "
                                  f"{t.loc[key, 'pr_ci_high']:+.4f}]") if has else "",
                "share of movement": r[f"pr_auc_share_{step}"],
                "direction": r[f"pr_auc_direction_{step}"],
                "ROC-AUC change": r[f"roc_auc_gap_{step}"],
                "DeLong 95% CI": (f"[{t.loc[key, 'delong_ci_low']:+.4f}, "
                                  f"{t.loc[key, 'delong_ci_high']:+.4f}]") if has else "",
                "DeLong p": t.loc[key, "delong_p"] if has else np.nan,
                "McNemar p": t.loc[key, "mcnemar_p"] if has else np.nan,
            })
        L += [U.to_markdown(pd.DataFrame(rows)), ""]

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
