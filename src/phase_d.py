"""Phase D — the imbalance ablation and the protocol audit.

The ablation is a full factorial: four architectures x five treatments, at the
two extreme horizons, five seeds each.  Four of the treatments change training;
the fifth (cost-sensitive thresholding) changes only the decision rule, so it is
applied on top of the untreated model to show what a better threshold alone buys.

The protocol audit is the engine of Contribution 5.  It runs the same LSTM three
ways and puts the numbers side by side, so the "91-99% accuracy" the literature
reports can be traced to specific methodological choices rather than asserted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import experiment as E
from . import imbalance as IMB
from . import metrics as M
from . import models as Models
from . import train as T
from . import tuning as Tune
from . import utils as U

AUDIT_HORIZONS = (1, 4)


# --------------------------------------------------------------------------
# The factorial ablation
# --------------------------------------------------------------------------
def run_ablation(splits, horizons=AUDIT_HORIZONS, seeds=C.SEEDS, force: bool = False):
    records, scores = [], []
    for h in horizons:
        b = E.bundle(splits, horizon=h)
        for arch in C.ARCHITECTURES:
            hp = Tune.best_hparams(arch)
            for treatment in C.TREATMENTS:
                for seed in seeds:
                    spec = E.RunSpec(arch=arch, horizon=h, treatment=treatment,
                                     seed=seed, hp=hp)
                    rec = E.run_deep(spec, b, force=force)
                    records.append(rec)
                    s = E.score(spec.key)
                    s.update({"arch": arch, "horizon": h, "treatment": treatment, "seed": seed})
                    scores.append(s)
                    print(f"[phase D] {spec.key:42s} PR-AUC {s['pr_auc']:.4f} "
                          f"cost@20 {s['cost_1to20']:.4f} ({rec['seconds']:.0f}s)", flush=True)

                # Treatment 5: same untreated model, cost-minimising threshold.
                if treatment == "none":
                    for r in C.COST_RATIOS:
                        for seed in seeds:
                            key = E.RunSpec(arch=arch, horizon=h, treatment="none",
                                            seed=seed).key
                            s = E.score(key, thr_rule="cost", fn_cost=float(r))
                            s.update({"arch": arch, "horizon": h,
                                      "treatment": f"cost_threshold_1to{r}", "seed": seed})
                            scores.append(s)
    return records, scores


def ablation_table(scores) -> pd.DataFrame:
    df = pd.DataFrame(scores)
    metrics = ["roc_auc", "pr_auc", "f1", "recall", "specificity", "precision", "accuracy"]
    cost_cols = [c for c in df.columns if c.startswith("cost_1to")]
    agg = {}
    for m in metrics + cost_cols:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    agg["n_seeds"] = ("seed", "count")
    return df.groupby(["horizon", "arch", "treatment"], dropna=False).agg(**agg).reset_index()


def heatmap(table: pd.DataFrame, horizon: int, out_stem, metric: str = "pr_auc_mean",
            treatments: list[str] | None = None, label: str | None = None,
            lower_is_better: bool = False) -> None:
    """architecture x treatment grid for one metric."""
    import matplotlib.pyplot as plt

    sub = table[table["horizon"] == horizon]
    treatments = [t for t in (treatments or list(C.TREATMENTS)) if t in set(sub["treatment"])]
    archs = list(C.ARCHITECTURES)
    Z = np.full((len(archs), len(treatments)), np.nan)
    for i, a in enumerate(archs):
        for j, t in enumerate(treatments):
            v = sub.query("arch == @a and treatment == @t")[metric]
            if len(v):
                Z[i, j] = float(v.iloc[0])

    fig, ax = plt.subplots(figsize=(1.35 * len(treatments) + 3.4, 0.8 * len(archs) + 2.4))
    im = ax.imshow(Z, cmap="viridis_r" if lower_is_better else "viridis", aspect="auto")
    ax.set_xticks(range(len(treatments)), treatments, rotation=35, ha="right")
    ax.set_yticks(range(len(archs)), archs)
    lo, hi = np.nanmin(Z), np.nanmax(Z)
    mid = lo + 0.6 * (hi - lo) if hi > lo else hi
    for i in range(len(archs)):
        for j in range(len(treatments)):
            if np.isfinite(Z[i, j]):
                dark = (Z[i, j] > mid) if lower_is_better else (Z[i, j] < mid)
                ax.text(j, i, f"{Z[i, j]:.4f}", ha="center", va="center",
                        color="white" if dark else "black", fontsize=8)
    name = label or metric
    ax.set_title(f"Test {name} by architecture and imbalance treatment\n"
                 f"h={horizon}, mean of {len(C.SEEDS)} seeds"
                 + ("  (lower is better)" if lower_is_better else ""))
    fig.colorbar(im, ax=ax, label=name)
    U.savefig(fig, out_stem)


# --------------------------------------------------------------------------
# The protocol audit
# --------------------------------------------------------------------------
def _pooled(splits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All windows in one array, with each row's real split label."""
    order = ("train", "val", "test")
    X = np.concatenate([splits[s].X for s in order], axis=0)
    Y = np.concatenate([splits[s].y for s in order], axis=0)
    sp = np.concatenate([np.full(splits[s].n, s) for s in order])
    return X, Y, sp


def protocol_audit(splits, horizons=AUDIT_HORIZONS, seeds=C.SEEDS,
                   arch: str = "lstm", force: bool = False) -> pd.DataFrame:
    """Three protocols, one architecture, side by side.

    These runs deliberately break the protocol, so they do not go through
    `run_deep` and leave no run record -- nothing they produce can reach a
    reported table by accident.  They get their own result cache instead.
    """
    cache = C.RESULTS / "protocol_audit_runs.csv"
    need = {"pr_auc_lift", "scored_on"}
    if cache.exists() and not force:
        got = pd.read_csv(cache)
        if need <= set(got.columns):
            print(f"[phase D] reusing {cache.name}")
            return got
        print(f"[phase D] {cache.name} predates {sorted(need)}; re-scoring")

    hp = Tune.best_hparams(arch)
    Xp, Yp, SPp = _pooled(splits)
    rows = []

    for h in horizons:
        yp = Yp[:, h - 1].astype(int)

        n_orig = len(yp)                       # rows 0..n_orig-1 are real windows

        for seed in seeds:
            Xr, yr, parent, st = IMB.smote_with_parents(Xp, yp, seed=seed)

            # ---- Inflated: SMOTE on everything, then a random 80/20 split ----
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(yr))
            cut = int(0.8 * len(yr))
            tr_i, te_i = perm[:cut], perm[cut:]
            va_i = tr_i[: max(1, len(tr_i) // 10)]     # a slice of train, as such papers do
            y_te, p_te = _protocol_predictions(
                "inflated", h, seed, arch, hp, Xr, yr, tr_i, va_i, te_i, force=force)
            rows += _audit_rows(
                "inflated", h, seed, arch, y_te, p_te, st,
                "random 80/20 split of all windows; SMOTE applied before splitting; "
                "accuracy at 0.5",
                natural=te_i < n_orig)

            # ---- Half-fixed: chronological split, but SMOTE before splitting ----
            sp_r = np.concatenate([SPp, SPp[parent[len(SPp):]]])
            tr_j = np.flatnonzero(sp_r == "train")
            va_j = np.flatnonzero(sp_r == "val")
            te_j = np.flatnonzero(sp_r == "test")
            y_te, p_te = _protocol_predictions(
                "half_fixed", h, seed, arch, hp, Xr, yr, tr_j, va_j, te_j, force=force)
            rows += _audit_rows(
                "half_fixed", h, seed, arch, y_te, p_te, st,
                "chronological split, but SMOTE applied before splitting, so synthetic "
                "positives cross the boundary",
                natural=te_j < n_orig)

            # ---- Correct: the inherited split, SMOTE inside train only ----
            key = E.RunSpec(arch=arch, horizon=h, treatment="smote", seed=seed).key
            te = E.load_preds(key, "test"); va = E.load_preds(key, "val")
            thr, _ = M.best_f1_threshold(va["y_true"].to_numpy(), va["p_hat"].to_numpy())
            rows += _audit_rows(
                "correct", h, seed, arch, te["y_true"].to_numpy(),
                te["p_hat"].to_numpy(), {"applied": True},
                "inherited chronological split; SMOTE fitted and applied inside train "
                "only; threshold chosen on val",
                thr=thr, natural=None)
    return pd.DataFrame(rows)


PROTOCOL_PREDS = C.RESULTS / "preds_protocol"


def _protocol_predictions(protocol: str, h: int, seed: int, arch: str, hp,
                          Xr, yr, tr_i, va_i, te_i, force: bool = False):
    """Train (or reuse) one protocol run and return its test labels and scores.

    These runs deliberately break the protocol, so they never go through
    `run_deep` and leave no run record -- nothing they produce can reach a
    reported table by accident.  They cache their *predictions* here instead, so
    that changing how the audit is scored (as the base-rate correction did) is a
    re-read rather than another two hours of training.
    """
    PROTOCOL_PREDS.mkdir(parents=True, exist_ok=True)
    path = PROTOCOL_PREDS / f"{protocol}_{h}_{seed}.parquet"
    if path.exists() and not force:
        df = pd.read_parquet(path)
        return df["y_true"].to_numpy().astype(int), df["p_hat"].to_numpy()

    res = T.train_one(arch, Xr[tr_i], yr[tr_i], Xr[va_i], yr[va_i],
                      Xr[te_i], yr[te_i], treatment="none", seed=seed, hp=hp, horizon=h)
    y_te = np.asarray(yr[te_i]).astype(np.int8)
    p_te = np.asarray(res.test_pred).ravel().astype(np.float32)
    pd.DataFrame({"resampled_row": np.asarray(te_i, dtype=np.int64),
                  "y_true": y_te, "p_hat": p_te}).to_parquet(path, index=False)
    print(f"[phase D] protocol {protocol} h={h} seed={seed} trained "
          f"({res.seconds:.0f}s, {len(te_i):,} test rows)", flush=True)
    return y_te.astype(int), p_te


def _audit_rows(protocol, h, seed, arch, y, p, st, description,
                thr: float = 0.5, natural: np.ndarray | None = None) -> list[dict]:
    """One row on the test set as the protocol leaves it, one on real rows only.

    The inflated and half-fixed protocols score a SMOTE-balanced test set at
    roughly a 50% positive rate; the correct protocol scores the natural test
    set at 0.19%.  PR-AUC is not comparable across those, because it is bounded
    below by the base rate.  Every row therefore carries `pr_auc_lift`
    (PR-AUC / base rate), which is comparable, and every resampled protocol also
    gets a `*_natural_test` twin: the *same model and the same predictions*,
    scored only on the test rows that are real windows rather than interpolated
    ones.  That twin isolates leakage from base-rate inflation, because it
    differs from the correct protocol only in how the model was trained.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).ravel()

    def one(tag, yy, pp, note, scored_on):
        m = M.threshold_metrics(yy, pp, thr)
        base = float(yy.mean())
        pra = M.pr_auc(yy, pp)
        return {
            "protocol": tag, "horizon": h, "seed": seed, "arch": arch,
            "scored_on": scored_on,
            "n_test": len(yy), "test_positive_rate": base, "threshold": thr,
            "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "f1": m["f1"], "recall": m["recall"], "specificity": m["specificity"],
            "precision": m["precision"],
            "roc_auc": M.roc_auc(yy, pp), "pr_auc": pra,
            "pr_auc_lift": pra / base if base > 0 else float("nan"),
            "n_synthetic": st.get("n_synthetic", 0),
            "description": note,
        }

    # The correct protocol never resamples its test set, so its rows are real by
    # construction; the resampled protocols get both views.
    out = [one(protocol, y, p, description,
               "the natural test set" if natural is None
               else "test set as the protocol leaves it")]
    if natural is not None:
        nat = np.asarray(natural, dtype=bool)
        if nat.sum() > 0 and 0 < y[nat].sum() < nat.sum():
            out.append(one(f"{protocol}_natural_test", y[nat], p[nat],
                           description + "; scored only on the real (non-synthetic) "
                                         "test windows, at the natural base rate",
                           "real windows only"))
    return out


PROTOCOL_ORDER = {"inflated": 0, "inflated_natural_test": 1, "half_fixed": 2,
                  "half_fixed_natural_test": 3, "correct": 4}


def audit_summary(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = (audit.groupby(["horizon", "protocol"])
           .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                pr_auc_lift_mean=("pr_auc_lift", "mean"),
                pr_auc_lift_std=("pr_auc_lift", "std"),
                roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                recall_mean=("recall", "mean"), f1_mean=("f1", "mean"),
                test_positive_rate=("test_positive_rate", "mean"),
                n_test=("n_test", "mean"), n_seeds=("seed", "count")).reset_index())
    agg["_o"] = agg["protocol"].map(PROTOCOL_ORDER)
    agg = agg.sort_values(["horizon", "_o"]).drop(columns="_o").reset_index(drop=True)

    # The attribution runs on ROC-AUC.  PR-AUC is bounded below by the base
    # rate, and the inflated and half-fixed protocols score a SMOTE-balanced
    # test set at ~50% positive against the correct protocol's 0.19%, so a
    # PR-AUC difference across them measures the base rate as much as the
    # leakage.  ROC-AUC is invariant to the base rate and carries the
    # attribution; PR-AUC is attributed only across the natural-base-rate
    # variants, where the comparison is like for like.  Accuracy carries
    # neither: it is high under the inflated protocol because the model
    # separates a balanced set and high again under the correct one because the
    # majority class is 99% of it, so its two large moves cancel.
    out = []
    for h, g in agg.groupby("horizon"):
        g = g.set_index("protocol")
        base = g.loc["correct", "test_positive_rate"]

        def share(a, b, c):
            d1, d2 = a - b, b - c
            t = abs(d1) + abs(d2)
            return d1, d2, (abs(d1) / t if t else np.nan), (abs(d2) / t if t else np.nan)

        r_inf, r_half, r_cor = (g.loc[k, "roc_auc_mean"]
                                for k in ("inflated", "half_fixed", "correct"))
        dr_split, dr_res, sh_split, sh_res = share(r_inf, r_half, r_cor)

        row = {
            "horizon": h,
            # --- primary: ROC-AUC, base-rate invariant --------------------
            "inflated_roc_auc": r_inf, "half_fixed_roc_auc": r_half,
            "correct_roc_auc": r_cor,
            "roc_auc_lost_to_chronological_split": dr_split,
            "roc_auc_lost_to_resampling_inside_train": dr_res,
            "total_roc_auc_collapse": r_inf - r_cor,
            "share_chronological_split": sh_split,
            "share_resampling_inside_train": sh_res,
            # --- PR-AUC, comparable only as a lift over the base rate -----
            "inflated_pr_auc": g.loc["inflated", "pr_auc_mean"],
            "half_fixed_pr_auc": g.loc["half_fixed", "pr_auc_mean"],
            "correct_pr_auc": g.loc["correct", "pr_auc_mean"],
            "inflated_pr_auc_lift": g.loc["inflated", "pr_auc_lift_mean"],
            "half_fixed_pr_auc_lift": g.loc["half_fixed", "pr_auc_lift_mean"],
            "correct_pr_auc_lift": g.loc["correct", "pr_auc_lift_mean"],
            "inflated_test_base_rate": g.loc["inflated", "test_positive_rate"],
            "correct_test_base_rate": base,
            # --- the diagnostic accuracy cannot give -----------------------
            "inflated_accuracy": g.loc["inflated", "accuracy_mean"],
            "half_fixed_accuracy": g.loc["half_fixed", "accuracy_mean"],
            "correct_accuracy": g.loc["correct", "accuracy_mean"],
            "majority_class_accuracy": 1.0 - base,
            "model_beats_majority_class": bool(g.loc["correct", "accuracy_mean"] > 1.0 - base),
        }

        # PR-AUC attribution, on equal base rates only.
        if {"inflated_natural_test", "half_fixed_natural_test"} <= set(g.index):
            p_inf = g.loc["inflated_natural_test", "pr_auc_mean"]
            p_half = g.loc["half_fixed_natural_test", "pr_auc_mean"]
            p_cor = g.loc["correct", "pr_auc_mean"]
            dp_split, dp_res, sp_split, sp_res = share(p_inf, p_half, p_cor)
            row.update({
                "natural_inflated_pr_auc": p_inf,
                "natural_half_fixed_pr_auc": p_half,
                "natural_inflated_base_rate": g.loc["inflated_natural_test", "test_positive_rate"],
                "pr_auc_natural_lost_to_chronological_split": dp_split,
                "pr_auc_natural_lost_to_resampling_inside_train": dp_res,
                "pr_auc_natural_share_chronological_split": sp_split,
                "pr_auc_natural_share_resampling_inside_train": sp_res,
                "natural_inflated_over_correct_pr_auc": p_inf / p_cor if p_cor else np.nan,
            })
        out.append(row)
    return agg, pd.DataFrame(out)


# --------------------------------------------------------------------------
def main(full: bool = True, seeds=C.SEEDS, horizons=AUDIT_HORIZONS,
         force: bool = False) -> dict:
    print("=" * 70)
    print("PHASE D — IMBALANCE ABLATION AND PROTOCOL AUDIT")
    print("=" * 70)
    splits = D.load_all(full=full)

    recs, scores = run_ablation(splits, horizons=horizons, seeds=seeds, force=force)
    table = ablation_table(scores)
    for h in horizons:
        sub = table[table["horizon"] == h]
        U.write_table(sub, C.RESULTS / f"imbalance_ablation_h{h}")
        # PR-AUC is threshold-free, so only the four training treatments can
        # move it; the cost-threshold rows share the untreated model's ranking
        # and differ only in where they cut it.  They get their own cost grid.
        heatmap(table, h, C.FIGURES / f"imbalance_heatmap_h{h}",
                metric="pr_auc_mean", label="PR-AUC")
        heatmap(table, h, C.FIGURES / f"imbalance_cost_heatmap_h{h}",
                metric="cost_1to20_mean", label="expected cost per window at 20:1",
                treatments=list(C.TREATMENTS)
                + [f"cost_threshold_1to{r}" for r in C.COST_RATIOS],
                lower_is_better=True)
    table.to_csv(C.RESULTS / "imbalance_ablation_all.csv", index=False)

    audit = protocol_audit(splits, horizons=horizons, seeds=seeds, force=force)
    audit.to_csv(C.RESULTS / "protocol_audit_runs.csv", index=False)
    agg, decomp = audit_summary(audit)
    U.write_table(agg, C.RESULTS / "protocol_audit")
    U.write_table(decomp, C.RESULTS / "protocol_audit_decomposition")
    print(agg.to_string(index=False))

    inflated = agg.query("protocol == 'inflated'")["accuracy_mean"].max()
    ok = bool(inflated > 0.95)
    print(f"[phase D] inflated accuracy = {inflated:.4f}; GATE {'PASS' if ok else 'FAIL'}")
    return {"gate_pass": ok, "inflated_accuracy": float(inflated)}
