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


def heatmap(table: pd.DataFrame, horizon: int, out_stem) -> None:
    import matplotlib.pyplot as plt

    sub = table[table["horizon"] == horizon]
    treatments = [t for t in C.TREATMENTS] + \
                 [f"cost_threshold_1to{r}" for r in C.COST_RATIOS]
    treatments = [t for t in treatments if t in set(sub["treatment"])]
    archs = list(C.ARCHITECTURES)
    Z = np.full((len(archs), len(treatments)), np.nan)
    for i, a in enumerate(archs):
        for j, t in enumerate(treatments):
            v = sub.query("arch == @a and treatment == @t")["pr_auc_mean"]
            if len(v):
                Z[i, j] = float(v.iloc[0])

    fig, ax = plt.subplots(figsize=(1.25 * len(treatments) + 3, 0.75 * len(archs) + 2.2))
    im = ax.imshow(Z, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(treatments)), treatments, rotation=35, ha="right")
    ax.set_yticks(range(len(archs)), archs)
    for i in range(len(archs)):
        for j in range(len(treatments)):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center",
                        color="white" if Z[i, j] < np.nanmax(Z) * 0.6 else "black", fontsize=9)
    ax.set_title(f"Test PR-AUC by architecture and imbalance treatment (h={horizon}, "
                 f"mean of {len(C.SEEDS)} seeds)")
    fig.colorbar(im, ax=ax, label="PR-AUC")
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
                   arch: str = "lstm") -> pd.DataFrame:
    """Three protocols, one architecture, side by side."""
    hp = Tune.best_hparams(arch)
    Xp, Yp, SPp = _pooled(splits)
    rows = []

    for h in horizons:
        yp = Yp[:, h - 1].astype(int)

        for seed in seeds:
            # ---- Inflated: SMOTE on everything, then a random 80/20 split ----
            Xr, yr, parent, st = IMB.smote_with_parents(Xp, yp, seed=seed)
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(yr))
            cut = int(0.8 * len(yr))
            tr_i, te_i = perm[:cut], perm[cut:]
            va_i = tr_i[: max(1, len(tr_i) // 10)]     # a slice of train, as such papers do
            res = T.train_one(arch, Xr[tr_i], yr[tr_i], Xr[va_i], yr[va_i],
                              Xr[te_i], yr[te_i], treatment="none", seed=seed, hp=hp,
                              horizon=h)
            rows.append(_audit_row("inflated", h, seed, arch, yr[te_i], res.test_pred, st,
                                   "random 80/20 split of all windows; SMOTE applied "
                                   "before splitting; accuracy at 0.5"))

            # ---- Half-fixed: chronological split, but SMOTE before splitting ----
            sp_r = np.concatenate([SPp, SPp[parent[len(SPp):]]])
            m_tr = sp_r == "train"; m_va = sp_r == "val"; m_te = sp_r == "test"
            res = T.train_one(arch, Xr[m_tr], yr[m_tr], Xr[m_va], yr[m_va],
                              Xr[m_te], yr[m_te], treatment="none", seed=seed, hp=hp,
                              horizon=h)
            rows.append(_audit_row("half_fixed", h, seed, arch, yr[m_te], res.test_pred, st,
                                   "chronological split, but SMOTE applied before "
                                   "splitting, so synthetic positives cross the boundary"))

            # ---- Correct: the inherited split, SMOTE inside train only ----
            key = E.RunSpec(arch=arch, horizon=h, treatment="smote", seed=seed).key
            te = E.load_preds(key, "test"); va = E.load_preds(key, "val")
            thr, _ = M.best_f1_threshold(va["y_true"].to_numpy(), va["p_hat"].to_numpy())
            rows.append(_audit_row("correct", h, seed, arch, te["y_true"].to_numpy(),
                                   te["p_hat"].to_numpy(), {"applied": True},
                                   "inherited chronological split; SMOTE fitted and "
                                   "applied inside train only; threshold chosen on val",
                                   thr=thr))
    return pd.DataFrame(rows)


def _audit_row(protocol, h, seed, arch, y, p, st, description, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).ravel()
    m = M.threshold_metrics(y, p, thr)
    return {
        "protocol": protocol, "horizon": h, "seed": seed, "arch": arch,
        "n_test": len(y), "test_positive_rate": float(y.mean()),
        "threshold": thr,
        "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
        "f1": m["f1"], "recall": m["recall"], "specificity": m["specificity"],
        "precision": m["precision"],
        "roc_auc": M.roc_auc(y, p), "pr_auc": M.pr_auc(y, p),
        "n_synthetic": st.get("n_synthetic", 0),
        "description": description,
    }


def audit_summary(audit: pd.DataFrame) -> pd.DataFrame:
    agg = (audit.groupby(["horizon", "protocol"])
           .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                recall_mean=("recall", "mean"), f1_mean=("f1", "mean"),
                test_positive_rate=("test_positive_rate", "mean"),
                n_test=("n_test", "mean"), n_seeds=("seed", "count")).reset_index())
    order = {"inflated": 0, "half_fixed": 1, "correct": 2}
    agg["_o"] = agg["protocol"].map(order)
    agg = agg.sort_values(["horizon", "_o"]).drop(columns="_o").reset_index(drop=True)

    # How much of the accuracy drop does each fix account for?
    out = []
    for h, g in agg.groupby("horizon"):
        g = g.set_index("protocol")
        a_inf = g.loc["inflated", "accuracy_mean"]
        a_half = g.loc["half_fixed", "accuracy_mean"]
        a_cor = g.loc["correct", "accuracy_mean"]
        total = a_inf - a_cor
        out.append({"horizon": h, "inflated_accuracy": a_inf,
                    "half_fixed_accuracy": a_half, "correct_accuracy": a_cor,
                    "drop_from_chronological_split": a_inf - a_half,
                    "drop_from_resampling_inside_train": a_half - a_cor,
                    "total_drop": total,
                    "share_chronological": (a_inf - a_half) / total if total else np.nan,
                    "share_resampling": (a_half - a_cor) / total if total else np.nan,
                    "correct_pr_auc": g.loc["correct", "pr_auc_mean"],
                    "inflated_pr_auc": g.loc["inflated", "pr_auc_mean"]})
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
        heatmap(table, h, C.FIGURES / f"imbalance_heatmap_h{h}")
    table.to_csv(C.RESULTS / "imbalance_ablation_all.csv", index=False)

    audit = protocol_audit(splits, horizons=horizons, seeds=seeds)
    audit.to_csv(C.RESULTS / "protocol_audit_runs.csv", index=False)
    agg, decomp = audit_summary(audit)
    U.write_table(agg, C.RESULTS / "protocol_audit")
    U.write_table(decomp, C.RESULTS / "protocol_audit_decomposition")
    print(agg.to_string(index=False))

    inflated = agg.query("protocol == 'inflated'")["accuracy_mean"].max()
    ok = bool(inflated > 0.95)
    print(f"[phase D] inflated accuracy = {inflated:.4f}; GATE {'PASS' if ok else 'FAIL'}")
    return {"gate_pass": ok, "inflated_accuracy": float(inflated)}
