"""Phase F — interpretability (Contribution 4).

Two independent readings of "which quarters carry the warning signal":

  1. attention weights, from the Transformer and the CNN-LSTM-Attention model;
  2. SHAP attributions on the best model, aggregated by ratio family and by
     time step.

Both are computed on masked data — a cell the dataset marks as imputed or
structurally undefined contributes nothing — and the two are then cross-checked
against each other with a rank correlation.  Disagreement is reported as a
finding, not smoothed over.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import data as D
from . import experiment as E
from . import metrics as M
from . import utils as U

N_BACKGROUND = 200
N_EXPLAIN_NEG = 1500


# --------------------------------------------------------------------------
# 1. Attention over quarters
# --------------------------------------------------------------------------
def step_observed(d: D.SplitData) -> np.ndarray:
    """(n, 8) — True where the quarter has at least one observed feature.

    A wholly forward-filled quarter carries no information of its own, so it is
    excluded from every attention average below.
    """
    return d.mask.sum(axis=2) > 0


def transformer_step_weight(attn: np.ndarray) -> np.ndarray:
    """(n, 8): how much attention each time step *receives*.

    `attn` is (layers, n, heads, query, key).  The classifier reads a global
    average over positions, so the quantity that matters for "which quarter
    matters" is the attention a key position receives, averaged over queries,
    heads and layers.
    """
    return attn.mean(axis=(0, 2, 3))


def cnn_step_weight(alpha: np.ndarray, window: int = C.WINDOW_LEN) -> np.ndarray:
    """(n, 4) pooled attention expanded onto the 8 original quarters.

    MaxPool1D(2) after a same-padded Conv1D makes pooled position j the pair of
    quarters (2j, 2j+1), so each pooled weight is split evenly across its two
    source quarters.  The raw four-position weights are reported alongside.
    """
    n, L = alpha.shape
    out = np.repeat(alpha, window // L, axis=1) / (window // L)
    return out


def attention_table(splits, horizon: int, seeds=C.SEEDS) -> tuple[pd.DataFrame, dict]:
    d = splits["test"]
    obs = step_observed(d)
    rows, raw = [], {}

    for arch in ("transformer", "cnn_lstm_attn"):
        per_seed = []
        for seed in seeds:
            key = E.RunSpec(arch=arch, horizon=horizon, treatment="class_weight",
                            seed=seed).key
            a = E.load_attention(key)
            if a is None:
                continue
            w = transformer_step_weight(a) if arch == "transformer" else cnn_step_weight(a)
            per_seed.append(w)
        if not per_seed:
            continue
        W = np.mean(per_seed, axis=0)                      # (n, 8), seed-averaged
        raw[arch] = W

        y = d.target(horizon).astype(int)
        p = np.mean([E.load_preds(E.RunSpec(arch=arch, horizon=horizon,
                                            treatment="class_weight", seed=s).key,
                                  "test")["p_hat"].to_numpy() for s in seeds], axis=0)
        pv = np.mean([E.load_preds(E.RunSpec(arch=arch, horizon=horizon,
                                             treatment="class_weight", seed=s).key,
                                   "val")["p_hat"].to_numpy() for s in seeds], axis=0)
        yv = E.load_preds(E.RunSpec(arch=arch, horizon=horizon, treatment="class_weight",
                                    seed=seeds[0]).key, "val")["y_true"].to_numpy()
        thr, _ = M.best_f1_threshold(yv, pv)
        pred = (p >= thr).astype(int)

        groups = {
            "true_positive": (y == 1) & (pred == 1),
            "true_negative": (y == 0) & (pred == 0),
            "all_positive": y == 1,
            "all_negative": y == 0,
        }
        for gname, g in groups.items():
            if g.sum() == 0:
                continue
            for t in range(C.WINDOW_LEN):
                m = g & obs[:, t]
                rows.append({
                    "architecture": arch, "horizon": horizon, "group": gname,
                    "time_step": f"t-{C.WINDOW_LEN - 1 - t}", "step_index": t,
                    "mean_attention": float(W[m, t].mean()) if m.sum() else np.nan,
                    "n_windows": int(m.sum()),
                    "frac_step_observed": float(obs[g, t].mean()),
                    "mean_cell_observed": float(d.mask[g, t, :].mean()),
                })
    return pd.DataFrame(rows), raw


def attention_figure(tab: pd.DataFrame, out_stem) -> None:
    import matplotlib.pyplot as plt

    archs = sorted(tab["architecture"].unique())
    horizons = sorted(tab["horizon"].unique())
    fig, axes = plt.subplots(len(horizons), len(archs),
                             figsize=(5.2 * len(archs), 3.4 * len(horizons)),
                             squeeze=False, sharex=True)
    labels = [f"t-{C.WINDOW_LEN - 1 - t}" for t in range(C.WINDOW_LEN)]
    for i, h in enumerate(horizons):
        for jx, arch in enumerate(archs):
            ax = axes[i][jx]
            for g, style in (("true_positive", "-o"), ("true_negative", "--s")):
                sub = tab.query("architecture == @arch and horizon == @h and group == @g") \
                         .sort_values("step_index")
                if len(sub):
                    ax.plot(sub["step_index"], sub["mean_attention"], style,
                            label=g.replace("_", " "), ms=4, lw=1.5)
            ax.axhline(1 / C.WINDOW_LEN, ls=":", c="grey", lw=1,
                       label="uniform (1/8)" if (i == 0 and jx == 0) else None)
            ax.set_title(f"{arch}, h={h}", fontsize=10)
            ax.set_xticks(range(C.WINDOW_LEN), labels)
            ax.grid(alpha=0.3)
            if jx == 0:
                ax.set_ylabel("mean attention (masked)")
            if i == 0 and jx == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Attention over quarters, test set, imputed quarters excluded")
    fig.tight_layout()
    U.savefig(fig, out_stem)


# --------------------------------------------------------------------------
# 2. SHAP
# --------------------------------------------------------------------------
def shap_attributions(splits, arch: str, horizon: int, seed: int = 0,
                      n_background: int = N_BACKGROUND,
                      n_explain_neg: int = N_EXPLAIN_NEG,
                      seed_rng: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """|SHAP| per (window, time step, feature) for the chosen model.

    Expected gradients (`shap.GradientExplainer`) is the SHAP estimator used:
    it is the gradient-based member of the DeepSHAP family and, unlike
    KernelSHAP, is tractable over 232 inputs for thousands of windows.  A
    KernelSHAP cross-check on a small subsample is run separately.
    """
    import shap
    import torch

    key = E.RunSpec(arch=arch, horizon=horizon, treatment="class_weight", seed=seed).key
    model = E.load_model(key)

    rng = np.random.default_rng(seed_rng)
    tr, te = splits["train"], splits["test"]
    bg_idx = rng.choice(tr.n, size=min(n_background, tr.n), replace=False)

    y = te.target(horizon).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = rng.choice(np.flatnonzero(y == 0), size=min(n_explain_neg, int((y == 0).sum())),
                     replace=False)
    ex_idx = np.concatenate([pos, neg])

    bg = torch.as_tensor(tr.X[bg_idx], dtype=torch.float32)
    ex = torch.as_tensor(te.X[ex_idx], dtype=torch.float32)

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    expl = shap.GradientExplainer(Wrapped(model), bg)
    sv = expl.shap_values(ex, nsamples=100)
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv[..., 0]
    meta = {"explainer": "shap.GradientExplainer (expected gradients)",
            "n_background": int(len(bg_idx)), "n_explained": int(len(ex_idx)),
            "n_positive_explained": int(len(pos)), "arch": arch, "horizon": horizon,
            "seed": seed, "shap_shape": list(sv.shape)}
    return sv, ex_idx, meta


def aggregate_shap(sv: np.ndarray, mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """|SHAP| aggregated by family x time step, by family, and per feature.

    Masked cells are excluded from every mean: their input was set to the train
    mean, so an attribution there measures the imputation, not the firm.
    """
    a = np.abs(sv)
    m = mask.astype(bool)
    a = np.where(m, a, np.nan)

    per_cell = np.nanmean(a, axis=0)                       # (8, 29)
    counts = m.sum(axis=0)

    fam_rows, cell_rows = [], []
    for fam, cols in C.FAMILIES.items():
        for t in range(C.WINDOW_LEN):
            v = np.nanmean(per_cell[t, cols])
            cell_rows.append({"family": fam, "time_step": f"t-{C.WINDOW_LEN - 1 - t}",
                              "step_index": t, "mean_abs_shap": float(v),
                              "n_observed_cells": int(counts[t, cols].sum())})
        fam_rows.append({"family": fam, "n_features": len(cols),
                         "mean_abs_shap": float(np.nanmean(per_cell[:, cols])),
                         "total_abs_shap": float(np.nansum(per_cell[:, cols]))})

    feat_rows = [{"feature": C.FEATURE_NAMES[f], "index": f,
                  "family": next(k for k, v in C.FAMILIES.items() if f in v),
                  "mean_abs_shap": float(np.nanmean(per_cell[:, f])),
                  "is_altman": f in C.ALTMAN_IDX,
                  "observed_rate": float(m[:, :, f].mean())}
                 for f in range(C.N_FEATURES)]

    fam = pd.DataFrame(fam_rows).sort_values("mean_abs_shap", ascending=False)
    grid = pd.DataFrame(cell_rows)
    feat = pd.DataFrame(feat_rows).sort_values("mean_abs_shap", ascending=False)
    fam["share"] = fam["total_abs_shap"] / fam["total_abs_shap"].sum()
    feat["rank"] = range(1, len(feat) + 1)
    return fam, grid, feat


def shap_by_quarter(grid: pd.DataFrame) -> pd.Series:
    return grid.groupby("step_index")["mean_abs_shap"].mean()


def shap_heatmap(grid: pd.DataFrame, horizon: int, out_stem) -> None:
    import matplotlib.pyplot as plt

    fams = list(C.FAMILIES)
    Z = np.array([[float(grid.query("family == @f and step_index == @t")["mean_abs_shap"].iloc[0])
                   for t in range(C.WINDOW_LEN)] for f in fams])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    im = ax.imshow(Z, cmap="magma", aspect="auto")
    ax.set_xticks(range(C.WINDOW_LEN), [f"t-{C.WINDOW_LEN - 1 - t}" for t in range(C.WINDOW_LEN)])
    ax.set_yticks(range(len(fams)), fams)
    for i in range(len(fams)):
        for jx in range(C.WINDOW_LEN):
            ax.text(jx, i, f"{Z[i, jx]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if Z[i, jx] < Z.max() * 0.6 else "black")
    ax.set_title(f"Mean |SHAP| by ratio family and quarter (h={horizon}, masked cells excluded)")
    fig.colorbar(im, ax=ax, label="mean |SHAP|")
    U.savefig(fig, out_stem)


# --------------------------------------------------------------------------
# 3. Cross-check
# --------------------------------------------------------------------------
def cross_check(attn_tab: pd.DataFrame, grid: pd.DataFrame, horizon: int,
                arch: str) -> list[dict]:
    sh = shap_by_quarter(grid)
    rows = []
    for a in attn_tab["architecture"].unique():
        for group in ("true_positive", "all_positive"):
            sub = attn_tab.query("architecture == @a and horizon == @horizon and group == @group") \
                          .sort_values("step_index")
            if len(sub) != C.WINDOW_LEN:
                continue
            at = sub["mean_attention"].to_numpy()
            sv = sh.reindex(range(C.WINDOW_LEN)).to_numpy()
            ok = np.isfinite(at) & np.isfinite(sv)
            if ok.sum() < 3:
                continue
            rho, prho = stats.spearmanr(at[ok], sv[ok])
            tau, ptau = stats.kendalltau(at[ok], sv[ok])
            rows.append({
                "horizon": horizon, "attention_model": a, "shap_model": arch,
                "group": group, "n_steps": int(ok.sum()),
                "spearman_rho": float(rho), "spearman_p": float(prho),
                "kendall_tau": float(tau), "kendall_p": float(ptau),
                "attention_argmax": f"t-{C.WINDOW_LEN - 1 - int(np.nanargmax(at))}",
                "shap_argmax": f"t-{C.WINDOW_LEN - 1 - int(np.nanargmax(sv))}",
                "agree_on_top_quarter": int(np.nanargmax(at)) == int(np.nanargmax(sv)),
            })
    return rows


# --------------------------------------------------------------------------
def main(full: bool = True, horizons=(1, 4), seeds=C.SEEDS,
         best_arch: str | None = None) -> dict:
    print("=" * 70)
    print("PHASE F — INTERPRETABILITY")
    print("=" * 70)
    splits = D.load_all(full=full)

    if best_arch is None:
        deep = pd.read_csv(C.RESULTS / "deep_all.csv")
        best_arch = (deep[deep["horizon"] == 4]
                     .sort_values("pr_auc_mean", ascending=False)["arch"].iloc[0])
    print(f"[phase F] best architecture at h=4 by mean test PR-AUC: {best_arch}")

    attn_tabs, xchecks, fam_tabs, feat_tabs, metas = [], [], [], [], []
    for h in horizons:
        tab, _ = attention_table(splits, h, seeds=seeds)
        attn_tabs.append(tab)

        sv, ex_idx, meta = shap_attributions(splits, best_arch, h, seed=seeds[0])
        fam, grid, feat = aggregate_shap(sv, splits["test"].mask[ex_idx])
        fam["horizon"] = h; grid["horizon"] = h; feat["horizon"] = h
        meta["horizon"] = h
        fam_tabs.append(fam); feat_tabs.append(feat); metas.append(meta)
        shap_heatmap(grid, h, C.FIGURES / f"shap_family_by_quarter_h{h}")
        if h == horizons[-1]:
            # The canonical filename the phase gate names.
            shap_heatmap(grid, h, C.FIGURES / "shap_family_by_quarter")
        grid.to_csv(C.RESULTS / f"shap_family_by_quarter_h{h}.csv", index=False)
        xchecks += cross_check(tab, grid, h, best_arch)

    attn = pd.concat(attn_tabs, ignore_index=True)
    U.write_table(attn, C.RESULTS / "attention_by_quarter", floatfmt="%.5f")
    attention_figure(attn, C.FIGURES / "attention_by_quarter")

    fams = pd.concat(fam_tabs, ignore_index=True)
    feats = pd.concat(feat_tabs, ignore_index=True)
    U.write_table(fams, C.RESULTS / "shap_family_importance", floatfmt="%.5f")
    U.write_table(feats, C.RESULTS / "shap_feature_importance", floatfmt="%.5f")
    xc = pd.DataFrame(xchecks)
    U.write_table(xc, C.RESULTS / "interpretability_crosscheck", floatfmt="%.4f")

    write_summary(attn, fams, feats, xc, metas, best_arch, horizons)
    return {"best_arch": best_arch, "n_crosschecks": len(xc)}


def write_summary(attn, fams, feats, xc, metas, best_arch, horizons) -> None:
    L = ["# Interpretability summary", "",
         f"SHAP model: **{best_arch}** (best mean test PR-AUC at h=4). "
         f"Estimator: {metas[0]['explainer']}, "
         f"{metas[0]['n_background']} background windows, "
         f"{metas[0]['n_explained']} explained windows per horizon.", "",
         "Masked cells — imputed after forward fill, or structurally undefined — are",
         "excluded from every average in this file.", ""]

    for h in horizons:
        L += [f"## Horizon h = {h}", "", "### Which quarters attention points at", ""]
        sub = attn.query("horizon == @h and group == 'true_positive'")
        for a in sorted(sub["architecture"].unique()):
            s = sub.query("architecture == @a").sort_values("mean_attention", ascending=False)
            top = ", ".join(f"{r.time_step} ({r.mean_attention:.4f})" for r in s.head(3).itertuples())
            L.append(f"- **{a}** on true positives: heaviest on {top} "
                     f"(uniform would be {1 / C.WINDOW_LEN:.4f}).")
        L += ["", "### Which quarters SHAP points at", ""]
        g = pd.read_csv(C.RESULTS / f"shap_family_by_quarter_h{h}.csv")
        q = g.groupby(["step_index"])["mean_abs_shap"].mean().sort_values(ascending=False)
        L.append("- " + ", ".join(f"t-{C.WINDOW_LEN - 1 - int(i)} ({v:.4f})"
                                  for i, v in q.head(3).items()) + " carry the most |SHAP|.")
        L += ["", "### Agreement between the two methods", "",
              U.to_markdown(xc[xc["horizon"] == h][
                  ["attention_model", "group", "spearman_rho", "spearman_p",
                   "kendall_tau", "attention_argmax", "shap_argmax",
                   "agree_on_top_quarter"]]), ""]
        f = feats[feats["horizon"] == h].head(10)
        L += ["### Top-10 features by mean |SHAP|", "",
              U.to_markdown(f[["rank", "feature", "family", "mean_abs_shap",
                               "is_altman", "observed_rate"]]), ""]
        alt = feats[(feats["horizon"] == h) & feats["is_altman"]]
        r29 = feats[(feats["horizon"] == h) & (feats["feature"] == "r29_negative_equity_flag")]
        L += [f"- Altman's five rank {sorted(alt['rank'].tolist())} of 29.",
              f"- `r29_negative_equity_flag` ranks {int(r29['rank'].iloc[0])} of 29.", ""]

    L += ["## Family importance", "",
          U.to_markdown(fams[["horizon", "family", "n_features", "mean_abs_shap", "share"]]), ""]
    (C.RESULTS / "interpretability_summary.md").write_text("\n".join(L), encoding="utf-8")
