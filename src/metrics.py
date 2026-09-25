"""Evaluation metrics, threshold rules and the two significance tests.

Everything here consumes `(y_true, p_hat)` arrays, so it is shared by the
classical baselines and the deep models without either knowing about the other.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from . import config as C

EPS = 1e-12


# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------
def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision — the honest headline at a ~1% base rate."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def confusion_at(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    yhat = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y).astype(int), yhat, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    tn, fp, fn, tp = confusion_at(y, p, thr)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = tn + fp + fn + tp
    return {
        "threshold": float(thr),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "accuracy": float((tp + tn) / n) if n else float("nan"),
        "alarm_rate": float((tp + fp) / n) if n else float("nan"),
    }


def evaluate(y: np.ndarray, p: np.ndarray, thr: float | None = None,
             cost_ratios: tuple[int, ...] = C.COST_RATIOS) -> dict:
    """Threshold-free metrics, plus thresholded ones when `thr` is given."""
    out = {
        "n": int(len(y)),
        "n_pos": int(np.asarray(y).sum()),
        "base_rate": float(np.asarray(y).mean()),
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
        "brier": float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)),
    }
    if thr is not None:
        out.update(threshold_metrics(y, p, thr))
        for r in cost_ratios:
            out[f"cost_1to{r}"] = expected_cost(y, p, thr, fn_cost=float(r), fp_cost=1.0)
    return out


# --------------------------------------------------------------------------
# Cost — Radovanovic & Haas (2023), expressed per window
# --------------------------------------------------------------------------
def expected_cost(y: np.ndarray, p: np.ndarray, thr: float,
                  fn_cost: float, fp_cost: float = 1.0) -> float:
    """Expected cost per window.

    A Type I error is a false alarm on a healthy firm (cost `fp_cost`); a
    Type II error is a missed bankruptcy (cost `fn_cost`).  The ratios in the
    spec (1:1, 10:1, 20:1, 50:1) are Type II : Type I.
    """
    tn, fp, fn, tp = confusion_at(y, p, thr)
    n = tn + fp + fn + tp
    return float((fp * fp_cost + fn * fn_cost) / n) if n else float("nan")


def best_cost_threshold(y: np.ndarray, p: np.ndarray, fn_cost: float,
                        fp_cost: float = 1.0) -> tuple[float, float]:
    """Threshold minimising expected cost, searched over the observed scores.

    Always called on validation; never on test (leakage rule 3).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    n, P = len(ys), int(ys.sum())
    # Predicting positive for the top k scores.
    tp = np.concatenate([[0], np.cumsum(ys)])
    fp = np.concatenate([[0], np.cumsum(1 - ys)])
    fn = P - tp
    cost = (fp * fp_cost + fn * fn_cost) / n
    k = int(np.argmin(cost))
    # Threshold sits just above the k-th score (k = 0 means "never alarm").
    thr = float(ps[k - 1]) if k > 0 else float(np.nextafter(ps[0], np.inf))
    return thr, float(cost[k])


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the given (validation) split."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return 0.5, float("nan")
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, EPS)
    # precision_recall_curve returns len(thr) = len(prec) - 1
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5, float(f1[i])


# --------------------------------------------------------------------------
# DeLong — correlated ROC-AUC comparison (Sun & Xu 2014 fast algorithm)
# --------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _structural_components(preds: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V10 / V01 components and AUCs for k predictors, positives first."""
    k, total = preds.shape
    n = total - m
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, total))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r, :])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v01, v10


def delong_var(y: np.ndarray, preds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their covariance matrix under DeLong's estimator."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")       # positives first, stable
    ys = y[order]
    m = int(ys.sum())
    n = len(ys) - m
    if m == 0 or n == 0:
        k = len(preds)
        return np.full(k, np.nan), np.full((k, k), np.nan)
    P = np.vstack([np.asarray(p, dtype=float)[order] for p in preds])
    aucs, v01, v10 = _structural_components(P, m)
    s01 = np.cov(v01) if len(preds) > 1 else np.atleast_2d(np.var(v01, ddof=1))
    s10 = np.cov(v10) if len(preds) > 1 else np.atleast_2d(np.var(v10, ddof=1))
    cov = np.atleast_2d(s01) / m + np.atleast_2d(s10) / n
    return aucs, cov


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-sided DeLong test on AUC(p2) - AUC(p1), with a CI on the difference."""
    aucs, cov = delong_var(y, [p1, p2])
    if np.isnan(aucs).any():
        return {"auc1": float("nan"), "auc2": float("nan"), "diff": float("nan"),
                "se": float("nan"), "z": float("nan"), "p_value": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    diff = float(aucs[1] - aucs[0])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z))) if se > 0 else 1.0
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {
        "auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff, "se": se,
        "z": float(z), "p_value": pval,
        "ci_low": diff - half, "ci_high": diff + half,
    }


# --------------------------------------------------------------------------
# McNemar — paired comparison of thresholded decisions
# --------------------------------------------------------------------------
def mcnemar_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                 thr1: float, thr2: float, exact_below: int = 25) -> dict:
    """McNemar on which model is *correct*, not on which fires.

    Uses the exact binomial when the discordant count is small, which it will
    be at this base rate; otherwise the continuity-corrected chi-square.
    """
    y = np.asarray(y).astype(int)
    c1 = ((np.asarray(p1) >= thr1).astype(int) == y)
    c2 = ((np.asarray(p2) >= thr2).astype(int) == y)
    b = int(np.sum(c1 & ~c2))     # model 1 right, model 2 wrong
    c = int(np.sum(~c1 & c2))     # model 2 right, model 1 wrong
    nd = b + c
    if nd == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": 0.0,
                "p_value": 1.0, "test": "none (no discordant pairs)"}
    if nd < exact_below:
        pval = float(stats.binomtest(min(b, c), nd, 0.5).pvalue)
        stat, name = float(min(b, c)), "exact binomial"
    else:
        stat = (abs(b - c) - 1) ** 2 / nd
        pval = float(stats.chi2.sf(stat, 1))
        name = "chi2 (continuity corrected)"
    return {"b": b, "c": c, "n_discordant": nd, "statistic": float(stat),
            "p_value": pval, "test": name}


# --------------------------------------------------------------------------
# Bootstrap — DeLong covers ROC-AUC only, so PR-AUC differences use this
# --------------------------------------------------------------------------
def bootstrap_diff(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                   stat=pr_auc, n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """Stratified bootstrap CI on stat(p2) - stat(p1)."""
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    if len(pos) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    obs = stat(y, p2) - stat(y, p1)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        yb = y[idx]
        diffs[b] = stat(yb, p2[idx]) - stat(yb, p1[idx])
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value: how often the sign flips relative to the point estimate.
    frac = float(np.mean(diffs <= 0)) if obs > 0 else float(np.mean(diffs >= 0))
    pval = float(min(1.0, 2 * frac))
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": pval, "n_boot": n_boot}


# --------------------------------------------------------------------------
# Cluster bootstrap — the honest interval when rows are not independent
# --------------------------------------------------------------------------
# Stride-1 windowing gives one firm up to ~50 overlapping windows that share
# seven of their eight input quarters and one label-generating event.  A
# bootstrap that resamples windows treats those as independent evidence and
# returns an interval that is too narrow.  Resampling *firms*, and taking all of
# a drawn firm's windows, keeps the within-firm correlation intact.  These are
# the intervals the paper reports; the window-level ones are kept beside them.


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unique group labels, and for each the row positions belonging to it."""
    g = np.asarray(groups)
    order = np.argsort(g, kind="mergesort")
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    bounds = np.r_[starts, len(gs)]
    uniq = gs[starts]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    return uniq, members


def _cluster_resample(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[i] for i in pick])


def cluster_bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                         stat=pr_auc, n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> dict:
    """Firm-clustered bootstrap CI on a single model's statistic."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    obs = stat(y, p)
    _, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = _cluster_resample(members, rng)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals[b] = stat(yb, p[idx])
    ok = np.isfinite(vals)
    if ok.sum() < 50:
        return {"value": float(obs), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": int(ok.sum()), "n_clusters": len(members), "level": "cluster"}
    lo, hi = np.percentile(vals[ok], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": int(ok.sum()), "n_clusters": len(members), "level": "cluster"}


def cluster_bootstrap_diff(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                           groups: np.ndarray, stat=pr_auc, n_boot: int = 2000,
                           seed: int = 0, alpha: float = 0.05) -> dict:
    """Firm-clustered bootstrap CI and two-sided p on stat(p2) - stat(p1).

    Both models are scored on the same resampled rows each iteration, so the
    interval is on the paired difference and preserves both the correlation
    between the two models' errors and the correlation within a firm.
    """
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
    obs = stat(y, p2) - stat(y, p1)
    _, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    diffs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = _cluster_resample(members, rng)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        diffs[b] = stat(yb, p2[idx]) - stat(yb, p1[idx])
    ok = np.isfinite(diffs)
    if ok.sum() < 50:
        return {"diff": float(obs), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "n_boot": int(ok.sum()),
                "n_clusters": len(members), "level": "cluster"}
    d = diffs[ok]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    frac = float(np.mean(d <= 0)) if obs > 0 else float(np.mean(d >= 0))
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, 2 * frac)), "n_boot": int(ok.sum()),
            "n_clusters": len(members), "level": "cluster"}


def compare(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, groups: np.ndarray,
            thr1: float | None = None, thr2: float | None = None,
            n_boot: int = 2000, seed: int = 0) -> dict:
    """The full comparison the paper reports for any two models on one row set.

    Cluster-bootstrap intervals are primary.  DeLong and McNemar assume
    independent rows, so they are carried beside them under `window_level_`
    names rather than dropped: the difference between the two is itself
    informative about how much the independence assumption buys.
    """
    out = {}
    for name, stat in (("pr_auc", pr_auc), ("roc_auc", roc_auc)):
        cb = cluster_bootstrap_diff(y, p1, p2, groups, stat=stat, n_boot=n_boot, seed=seed)
        out[f"{name}_diff"] = cb["diff"]
        out[f"{name}_cluster_ci_low"] = cb["ci_low"]
        out[f"{name}_cluster_ci_high"] = cb["ci_high"]
        out[f"{name}_cluster_p"] = cb["p_value"]
        wb = bootstrap_diff(y, p1, p2, stat=stat, n_boot=n_boot, seed=seed)
        out[f"{name}_window_level_ci_low"] = wb["ci_low"]
        out[f"{name}_window_level_ci_high"] = wb["ci_high"]
    out["n_clusters"] = int(len(np.unique(groups)))
    dl = delong_test(y, p1, p2)
    out.update({"delong_window_level_diff": dl["diff"],
                "delong_window_level_ci_low": dl["ci_low"],
                "delong_window_level_ci_high": dl["ci_high"],
                "delong_window_level_p": dl["p_value"]})
    if thr1 is not None and thr2 is not None:
        mc = mcnemar_test(y, p1, p2, thr1, thr2)
        out.update({"mcnemar_window_level_b": mc["b"], "mcnemar_window_level_c": mc["c"],
                    "mcnemar_window_level_p": mc["p_value"]})
    return out


# --------------------------------------------------------------------------
# Practitioner metrics (Phase H)
# --------------------------------------------------------------------------
def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    """Precision, recall and lift among the k highest-scored rows."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n, total_pos = len(y), int(y.sum())
    k = int(min(max(k, 0), n))
    if k == 0 or total_pos == 0:
        return {"k": k, "tp": 0, "precision": float("nan"), "recall": 0.0,
                "lift": float("nan")}
    top = np.argpartition(-p, k - 1)[:k]
    tp = int(y[top].sum())
    base = total_pos / n
    return {"k": k, "tp": tp, "precision": tp / k, "recall": tp / total_pos,
            "lift": (tp / k) / base if base else float("nan")}


def firm_level_scores(y: np.ndarray, p: np.ndarray, groups: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Collapse windows to firms: each firm's highest score, and whether it ever fails.

    A credit officer decides about a firm, not about a window, so this is the
    view that matches the decision actually being made.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    _, members = _group_index(groups)
    fy = np.array([int(y[m].max()) for m in members])
    fp = np.array([float(p[m].max()) for m in members])
    return fy, fp


def capture_curve(y: np.ndarray, p: np.ndarray, n_points: int = 100):
    """Cumulative share of positives captured as the alarm budget grows."""
    import pandas as pd

    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(p, dtype=float))
    ys = y[order]
    total = max(int(ys.sum()), 1)
    cum = np.cumsum(ys) / total
    frac = np.arange(1, len(ys) + 1) / len(ys)
    take = np.unique(np.linspace(0, len(ys) - 1, n_points).astype(int))
    return pd.DataFrame({"frac_flagged": frac[take], "frac_captured": cum[take]})


def logistic_scale(p: np.ndarray) -> np.ndarray:
    """Squash an unbounded score into (0, 1) so it can be asked about calibration.

    Altman, Ohlson and Zmijewski emit scores, not probabilities; a reliability
    curve on a raw Z is meaningless.  Standardising and passing through a
    logistic is monotone, so it leaves every ranking metric untouched and only
    puts the score on an axis where "predicted 3% and 3% failed" is a statement.
    Any model scored this way is flagged `is_probability = False`.
    """
    p = np.asarray(p, dtype=float)
    sd = p.std()
    z = (p - p.mean()) / (sd if sd > 0 else 1.0)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10,
                strategy: str = "quantile"):
    """Reliability table plus Brier score and expected calibration error.

    Quantile bins, because at a ~1% base rate uniform bins put almost every row
    into the first bin and the curve says nothing.
    """
    import pandas as pd

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    if len(edges) < 3:
        edges = np.array([p.min(), float(np.median(p)), p.max() + 1e-12])
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows, ece, n = [], 0.0, len(y)
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        conf, obs = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": conf,
                     "observed_rate": obs, "gap": obs - conf})
        ece += (m.sum() / n) * abs(obs - conf)
    return pd.DataFrame(rows), {"brier": float(np.mean((p - y) ** 2)),
                                "ece": float(ece), "n_bins_used": len(rows),
                                "strategy": strategy}
