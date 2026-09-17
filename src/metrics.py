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
