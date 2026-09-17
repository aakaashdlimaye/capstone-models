"""Imbalance treatments.  Every one of them is fitted on and applied to the
train split only — val and test are never resampled or reweighted (leakage
rule 2).  Each function returns the statistics it used so the leakage audit can
prove where they came from.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

TREATMENTS = ("none", "class_weight", "smote", "focal")


# --------------------------------------------------------------------------
# Class weights
# --------------------------------------------------------------------------
def class_weights(y_train: np.ndarray) -> dict:
    """Inverse-frequency weights computed from train labels alone."""
    y = np.asarray(y_train).astype(int).ravel()
    n, n_pos = len(y), int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0:
        return {"pos_weight": 1.0, "n": n, "n_pos": 0, "n_neg": n_neg,
                "w_pos": 1.0, "w_neg": 1.0}
    return {
        "pos_weight": float(n_neg / n_pos),       # what BCEWithLogitsLoss takes
        "w_pos": float(n / (2 * n_pos)),          # sklearn 'balanced' form
        "w_neg": float(n / (2 * n_neg)),
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
    }


# --------------------------------------------------------------------------
# SMOTE on the flattened window
# --------------------------------------------------------------------------
def smote_sequences(X: np.ndarray, y: np.ndarray, seed: int = 0,
                    sampling_strategy: float | str = 1.0,
                    k_neighbors: int = 5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Flatten (n, T, F) -> (n, T*F), oversample, reshape back.

    This is the standard practice the literature uses and it carries a known
    artefact: an interpolated sequence is a weighted blend of two real firms'
    eight-quarter histories, so it is not a firm that ever existed and its
    quarter-to-quarter dynamics are a linear mix rather than an observed path.
    Reported as such rather than presented as data augmentation.
    """
    from imblearn.over_sampling import SMOTE

    n, T, Fd = X.shape
    y = np.asarray(y).astype(int).ravel()
    n_pos = int(y.sum())
    stats = {"n_in": n, "n_pos_in": n_pos, "k_neighbors": k_neighbors,
             "sampling_strategy": sampling_strategy, "seed": seed}
    if n_pos < 2:
        stats["applied"] = False
        stats["reason"] = "fewer than 2 positives"
        return X, y, stats

    k = min(k_neighbors, n_pos - 1)
    sm = SMOTE(random_state=seed, k_neighbors=k, sampling_strategy=sampling_strategy)
    Xr, yr = sm.fit_resample(X.reshape(n, T * Fd).astype(np.float64), y)
    stats.update({"applied": True, "k_used": k, "n_out": int(len(yr)),
                  "n_pos_out": int(yr.sum()),
                  "n_synthetic": int(len(yr) - n)})
    return Xr.reshape(-1, T, Fd).astype(np.float32), yr.astype(np.int64), stats


# --------------------------------------------------------------------------
# Focal loss — Chang, Liu & Deng (2025) use gamma = 2, alpha = 0.25
# --------------------------------------------------------------------------
def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor,
                           gamma: float = 2.0, alpha: float = 0.25,
                           reduction: str = "mean") -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    loss = a_t * (1 - p_t).pow(gamma) * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def make_loss(treatment: str, y_train: np.ndarray, gamma: float = 2.0,
              alpha: float = 0.25, device=None):
    """The loss function for a treatment, plus the statistics it was built from."""
    stats = {"treatment": treatment}
    if treatment == "class_weight":
        cw = class_weights(y_train)
        stats.update(cw)
        pw = torch.tensor(cw["pos_weight"], dtype=torch.float32, device=device)

        def loss_fn(logits, target):
            return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    elif treatment == "focal":
        stats.update({"gamma": gamma, "alpha": alpha})

        def loss_fn(logits, target):
            return focal_loss_with_logits(logits, target, gamma=gamma, alpha=alpha)
    else:  # 'none' and 'smote' both train under plain BCE
        def loss_fn(logits, target):
            return F.binary_cross_entropy_with_logits(logits, target)
    return loss_fn, stats


# --------------------------------------------------------------------------
# SMOTE with parent provenance — only the protocol audit needs this
# --------------------------------------------------------------------------
def smote_with_parents(X: np.ndarray, y: np.ndarray, seed: int = 0,
                       k_neighbors: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Textbook SMOTE, returning each row's originating index.

    imbalanced-learn does not expose which real sample seeded each synthetic
    one, and the protocol audit needs it: to reproduce the "SMOTE before the
    split" mistake faithfully, a synthetic row has to inherit the split and end
    quarter of the real firm it was interpolated from.  The algorithm is the
    standard one (Chawla et al. 2002) and is unit-tested against imbalanced-learn
    for the same seed-free properties.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n, T, Fd = X.shape
    y = np.asarray(y).astype(int).ravel()
    flat = X.reshape(n, T * Fd)
    pos_idx = np.flatnonzero(y == 1)
    n_pos, n_neg = len(pos_idx), int((y == 0).sum())
    n_new = n_neg - n_pos
    parents = np.arange(n)
    if n_pos < 2 or n_new <= 0:
        return X, y, parents, {"applied": False, "n_pos": n_pos, "n_neg": n_neg}

    k = min(k_neighbors, n_pos - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(flat[pos_idx])
    _, nbr = nn.kneighbors(flat[pos_idx])
    nbr = nbr[:, 1:]                                   # drop self

    base = rng.integers(0, n_pos, size=n_new)
    pick = nbr[base, rng.integers(0, k, size=n_new)]
    lam = rng.random((n_new, 1))
    synth = flat[pos_idx[base]] + lam * (flat[pos_idx[pick]] - flat[pos_idx[base]])

    Xo = np.concatenate([flat, synth], axis=0).reshape(-1, T, Fd).astype(np.float32)
    yo = np.concatenate([y, np.ones(n_new, dtype=int)])
    po = np.concatenate([parents, pos_idx[base]])
    stats = {"applied": True, "n_pos": n_pos, "n_neg": n_neg, "n_synthetic": int(n_new),
             "k_used": k, "seed": seed}
    return Xo, yo, po, stats
