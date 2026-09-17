"""The two leakage rules that live in code rather than in the dataset:
SMOTE is applied to train only, and thresholds are chosen on val only.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src import config as C
from src import experiment as E
from src import imbalance as IMB
from src import metrics as M
from src import models as Models
from src import train as T


def _toy(n=400, seed=0, pos_rate=0.05):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(np.float32)
    X = rng.normal(size=(n, C.WINDOW_LEN, 6)).astype(np.float32)
    X += y[:, None, None] * 1.5
    return X, y


# --------------------------------------------------------------------------
# SMOTE
# --------------------------------------------------------------------------
def test_smote_balances_and_reshapes_back():
    X, y = _toy(600, seed=1)
    Xr, yr, st = IMB.smote_sequences(X, y, seed=0)
    assert Xr.shape[1:] == X.shape[1:], "the (8, 29) window shape must survive the round trip"
    assert int(yr.sum()) == int((yr == 0).sum()), "default strategy balances the classes"
    assert st["applied"] and st["n_synthetic"] == len(yr) - len(y)
    # The real rows come back first and unchanged.
    assert np.allclose(Xr[: len(y)], X)
    assert np.array_equal(yr[: len(y)], y.astype(int))


def test_smote_synthetic_rows_lie_between_two_real_positives():
    X, y = _toy(300, seed=2, pos_rate=0.1)
    Xr, yr, parent, st = IMB.smote_with_parents(X, y, seed=0)
    pos = X[y == 1].reshape((y == 1).sum(), -1)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    synth = Xr[len(y):].reshape(-1, X.shape[1] * X.shape[2])
    assert (synth >= lo - 1e-5).all() and (synth <= hi + 1e-5).all()
    assert (yr[len(y):] == 1).all()
    assert (y[parent[len(y):]] == 1).all(), "every synthetic row descends from a positive"


def test_smote_touches_only_the_training_arrays():
    """The trained model must see resampled train rows and untouched val/test."""
    Xtr, ytr = _toy(500, seed=3)
    Xva, yva = _toy(200, seed=4)
    Xte, yte = _toy(200, seed=5)
    va_before, te_before = Xva.copy(), Xte.copy()

    res = T.train_one("lstm", Xtr, ytr, Xva, yva, Xte, yte, treatment="smote",
                      seed=0, hp=T.HParams(max_epochs=1, batch_size=128), horizon=4)

    assert np.array_equal(Xva, va_before), "val tensor was modified"
    assert np.array_equal(Xte, te_before), "test tensor was modified"
    assert len(res.val_pred) == len(yva), "val was resampled"
    assert len(res.test_pred) == len(yte), "test was resampled"
    st = res.imbalance_stats
    assert st["applied"] and st["n_in"] == len(ytr), \
        "SMOTE must be fitted on the training rows, and only those"
    assert st["n_pos_in"] == int(ytr.sum())


def test_class_weights_come_from_train_labels_only():
    ytr = np.array([0] * 90 + [1] * 10)
    cw = IMB.class_weights(ytr)
    assert cw["n"] == 100 and cw["n_pos"] == 10 and cw["n_neg"] == 90
    assert cw["pos_weight"] == pytest.approx(9.0)
    assert cw["w_pos"] == pytest.approx(100 / 20) and cw["w_neg"] == pytest.approx(100 / 180)


def test_focal_loss_down_weights_easy_examples():
    logits = torch.tensor([[4.0], [0.0]])          # confident-correct, then uncertain
    target = torch.tensor([[1.0], [1.0]])
    per = IMB.focal_loss_with_logits(logits, target, reduction="none")
    assert per[0] < per[1], "an easy positive must contribute less than a hard one"
    plain = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    assert (per <= plain).all(), "focal loss never exceeds plain BCE at alpha <= 1"


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
def test_threshold_comes_from_val_and_ignores_test(tmp_path, monkeypatch):
    import pandas as pd

    key = "unit_threshold_check"
    rng = np.random.default_rng(0)
    n = 2000
    yv = (rng.random(n) < 0.05).astype(int)
    pv = rng.random(n) * 0.5 + yv * 0.4
    yt = (rng.random(n) < 0.05).astype(int)
    pt = rng.random(n) * 0.5 + yt * 0.4

    for path, y, p in ((C.PREDS, yt, pt), (E.VAL_PREDS, yv, pv)):
        pd.DataFrame({"cik": np.arange(n).astype(str),
                      "end_quarter": ["2022Q1"] * n,
                      "y_true": y.astype(np.int8),
                      "p_hat": p.astype(np.float32)}).to_parquet(path / f"{key}.parquet")
    try:
        s = E.score(key)
        expected, _ = M.best_f1_threshold(yv, pv)
        assert s["threshold"] == pytest.approx(expected), \
            "the applied threshold must be the one maximising F1 on val"

        # Scrambling test labels must not move the threshold.
        pd.DataFrame({"cik": np.arange(n).astype(str),
                      "end_quarter": ["2022Q1"] * n,
                      "y_true": rng.permutation(yt).astype(np.int8),
                      "p_hat": pt.astype(np.float32)}).to_parquet(C.PREDS / f"{key}.parquet")
        assert E.score(key)["threshold"] == pytest.approx(expected)
    finally:
        (C.PREDS / f"{key}.parquet").unlink(missing_ok=True)
        (E.VAL_PREDS / f"{key}.parquet").unlink(missing_ok=True)


def test_cost_threshold_minimises_expected_cost():
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.02).astype(int)
    p = rng.random(n) * 0.6 + y * 0.35
    for ratio in C.COST_RATIOS:
        thr, cost = M.best_cost_threshold(y, p, fn_cost=float(ratio))
        brute = min(M.expected_cost(y, p, t, fn_cost=float(ratio))
                    for t in np.unique(np.concatenate([p, [p.max() + 1e-9]])))
        assert cost == pytest.approx(brute, abs=1e-12)
        assert M.expected_cost(y, p, thr, fn_cost=float(ratio)) == pytest.approx(cost, abs=1e-12)


def test_higher_miss_cost_lowers_the_threshold():
    rng = np.random.default_rng(2)
    y = (rng.random(4000) < 0.03).astype(int)
    p = rng.random(4000) * 0.6 + y * 0.3
    thrs = [M.best_cost_threshold(y, p, fn_cost=float(r))[0] for r in (1, 10, 20, 50)]
    assert thrs == sorted(thrs, reverse=True), \
        "as a missed bankruptcy gets more expensive the model should alarm sooner"
