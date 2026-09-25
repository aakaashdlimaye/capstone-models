"""The four correctness fixes, tested where they can be tested in isolation:
cluster bootstrapping, natural-base-rate protocol scoring, precision@k against
a hand-computed case, and calibration / ECE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import metrics as M


# --------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------
def _duplicated(n_groups=200, k=20, seed=0, sep1=0.3, sep2=0.9):
    """Each group is one row repeated k times: no extra information at all."""
    rng = np.random.default_rng(seed)
    base_y = (rng.random(n_groups) < 0.12).astype(int)
    y = np.repeat(base_y, k)
    groups = np.repeat(np.arange(n_groups), k)
    p1 = np.repeat(rng.random(n_groups) + sep1 * base_y, k)
    p2 = np.repeat(rng.random(n_groups) + sep2 * base_y, k)
    return y, p1, p2, groups


def test_cluster_ci_is_wider_when_rows_are_duplicates():
    """The whole point: duplicated rows must not narrow the interval."""
    y, p1, p2, g = _duplicated()
    cb = M.cluster_bootstrap_diff(y, p1, p2, g, n_boot=400, seed=0)
    wb = M.bootstrap_diff(y, p1, p2, n_boot=400, seed=0)
    cluster_width = cb["ci_high"] - cb["ci_low"]
    window_width = wb["ci_high"] - wb["ci_low"]
    assert cluster_width > window_width, (
        f"cluster {cluster_width:.4f} should exceed window {window_width:.4f}")
    assert cb["level"] == "cluster"
    assert cb["n_clusters"] == len(np.unique(g))


def test_cluster_ci_widens_as_duplication_grows():
    widths = []
    for k in (2, 10, 40):
        y, p1, p2, g = _duplicated(k=k)
        cb = M.cluster_bootstrap_diff(y, p1, p2, g, n_boot=300, seed=1)
        wb = M.bootstrap_diff(y, p1, p2, n_boot=300, seed=1)
        widths.append((cb["ci_high"] - cb["ci_low"]) / (wb["ci_high"] - wb["ci_low"]))
    assert widths[-1] > widths[0], (
        "the cluster/window width ratio should grow with duplication, got "
        f"{[round(w, 2) for w in widths]}")


def test_cluster_ci_brackets_the_point_estimate():
    y, p1, p2, g = _duplicated(seed=3)
    cb = M.cluster_bootstrap_diff(y, p1, p2, g, n_boot=400, seed=0)
    assert cb["ci_low"] <= cb["diff"] <= cb["ci_high"]
    single = M.cluster_bootstrap_ci(y, p2, g, n_boot=400, seed=0)
    assert single["ci_low"] <= single["value"] <= single["ci_high"]


def test_cluster_and_window_agree_when_every_group_is_one_row():
    """With no clustering present the two estimators should broadly coincide."""
    rng = np.random.default_rng(7)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    p1, p2 = rng.random(n) + 0.3 * y, rng.random(n) + 0.8 * y
    g = np.arange(n)
    cb = M.cluster_bootstrap_diff(y, p1, p2, g, n_boot=400, seed=0)
    wb = M.bootstrap_diff(y, p1, p2, n_boot=400, seed=0)
    cw, ww = cb["ci_high"] - cb["ci_low"], wb["ci_high"] - wb["ci_low"]
    assert 0.5 < cw / ww < 2.0, f"widths diverge without clustering: {cw:.4f} vs {ww:.4f}"


def test_group_index_partitions_every_row_exactly_once():
    g = np.array(["b", "a", "b", "c", "a", "b"])
    uniq, members = M._group_index(g)
    assert list(uniq) == ["a", "b", "c"]
    assert sorted(np.concatenate(members).tolist()) == list(range(len(g)))
    for u, m in zip(uniq, members):
        assert set(g[m]) == {u}


def test_compare_labels_window_level_tests_as_such():
    y, p1, p2, g = _duplicated(seed=5)
    out = M.compare(y, p1, p2, g, thr1=0.5, thr2=0.5, n_boot=200)
    assert "pr_auc_cluster_ci_low" in out and "pr_auc_window_level_ci_low" in out
    assert "delong_window_level_p" in out and "mcnemar_window_level_p" in out
    # Nothing carrying an independence assumption may be unlabelled.
    for k in out:
        if "delong" in k or "mcnemar" in k:
            assert "window_level" in k, f"{k} hides its independence assumption"


# --------------------------------------------------------------------------
# Natural-base-rate protocol scoring
# --------------------------------------------------------------------------
def test_audit_rows_add_a_natural_twin_at_the_real_base_rate():
    from src import phase_d as PD

    rng = np.random.default_rng(0)
    n_real, n_synth = 1000, 900
    # Real rows are 2% positive; synthetic rows are all positive, as SMOTE makes them.
    y = np.concatenate([(rng.random(n_real) < 0.02).astype(int), np.ones(n_synth, int)])
    p = rng.random(n_real + n_synth)
    natural = np.arange(n_real + n_synth) < n_real

    rows = PD._audit_rows("inflated", 1, 0, "lstm", y, p, {"n_synthetic": n_synth},
                          "desc", natural=natural)
    assert len(rows) == 2
    full, nat = rows
    assert full["protocol"] == "inflated"
    assert nat["protocol"] == "inflated_natural_test"
    assert nat["n_test"] == n_real
    assert nat["test_positive_rate"] < 0.05 < full["test_positive_rate"]
    # The lift is what makes the two comparable.
    for r in rows:
        assert r["pr_auc_lift"] == pytest.approx(r["pr_auc"] / r["test_positive_rate"])


def test_audit_rows_emit_only_one_row_when_there_is_no_resampling():
    from src import phase_d as PD

    y = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    p = np.linspace(0, 1, 8)
    rows = PD._audit_rows("correct", 4, 0, "lstm", y, p, {}, "desc", natural=None)
    assert len(rows) == 1 and rows[0]["protocol"] == "correct"
    assert rows[0]["scored_on"] == "the natural test set"


def test_audit_summary_attributes_on_roc_auc():
    from src import phase_d as PD

    rows = []
    for seed in range(3):
        for proto, roc, pr, base, acc in (
                ("inflated", 0.99, 0.99, 0.50, 0.98),
                ("inflated_natural_test", 0.99, 0.30, 0.02, 0.98),
                ("half_fixed", 0.95, 0.95, 0.51, 0.86),
                ("half_fixed_natural_test", 0.95, 0.20, 0.02, 0.86),
                ("correct", 0.75, 0.04, 0.011, 0.985)):
            rows.append({"horizon": 4, "protocol": proto, "seed": seed,
                         "roc_auc": roc, "pr_auc": pr, "pr_auc_lift": pr / base,
                         "test_positive_rate": base, "accuracy": acc,
                         "recall": 0.1, "f1": 0.1, "n_test": 1000})
    agg, dec = PD.audit_summary(pd.DataFrame(rows))
    d = dec.iloc[0]
    # ROC-AUC attribution must be bounded and sum to one.
    assert 0 <= d["share_chronological_split"] <= 1
    assert d["share_chronological_split"] + d["share_resampling_inside_train"] == pytest.approx(1.0)
    # PR-AUC attribution exists and uses the natural-base-rate variants.
    assert d["natural_inflated_pr_auc"] == pytest.approx(0.30)
    assert d["natural_inflated_base_rate"] == pytest.approx(0.02)
    assert "majority_class_accuracy" in d


# --------------------------------------------------------------------------
# Precision@k, hand computed
# --------------------------------------------------------------------------
def test_precision_at_k_hand_computed():
    #        scores: .95  .90  .80  .70  .60  .50  .40  .30  .20  .10
    y = np.array([1,    0,   1,   0,   0,   1,   0,   0,   0,   0])
    p = np.array([.95, .90, .80, .70, .60, .50, .40, .30, .20, .10])
    #  top 3 = .95(1) .90(0) .80(1) -> tp 2, precision 2/3, recall 2/3
    r = M.precision_at_k(y, p, 3)
    assert r["tp"] == 2
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["lift"] == pytest.approx((2 / 3) / 0.3)
    #  top 1 = .95(1) -> perfect precision, one third of the positives
    r1 = M.precision_at_k(y, p, 1)
    assert r1["precision"] == pytest.approx(1.0) and r1["recall"] == pytest.approx(1 / 3)
    #  k = n recovers the base rate and full recall
    rn = M.precision_at_k(y, p, len(y))
    assert rn["precision"] == pytest.approx(0.3) and rn["recall"] == pytest.approx(1.0)


def test_precision_at_k_clamps_k_and_handles_no_positives():
    y = np.zeros(10, int); p = np.random.default_rng(0).random(10)
    assert np.isnan(M.precision_at_k(y, p, 5)["precision"])
    y2 = np.array([1, 0, 0]); p2 = np.array([.9, .5, .1])
    assert M.precision_at_k(y2, p2, 99)["k"] == 3


def test_firm_level_takes_each_firm_highest_score_and_ever_failed():
    y = np.array([0, 1, 0, 0, 0])
    p = np.array([.2, .4, .9, .1, .3])
    g = np.array(["a", "a", "b", "b", "c"])
    fy, fp = M.firm_level_scores(y, p, g)
    assert list(fy) == [1, 0, 0]          # firm a failed, b and c did not
    assert list(fp) == pytest.approx([0.4, 0.9, 0.3])


def test_capture_curve_is_monotone_and_ends_at_one():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    p = rng.random(500) + 0.5 * y
    cc = M.capture_curve(y, p)
    assert (np.diff(cc["frac_captured"]) >= -1e-12).all()
    assert cc["frac_captured"].iloc[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Calibration and ECE
# --------------------------------------------------------------------------
def test_ece_is_zero_for_a_perfectly_calibrated_score():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)       # by construction P(y=1) = p
    _, st = M.calibration(y, p, n_bins=10)
    assert st["ece"] < 0.01, f"ECE {st['ece']:.4f} should be ~0 for a calibrated score"
    assert st["brier"] == pytest.approx(float(np.mean((p - y) ** 2)))


def test_ece_is_large_for_a_badly_calibrated_score():
    rng = np.random.default_rng(0)
    n = 50_000
    y = (rng.random(n) < 0.02).astype(int)          # 2% really fail
    p = np.clip(rng.random(n) * 0.4 + 0.5, 0, 1)    # but the model says ~70%
    _, st = M.calibration(y, p, n_bins=10)
    assert st["ece"] > 0.5, f"ECE {st['ece']:.4f} should be large when p >> observed"


def test_calibration_uses_quantile_bins_so_a_rare_event_still_spreads():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(9900) * 0.01, rng.random(100)])
    y = (rng.random(10_000) < p).astype(int)
    tab, st = M.calibration(y, p, n_bins=10, strategy="quantile")
    assert st["n_bins_used"] >= 8, "quantile bins should not collapse on a skewed score"
    assert tab["n"].min() > 0


def test_logistic_scale_is_monotone_and_bounded():
    x = np.array([-500.0, -3, 0, 2, 17, 900])
    s = M.logistic_scale(x)
    assert (np.diff(s) > 0).all()
    assert (s > 0).all() and (s < 1).all()
