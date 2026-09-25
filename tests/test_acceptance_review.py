"""Acceptance checks for the review fixes and Phase H, against the artefacts.

These assert properties of what is on disk, not of the code that wrote it:
every reported interval is firm-clustered, no PR-AUC is compared across unequal
base rates without a lift column, and no table mixes aggregation conventions
without naming them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C

pytestmark = pytest.mark.skipif(
    not (C.RESULTS / "deep_all.csv").exists(),
    reason="no full run on disk; run `python run_all.py` first",
)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(C.RESULTS / name)


def _have(name: str) -> bool:
    return (C.RESULTS / name).exists()


# --------------------------------------------------------------------------
# Intervals must be firm-clustered
# --------------------------------------------------------------------------
def test_every_decomposition_interval_is_cluster_based():
    d = _read("decomposition_all.csv")
    steps = d[d["step"].notna()]
    assert len(steps) > 0
    for col in ("pr_auc_cluster_ci_low", "pr_auc_cluster_ci_high",
                "roc_auc_cluster_ci_low", "roc_auc_cluster_ci_high",
                "pr_auc_cluster_p", "roc_auc_cluster_p", "n_clusters"):
        assert col in steps.columns, f"{col} missing — intervals are not cluster-based"
        assert steps[col].notna().all(), f"{col} has gaps"
    assert (steps["n_clusters"] > 100).all(), "cluster count looks like rows, not firms"


def test_anything_assuming_independent_rows_is_labelled_window_level():
    d = _read("decomposition_all.csv")
    for col in d.columns:
        if "delong" in col or "mcnemar" in col:
            assert "window_level" in col, f"{col} hides its independence assumption"


def test_cluster_intervals_are_not_narrower_than_window_intervals():
    """On real, clustered data the firm-level interval should be the wider one."""
    d = _read("decomposition_all.csv")
    steps = d[d["step"].notna()]
    cw = steps["pr_auc_cluster_ci_high"] - steps["pr_auc_cluster_ci_low"]
    ww = steps["pr_auc_window_level_ci_high"] - steps["pr_auc_window_level_ci_low"]
    frac = float((cw >= ww * 0.95).mean())
    assert frac > 0.7, (f"only {frac:.0%} of cluster intervals are at least as wide as "
                        "the window-level ones; clustering should not tighten them")


def test_decomposition_rows_declare_their_aggregation():
    d = _read("decomposition_all.csv")
    assert "aggregation" in d.columns
    assert set(d["aggregation"].dropna()) <= {"ensemble of 5 seeds", "single fit"}


# --------------------------------------------------------------------------
# PR-AUC must be compared only on equal base rates, or as a lift
# --------------------------------------------------------------------------
def test_protocol_audit_pr_auc_is_comparable_or_labelled():
    a = _read("protocol_audit.csv")
    assert "pr_auc_lift_mean" in a.columns, "PR-AUC across base rates needs a lift column"
    assert "test_positive_rate" in a.columns
    protos = set(a["protocol"])
    assert {"inflated_natural_test", "half_fixed_natural_test"} <= protos, (
        f"missing natural-base-rate variants; have {sorted(protos)}")
    for h in a["horizon"].unique():
        g = a[a["horizon"] == h].set_index("protocol")
        corr = g.loc["correct", "test_positive_rate"]
        nat = g.loc["inflated_natural_test", "test_positive_rate"]
        inf = g.loc["inflated", "test_positive_rate"]
        assert nat < 0.2, f"h={h}: the natural twin is still resampled ({nat:.3f})"
        assert inf > 0.3, f"h={h}: the inflated protocol is not balanced ({inf:.3f})"
        assert abs(nat - corr) < 0.05, (
            f"h={h}: natural twin {nat:.4f} and correct {corr:.4f} must share a base rate")


def test_protocol_decomposition_attributes_on_a_base_rate_invariant_metric():
    d = _read("protocol_audit_decomposition.csv")
    for col in ("share_chronological_split", "share_resampling_inside_train"):
        assert ((d[col] >= 0) & (d[col] <= 1)).all(), f"{col} outside [0, 1]"
    assert np.allclose(d["share_chronological_split"]
                       + d["share_resampling_inside_train"], 1.0)
    for col in ("inflated_roc_auc", "half_fixed_roc_auc", "correct_roc_auc",
                "roc_auc_lost_to_chronological_split",
                "roc_auc_lost_to_resampling_inside_train"):
        assert col in d.columns, f"{col} missing — attribution is not on ROC-AUC"
    if "pr_auc_natural_share_chronological_split" in d.columns:
        s = (d["pr_auc_natural_share_chronological_split"]
             + d["pr_auc_natural_share_resampling_inside_train"]).dropna()
        assert np.allclose(s, 1.0)


@pytest.mark.skipif(not _have("external_protocol_audit.csv"), reason="Phase G not run")
def test_external_protocol_audit_also_carries_lift():
    a = _read("external_protocol_audit.csv")
    assert "pr_auc_lift_mean" in a.columns
    assert "inflated_natural_test" in set(a["protocol"])


# --------------------------------------------------------------------------
# Aggregation conventions must be named
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["deep_all.csv", "deep_1.csv", "deep_4.csv"])
def test_no_deep_table_mixes_aggregation_conventions_unlabelled(name):
    d = _read(name)
    assert any(c.endswith("_mean") for c in d.columns), f"{name}: no per-seed columns"
    assert any(c.endswith("_ensemble") for c in d.columns), f"{name}: no ensemble columns"
    assert "aggregation_note" in d.columns, f"{name} does not name its conventions"
    assert not np.allclose(d["pr_auc_mean"], d["pr_auc_ensemble"], atol=1e-9), (
        f"{name}: the two conventions are identical, which cannot be right")


# --------------------------------------------------------------------------
# Phase H
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _have("h1_temporal_control.csv"), reason="Phase H not run")
def test_h1_answers_whether_the_window_helps():
    h1 = _read("h1_temporal_control.csv")
    assert set(h1["horizon"]) == set(C.HORIZONS)
    for col in ("t0_pr_auc", "window_pr_auc", "window_minus_t0_pr_auc",
                "pr_auc_cluster_ci_low", "pr_auc_cluster_ci_high", "window_helps"):
        assert col in h1.columns and h1[col].notna().all()
    assert (h1["pr_auc_cluster_ci_low"] <= h1["window_minus_t0_pr_auc"] + 1e-9).all()
    assert (h1["window_minus_t0_pr_auc"] <= h1["pr_auc_cluster_ci_high"] + 1e-9).all()


@pytest.mark.skipif(not _have("h2_static_deep_control.csv"), reason="Phase H not run")
def test_h2_separates_deep_from_temporal():
    h2 = _read("h2_static_deep_control.csv")
    for col in ("mlp_t0_pr_auc", "lstm_window_pr_auc", "lstm_minus_mlp_pr_auc",
                "time_axis_helps_beyond_nonlinearity"):
        assert col in h2.columns


@pytest.mark.skipif(not _have("h3_pairwise_significance.csv"), reason="Phase H not run")
def test_h3_tests_the_headline_architecture_comparisons():
    h3 = _read("h3_pairwise_significance.csv")
    pairs = set(zip(h3["model_a"], h3["model_b"]))
    assert ("xgboost", "transformer") in pairs
    assert ("transformer", "lstm") in pairs
    for col in ("pr_auc_cluster_ci_low", "pr_auc_cluster_ci_high",
                "delong_window_level_p", "mcnemar_window_level_p",
                "a_aggregation", "b_aggregation"):
        assert col in h3.columns


@pytest.mark.skipif(not _have("h4_precision_at_k.csv"), reason="Phase H not run")
def test_h4_reports_firm_level_precision_at_k():
    pk = _read("h4_precision_at_k.csv")
    assert set(pk["level"]) == {"window", "firm"}
    assert {"k=50", "k=100", "k=200", "top 1%", "top 5%"} <= set(pk["budget"])
    good = pk[pk["precision"].notna()]
    assert ((good["precision"] >= 0) & (good["precision"] <= 1)).all()
    assert ((good["recall"] >= 0) & (good["recall"] <= 1)).all()
    for h in pk["horizon"].unique():
        w = pk[(pk.horizon == h) & (pk.level == "window")]["n"].max()
        f = pk[(pk.horizon == h) & (pk.level == "firm")]["n"].max()
        assert f < w, f"h={h}: firm rows {f} should be fewer than window rows {w}"


@pytest.mark.skipif(not _have("h5_calibration.csv"), reason="Phase H not run")
def test_h5_reports_brier_and_ece():
    cal = _read("h5_calibration.csv")
    for col in ("brier", "ece", "is_probability", "base_rate"):
        assert col in cal.columns and cal[col].notna().all()
    assert ((cal["ece"] >= 0) & (cal["ece"] <= 1)).all()
    assert (cal["brier"] >= 0).all()


@pytest.mark.skipif(not _have("h6_sector_breakdown.csv"), reason="Phase H not run")
def test_h6_says_too_few_rather_than_reporting_a_thin_slice():
    sec = _read("h6_sector_breakdown.csv")
    assert "status" in sec.columns
    thin = sec[sec["n_positive"] < 20]
    if len(thin):
        assert thin["status"].astype(str).str.startswith("too few").all()
        assert thin["pr_auc"].isna().all(), "a slice below the threshold reported a number"
    reported = sec[sec["status"] == "reported"]
    if len(reported):
        assert (reported["n_positive"] >= 20).all()
        assert reported["pr_auc_cluster_ci_low"].notna().all()


@pytest.mark.skipif(not _have("h6_size_breakdown.csv"), reason="Phase H not run")
def test_h6_size_terciles_are_cut_on_the_train_period():
    import json

    notes = json.loads((C.RESULTS / "h6_breakdown_notes.json").read_text(encoding="utf-8"))
    cuts = notes["train_asset_tercile_cuts"]
    assert len(cuts) == 2 and cuts[0] < cuts[1]
    size = _read("h6_size_breakdown.csv")
    assert set(size["group"]) == {"small", "mid", "large"}
