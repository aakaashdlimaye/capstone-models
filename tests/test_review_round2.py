"""The second round of review fixes: Platt scaling, the distribution-shift
flag's threshold, multi-seed tabular models, and the generated README block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import metrics as M


# --------------------------------------------------------------------------
# Platt scaling replaces the meaningless ECE on a raw score
# --------------------------------------------------------------------------
def test_platt_scaling_is_monotone_so_rankings_are_untouched():
    rng = np.random.default_rng(0)
    yv = (rng.random(4000) < 0.05).astype(int)
    pv = rng.normal(size=4000) + 1.5 * yv
    yt = (rng.random(2000) < 0.05).astype(int)
    pt = rng.normal(size=2000) + 1.5 * yt

    q = M.platt_scale(yv, pv, pt)
    order_before = np.argsort(pt)
    assert np.all(np.diff(q[order_before]) >= -1e-12), "Platt scaling must be monotone"
    assert M.roc_auc(yt, q) == pytest.approx(M.roc_auc(yt, pt))
    assert M.pr_auc(yt, q) == pytest.approx(M.pr_auc(yt, pt))
    assert (q > 0).all() and (q < 1).all()


def test_platt_scaling_puts_a_raw_score_on_a_sane_probability_scale():
    """A raw score has a meaningless ECE; the scaled version should not."""
    rng = np.random.default_rng(1)
    base = 0.02
    yv = (rng.random(20_000) < base).astype(int)
    pv = rng.normal(size=20_000) * 3 + 2.0 * yv + 50     # an Altman-like score
    yt = (rng.random(20_000) < base).astype(int)
    pt = rng.normal(size=20_000) * 3 + 2.0 * yt + 50

    _, raw = M.calibration(yt, M.logistic_scale(pt))
    _, scaled = M.calibration(yt, M.platt_scale(yv, pv, pt))
    assert raw["ece"] > 0.3, "the raw score should look wildly miscalibrated"
    assert scaled["ece"] < 0.05, f"Platt scaling should fix the level, got {scaled['ece']:.4f}"


def test_platt_scaling_falls_back_when_validation_has_one_class():
    yv = np.zeros(100, int)
    pv = np.random.default_rng(0).normal(size=100)
    pt = np.random.default_rng(1).normal(size=50)
    q = M.platt_scale(yv, pv, pt)
    assert q.shape == pt.shape and (q > 0).all() and (q < 1).all()


# --------------------------------------------------------------------------
# The distribution-shift flag must not fire on noise
# --------------------------------------------------------------------------
def test_shift_flag_requires_a_material_interval_backed_recovery():
    from src import phase_e as PE

    assert PE.MIN_RECOVERY_SHARE > 0, "a zero floor lets the flag fire on noise"

    def verdict(recovery, drop, ci_low):
        share = recovery / abs(drop)
        excludes = ci_low > 0
        return bool(excludes and share >= PE.MIN_RECOVERY_SHARE)

    # The case that was wrongly flagged True: a 5e-5 move on a 0.038 drop.
    assert not verdict(recovery=0.00005, drop=-0.03765, ci_low=-0.004)
    # Sign alone is not enough; the interval must clear zero.
    assert not verdict(recovery=0.020, drop=-0.038, ci_low=-0.001)
    # Large and clear of zero.
    assert verdict(recovery=0.020, drop=-0.038, ci_low=0.004)
    # Clear of zero but negligible against the drop.
    assert not verdict(recovery=0.0005, drop=-0.038, ci_low=0.0001)


# --------------------------------------------------------------------------
# Multi-seed tabular models
# --------------------------------------------------------------------------
def test_tabular_seed_key_is_distinct_per_seed():
    from src import tabular as TAB

    keys = {TAB.seed_key("xgb_t0_29", 4, s) for s in C.SEEDS}
    assert len(keys) == len(C.SEEDS)
    assert "xgb_t0_29_4" not in keys, "the ensemble key must not collide with a seed key"


def test_growth_indices_name_the_year_on_year_ratios():
    from src import phase_h as PH

    names = [C.FEATURE_NAMES[i] for i in PH.GROWTH_IDX]
    assert names == ["r21_revenue_growth", "r22_net_income_growth",
                     "r23_assets_growth", "r24_equity_growth"]
    assert len(PH.NON_GROWTH_IDX) == C.N_FEATURES - 4
    assert not set(PH.GROWTH_IDX) & set(PH.NON_GROWTH_IDX)


# --------------------------------------------------------------------------
# The README block is generated, not typed
# --------------------------------------------------------------------------
def test_readme_has_generated_markers():
    txt = (C.REPO / "README.md").read_text(encoding="utf-8")
    assert txt.count(report_start()) == 1 and txt.count(report_end()) == 1
    body = txt.split(report_start(), 1)[1].split(report_end(), 1)[0]
    assert "do not edit" in body.lower()


def report_start():
    from src import report as R
    return R.README_START


def report_end():
    from src import report as R
    return R.README_END


@pytest.mark.skipif(not (C.RESULTS / "deep_all.csv").exists(),
                    reason="no full run on disk")
def test_readme_findings_agree_with_the_csvs():
    """The generated block must quote the numbers that are actually on disk."""
    from src import report as R

    deep = pd.read_csv(C.RESULTS / "deep_all.csv")
    base = pd.read_csv(C.RESULTS / "baselines_all.csv")
    body = "\n".join(R.readme_findings())

    d4 = deep[deep["horizon"] == 4].set_index("arch")
    best_deep = d4["pr_auc_ensemble"].idxmax()
    assert f"{d4.loc[best_deep, 'pr_auc_ensemble']:.4f}" in body

    b4 = base[base["horizon"] == 4].set_index("model")
    best_ml = b4[b4["family"] == "ml"]["pr_auc"].idxmax()
    assert f"{b4.loc[best_ml, 'pr_auc']:.4f}" in body
    assert f"{b4.loc['altman_zdp', 'pr_auc']:.4f}" in body
    # The stale hand-written claims must be gone.
    assert "54% is the static formulation" not in body
