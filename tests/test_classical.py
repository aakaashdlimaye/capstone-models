"""Annualisation and the published scoring formulas.

The Z''/Z' tests compare against a worked example computed by hand from the
coefficients in the original papers, so a coefficient typo cannot pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import classical as K


# --------------------------------------------------------------------------
# Trailing-four-quarter annualisation
# --------------------------------------------------------------------------
def _panel(rows):
    return pd.DataFrame(rows)


def test_ttm_sums_four_contiguous_quarters():
    p = _panel([{"cik": "1", "quarter_idx": q, "Revenue": 10.0 * (q - 99)}
                for q in range(100, 108)])
    out = K.annualise(p, cols=["Revenue"]).sort_values("quarter_idx").set_index("quarter_idx")
    # q=103 is the first quarter with four contiguous observations: 10+20+30+40.
    assert out.loc[103, "Revenue_ttm"] == pytest.approx(100.0)
    assert not out.loc[103, "Revenue_ttm_fallback"]
    # q=102 has only three prior quarters, so it takes the x4 fallback (4 x 30)
    # rather than silently summing a short window.
    assert out.loc[102, "Revenue_ttm_fallback"]
    assert out.loc[102, "Revenue_ttm"] == pytest.approx(120.0)
    out = out.reset_index()
    assert out.loc[out.quarter_idx == 107, "Revenue_ttm"].iloc[0] == pytest.approx(
        50.0 + 60.0 + 70.0 + 80.0)


def test_ttm_falls_back_to_times_four_and_flags_it():
    p = _panel([{"cik": "1", "quarter_idx": 100, "Revenue": 7.0}])
    out = K.annualise(p, cols=["Revenue"])
    assert out["Revenue_ttm_fallback"].iloc[0]
    assert out["Revenue_ttm"].iloc[0] == pytest.approx(28.0)


def test_ttm_does_not_splice_across_a_gap():
    # Quarters 100, 101, 102 then a gap, then 104.  Quarter 104 has only three
    # of its four look-back quarters present, so it must fall back, not sum
    # 100+101+102+104 as a positional shift(4) would.
    p = _panel([{"cik": "1", "quarter_idx": q, "Revenue": 10.0}
                for q in (100, 101, 102, 104)])
    out = K.annualise(p, cols=["Revenue"]).set_index("quarter_idx")
    assert out.loc[104, "Revenue_ttm_fallback"]
    assert out.loc[104, "Revenue_ttm"] == pytest.approx(40.0)     # 4 x 10, the fallback


def test_ttm_is_per_firm():
    p = _panel([{"cik": c, "quarter_idx": q, "Revenue": v}
                for c, v in (("1", 10.0), ("2", 100.0))
                for q in range(100, 104)])
    out = K.annualise(p, cols=["Revenue"]).set_index(["cik", "quarter_idx"])
    assert out.loc[("1", 103), "Revenue_ttm"] == pytest.approx(40.0)
    assert out.loc[("2", 103), "Revenue_ttm"] == pytest.approx(400.0)


def test_lag4_matches_on_quarter_index():
    p = _panel([{"cik": "1", "quarter_idx": q, "v": float(q)} for q in (100, 101, 104, 105)])
    got = K.lag4(p, "v", "v4").to_numpy()
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert got[2] == 100.0 and got[3] == 101.0


# --------------------------------------------------------------------------
# The published formulas, against hand-computed values
# --------------------------------------------------------------------------
@pytest.fixture
def worked_example():
    """One firm, chosen so every term is easy to verify by hand.

        X1 = WC/TA               = 0.20
        X2 = RE/TA               = 0.10
        X3 = EBIT(annual)/TA     = 0.05
        X4 = book equity / TL    = 0.50
        X5 = Sales(annual)/TA    = 1.00
    """
    return pd.DataFrame([{"X1": 0.20, "X2": 0.10, "X3": 0.05, "X4": 0.50, "X5": 1.00}])


def test_z_double_prime_matches_hand_calculation(worked_example):
    # 6.56*0.20 + 3.26*0.10 + 6.72*0.05 + 1.05*0.50
    #   = 1.312 + 0.326 + 0.336 + 0.525 = 2.499
    got = K.altman_z_double_prime(worked_example)[0]
    assert got == pytest.approx(2.499, abs=1e-9)
    assert K.Z_DPRIME_COEF == {"X1": 6.56, "X2": 3.26, "X3": 6.72, "X4": 1.05}
    assert "X5" not in K.Z_DPRIME_COEF, "Z'' is the four-variable form; X5 is dropped"
    assert K.Z_DPRIME_CUTS == (1.10, 2.60)


def test_z_prime_matches_hand_calculation(worked_example):
    # 0.717*0.20 + 0.847*0.10 + 3.107*0.05 + 0.420*0.50 + 0.998*1.00
    #   = 0.1434 + 0.0847 + 0.15535 + 0.21 + 0.998 = 1.59145
    got = K.altman_z_prime(worked_example)[0]
    assert got == pytest.approx(1.59145, abs=1e-9)
    assert K.Z_PRIME_COEF == {"X1": 0.717, "X2": 0.847, "X3": 3.107,
                              "X4": 0.420, "X5": 0.998}
    assert K.Z_PRIME_CUTS == (1.23, 2.90)


def test_altman_zones_use_the_published_cutoffs():
    z = np.array([0.5, 1.10, 2.0, 2.60, 4.0])
    zone = K.altman_zone(z, K.Z_DPRIME_CUTS)
    assert list(zone) == ["distress", "grey", "grey", "grey", "safe"]


def test_zmijewski_matches_hand_calculation():
    df = pd.DataFrame([{"NITA": -0.10, "TLTA": 0.80, "CACL": 1.50}])
    # -4.3 - 4.5*(-0.10) + 5.7*0.80 - 0.004*1.50
    #   = -4.3 + 0.45 + 4.56 - 0.006 = 0.704
    assert K.zmijewski(df)[0] == pytest.approx(0.704, abs=1e-9)


def test_ohlson_matches_hand_calculation():
    df = pd.DataFrame([{"SIZE": 10.0, "TLTA": 0.80, "WCTA": 0.20, "CLCA": 0.50,
                        "OENEG": 0.0, "NITA": -0.10, "FUTL": 0.05, "INTWO": 1.0,
                        "CHIN": -0.25}])
    expected = (-1.32 - 0.407 * 10.0 + 6.03 * 0.80 - 1.43 * 0.20 + 0.0757 * 0.50
                - 1.72 * 0.0 - 2.37 * (-0.10) - 1.83 * 0.05 + 0.285 * 1.0
                - 0.521 * (-0.25))
    assert K.ohlson_o(df)[0] == pytest.approx(expected, abs=1e-12)


def test_altman_is_a_health_score_so_distress_ranks_low():
    healthy = pd.DataFrame([{"X1": 0.4, "X2": 0.3, "X3": 0.1, "X4": 2.0, "X5": 1.2}])
    sick = pd.DataFrame([{"X1": -0.3, "X2": -0.8, "X3": -0.2, "X4": 0.05, "X5": 0.4}])
    assert K.altman_z_double_prime(healthy)[0] > K.altman_z_double_prime(sick)[0]
    assert K.altman_z_prime(healthy)[0] > K.altman_z_prime(sick)[0]


def test_safe_div_returns_nan_not_inf():
    out = K._safe_div([1.0, 1.0, np.nan], [0.0, 2.0, 2.0])
    assert np.isnan(out[0]) and out[1] == 0.5 and np.isnan(out[2])


def test_imputer_is_fitted_on_train_only():
    df = pd.DataFrame({"split": ["train"] * 4 + ["val", "test"],
                       "X1": [1.0, 2.0, 3.0, 4.0, 1000.0, np.nan]})
    med = K.fit_imputer(df[df.split == "train"], cols=["X1"])
    assert med["X1"] == pytest.approx(2.5)          # ignores the val/test rows
    out, rates = K.apply_imputer(df, med, cols=["X1"])
    assert out["X1"].iloc[-1] == pytest.approx(2.5)
    assert int(rates.loc[rates.variable == "X1", "n_missing"].iloc[0]) == 1
