"""The loader's contract: shapes, names, labels, masks, and the row set."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as C
from src import data as D


@pytest.fixture(scope="module")
def splits():
    return D.load_all(full=True)


def test_shapes_and_counts(splits):
    for name, d in splits.items():
        assert d.X.shape == (C.EXPECTED_N[name], C.WINDOW_LEN, C.N_FEATURES)
        assert d.y.shape == (C.EXPECTED_N[name], 4)
        assert d.mask.shape == d.X.shape
        assert d.indicators.shape == (C.EXPECTED_N[name], C.WINDOW_LEN, 2)


def test_feature_names_in_order(splits):
    for d in splits.values():
        assert d.feature_names == C.FEATURE_NAMES
        assert d.feature_names[3] == "r04_wc_to_ta"       # Altman X1
        assert d.feature_names[15] == "r16_asset_turnover"  # Altman X5
        assert [d.feature_names[i] for i in C.ALTMAN_IDX] == [
            "r04_wc_to_ta", "r10_re_to_ta", "r09_ebit_to_ta",
            "r14_equity_to_liabilities", "r16_asset_turnover"]


def test_no_nan_in_X(splits):
    for d in splits.values():
        assert np.isfinite(d.X).all()


def test_labels_binary_and_monotone(splits):
    for d in splits.values():
        assert set(np.unique(d.y)).issubset({0, 1})
        assert (np.diff(d.y.astype(int), axis=1) >= 0).all()


def test_positive_counts_match_the_report(splits):
    for name, d in splits.items():
        got = {h: int(d.y[:, h - 1].sum()) for h in C.HORIZONS}
        assert got == C.EXPECTED_POSITIVES[name]


def test_mask_agrees_with_unscaled_nan(splits):
    for d in splits.values():
        assert (np.isnan(d.X_unscaled) == (d.mask == 0)).all()
        assert np.allclose(d.X[d.mask == 0], 0.0)


def test_loader_rejects_a_scrambled_label_column(splits):
    d = splits["val"]
    bad = D.SplitData(**{**d.__dict__, "y": d.y[:, ::-1].copy()})
    with pytest.raises(AssertionError, match="monotone"):
        D.assert_contract(bad, list(C.HORIZONS), full=True)


def test_loader_rejects_wrong_feature_names(splits):
    d = splits["val"]
    names = list(d.feature_names)
    names[0] = "not_a_ratio"
    bad = D.SplitData(**{**d.__dict__, "feature_names": names})
    with pytest.raises(AssertionError, match="feature_names"):
        D.assert_contract(bad, list(C.HORIZONS), full=True)


def test_row_set_contract():
    rs = D.build_row_sets(full=True)
    assert len(rs) == C.EXPECTED_TOTAL_WINDOWS
    assert rs.duplicated(["cik", "end_quarter"]).sum() == 0
    assert rs.groupby(["cik", "end_quarter"])["split"].nunique().max() == 1
    assert set(rs["split"]) == {"train", "val", "test"}


def test_target_selection(splits):
    d = splits["test"]
    for h in C.HORIZONS:
        assert np.array_equal(d.target(h), d.y[:, h - 1].astype(np.float32))
    with pytest.raises(ValueError):
        d.target(5)


def test_subset_keeps_arrays_aligned(splits):
    d = splits["val"]
    idx = np.array([0, 5, 17, 100])
    s = d.subset(idx)
    assert s.n == 4
    assert (s.cik == d.cik[idx]).all()
    assert np.array_equal(s.X, d.X[idx])
    assert np.array_equal(s.y, d.y[idx])


def test_flat_window_width(splits):
    d = splits["val"]
    assert d.flat().shape == (d.n, C.WINDOW_LEN * C.N_FEATURES + 2)
    assert d.flat(with_indicators=False).shape == (d.n, C.WINDOW_LEN * C.N_FEATURES)
    assert len(D.flat_feature_names()) == C.WINDOW_LEN * C.N_FEATURES + 2


def test_embargo_drops_only_boundary_windows(splits):
    man = D.load_manifest(full=True).set_index(["cik", "end_quarter"])
    for name in ("train", "val", "test"):
        d = splits[name]
        keep = D.embargo_mask(d, full=True)
        assert keep.shape == (d.n,)
        if name == "train":
            assert keep.all(), "embargo must not touch train, or the scaler changes"
        else:
            import pandas as pd

            idx = pd.MultiIndex.from_arrays([d.cik, d.end_quarter])
            start = man["start_quarter_idx"].reindex(idx).to_numpy()
            assert start[keep].min() > start[~keep].max() if (~keep).any() else True
