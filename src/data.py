"""Loading the frozen tensors, asserting the contract, and the row-set table.

Nothing here fits anything.  The scaler that produced `X` was fitted on train
inside the dataset repository and is never refitted (leakage rule 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class SplitData:
    """One split's tensors plus its index arrays."""

    split: str
    X: np.ndarray              # (n, 8, 29) float32, z-scored on train parameters
    X_unscaled: np.ndarray     # (n, 8, 29) float32, winsorised only
    mask: np.ndarray           # (n, 8, 29) uint8, 1 = observed
    y: np.ndarray              # (n, 4) int8, column h-1 is horizon h
    indicators: np.ndarray     # (n, 8, 2) float32, has_inventory / has_debt
    cik: np.ndarray            # (n,) str
    start_quarter: np.ndarray  # (n,) str
    end_quarter: np.ndarray    # (n,) str
    is_positive_firm: np.ndarray
    feature_names: list[str]
    indicator_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def target(self, h: int) -> np.ndarray:
        """Labels for horizon h (1..4) as float32."""
        if h not in C.HORIZONS:
            raise ValueError(f"horizon must be one of {C.HORIZONS}, got {h}")
        return self.y[:, h - 1].astype(np.float32)

    def all_targets(self) -> np.ndarray:
        """All four horizons, (n, 4) float32 — for the multi-horizon head."""
        return self.y.astype(np.float32)

    def window_index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([self.cik, self.end_quarter], names=["cik", "end_quarter"])

    def frame(self) -> pd.DataFrame:
        """Index arrays as a DataFrame, in tensor row order."""
        return pd.DataFrame(
            {
                "row": np.arange(self.n),
                "cik": self.cik,
                "start_quarter": self.start_quarter,
                "end_quarter": self.end_quarter,
                "split": self.split,
                "is_positive_firm": self.is_positive_firm,
                "y1": self.y[:, 0],
                "y2": self.y[:, 1],
                "y3": self.y[:, 2],
                "y4": self.y[:, 3],
            }
        )

    def subset(self, idx: np.ndarray) -> "SplitData":
        """A row subset that keeps every array aligned."""
        idx = np.asarray(idx)
        return SplitData(
            split=self.split,
            X=self.X[idx],
            X_unscaled=self.X_unscaled[idx],
            mask=self.mask[idx],
            y=self.y[idx],
            indicators=self.indicators[idx],
            cik=self.cik[idx],
            start_quarter=self.start_quarter[idx],
            end_quarter=self.end_quarter[idx],
            is_positive_firm=self.is_positive_firm[idx],
            feature_names=self.feature_names,
            indicator_names=self.indicator_names,
        )

    def features(self, cols: list[int] | None = None) -> np.ndarray:
        """X restricted to a feature subset (used by decomposition models C/D)."""
        return self.X if cols is None else self.X[:, :, cols]

    def flat(self, with_indicators: bool = True) -> np.ndarray:
        """The 8x29 window flattened to 232 features, plus the two t-0 indicators."""
        flat = self.X.reshape(self.n, -1)
        if with_indicators:
            flat = np.concatenate([flat, self.indicators[:, -1, :]], axis=1)
        return flat.astype(np.float32)


def flat_feature_names(with_indicators: bool = True) -> list[str]:
    names = [f"{f}@t-{7 - t}" for t in range(C.WINDOW_LEN) for f in C.FEATURE_NAMES]
    if with_indicators:
        names += [f"{i}@t-0" for i in C.INDICATOR_NAMES]
    return names


# --------------------------------------------------------------------------
# Loading with assertions
# --------------------------------------------------------------------------
def load_split(split: str, full: bool = True, strict: bool = True) -> SplitData:
    """Load one split and fail loudly on any contract violation."""
    path = C.sequences_path(split, full=full)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the dataset pipeline first")
    if path.stat().st_size < 1000:
        raise OSError(f"{path} is {path.stat().st_size} bytes — this is a Git LFS pointer; run `git lfs pull`")

    with np.load(path, allow_pickle=False) as z:
        d = SplitData(
            split=split,
            X=z["X"].astype(np.float32),
            X_unscaled=z["X_unscaled"].astype(np.float32),
            mask=z["mask"].astype(np.uint8),
            y=z["y"].astype(np.int8),
            indicators=z["indicators"].astype(np.float32),
            cik=np.asarray(z["cik"], dtype=str),
            start_quarter=np.asarray(z["start_quarter"], dtype=str),
            end_quarter=np.asarray(z["end_quarter"], dtype=str),
            is_positive_firm=z["is_positive_firm"].astype(np.int8),
            feature_names=[str(s) for s in z["feature_names"]],
            indicator_names=[str(s) for s in z["indicator_names"]],
        )
        horizons = [int(h) for h in z["horizons"]]

    assert_contract(d, horizons, full=full, strict=strict)
    return d


def assert_contract(d: SplitData, horizons: list[int], full: bool = True, strict: bool = True) -> None:
    n = d.n
    assert d.X.shape == (n, C.WINDOW_LEN, C.N_FEATURES), f"X shape {d.X.shape}"
    assert d.X_unscaled.shape == d.X.shape, "X_unscaled shape mismatch"
    assert d.mask.shape == d.X.shape, "mask shape mismatch"
    assert d.y.shape == (n, 4), f"y shape {d.y.shape}"
    assert d.indicators.shape == (n, C.WINDOW_LEN, 2), f"indicators shape {d.indicators.shape}"
    for arr, name in ((d.cik, "cik"), (d.start_quarter, "start_quarter"),
                      (d.end_quarter, "end_quarter"), (d.is_positive_firm, "is_positive_firm")):
        assert arr.shape == (n,), f"{name} shape {arr.shape}"

    assert d.feature_names == C.FEATURE_NAMES, (
        "feature_names does not match the expected 29 names in order; "
        f"first mismatch at {next((i for i, (a, b) in enumerate(zip(d.feature_names, C.FEATURE_NAMES)) if a != b), None)}"
    )
    assert d.indicator_names == C.INDICATOR_NAMES, f"indicator_names {d.indicator_names}"
    assert horizons == list(C.HORIZONS), f"horizons {horizons}"

    assert not np.isnan(d.X).any(), "X contains NaN"
    assert np.isfinite(d.X).all(), "X contains inf"
    # X_unscaled keeps NaN exactly where the mask says the cell was imputed or
    # structurally undefined; X carries the train mean (0 after scaling) there.
    nan_u = np.isnan(d.X_unscaled)
    assert (nan_u == (d.mask == 0)).all(), "X_unscaled NaN pattern does not match mask == 0"
    assert np.allclose(d.X[d.mask == 0], 0.0), "X is not 0 (the train mean) at masked cells"
    assert set(np.unique(d.y)).issubset({0, 1}), f"y values {np.unique(d.y)}"
    assert set(np.unique(d.mask)).issubset({0, 1}), f"mask values {np.unique(d.mask)}"

    # Labels must be monotone in the horizon: filing within h quarters implies
    # filing within h+1.  A violation would mean the label columns are scrambled.
    mono = (np.diff(d.y.astype(int), axis=1) >= 0).all()
    assert mono, "y is not monotone across horizons — label columns may be misordered"

    pos = {h: int(d.y[:, h - 1].sum()) for h in C.HORIZONS}
    exp = C.EXPECTED_POSITIVES[d.split]
    if strict:
        assert pos == exp, f"{d.split} positives {pos} != expected {exp}"
        if full:
            assert n == C.EXPECTED_N[d.split], f"{d.split} n={n} != {C.EXPECTED_N[d.split]}"
            firms = len(np.unique(d.cik[d.is_positive_firm == 1]))
            assert firms == C.EXPECTED_POSITIVE_FIRMS[d.split], (
                f"{d.split} positive firms {firms} != {C.EXPECTED_POSITIVE_FIRMS[d.split]}"
            )


@lru_cache(maxsize=4)
def load_all(full: bool = True, strict: bool = True) -> dict[str, SplitData]:
    """All three splits, cached so repeated phases do not re-read 44 MB."""
    return {s: load_split(s, full=full, strict=strict) for s in ("train", "val", "test")}


@lru_cache(maxsize=4)
def load_manifest(full: bool = True) -> pd.DataFrame:
    m = pd.read_csv(C.manifest_path(full=full), dtype={"cik": str})
    m["cik"] = m["cik"].astype(str)
    return m


# --------------------------------------------------------------------------
# The row-set contract
# --------------------------------------------------------------------------
def build_row_sets(full: bool = True, out: Path | None = None) -> pd.DataFrame:
    """One row per evaluated window, with its panel join key.

    Every model — classical or deep — scores exactly these rows.  That is what
    makes the A->B->C->D gaps attributable to the model rather than the sample.
    """
    splits = load_all(full=full)
    man = load_manifest(full=full)

    frames = []
    for name, d in splits.items():
        f = d.frame()
        f["tensor_row"] = f.pop("row")
        frames.append(f)
    rs = pd.concat(frames, ignore_index=True)

    # The panel join key: (cik, quarter) at the window's end quarter.
    rs["panel_cik"] = rs["cik"]
    rs["panel_quarter"] = rs["end_quarter"]

    man_keyed = man.set_index(["cik", "end_quarter"])
    idx = pd.MultiIndex.from_arrays([rs["cik"], rs["end_quarter"]])
    for col in ("end_quarter_idx", "start_quarter_idx", "event_date",
                "quarters_to_event", "n_observed_quarters", "pct_cells_present", "company"):
        rs[col] = man_keyed[col].reindex(idx).to_numpy()

    # Sanity: the manifest and the tensors must agree on split and labels.
    man_split = man_keyed["split"].reindex(idx).to_numpy()
    assert (man_split == rs["split"].to_numpy()).all(), "split disagreement between manifest and tensors"
    for h in C.HORIZONS:
        man_y = man_keyed[f"y{h}"].reindex(idx).to_numpy()
        assert (man_y == rs[f"y{h}"].to_numpy()).all(), f"y{h} disagreement between manifest and tensors"

    rs = rs[[
        "cik", "company", "start_quarter", "end_quarter", "end_quarter_idx", "start_quarter_idx",
        "split", "tensor_row", "panel_cik", "panel_quarter", "is_positive_firm",
        "y1", "y2", "y3", "y4", "event_date", "quarters_to_event",
        "n_observed_quarters", "pct_cells_present",
    ]]

    dup = rs.duplicated(subset=["cik", "end_quarter"]).sum()
    assert dup == 0, f"{dup} duplicate (cik, end_quarter) pairs in the row set"
    if full:
        assert len(rs) == C.EXPECTED_TOTAL_WINDOWS, f"row set has {len(rs)} rows, expected {C.EXPECTED_TOTAL_WINDOWS}"
    # No window index may appear in more than one split.
    per_index_splits = rs.groupby(["cik", "end_quarter"])["split"].nunique()
    assert (per_index_splits == 1).all(), "a window index appears in more than one split"

    if out is None:
        out = C.RESULTS / "row_sets.csv"
    rs.to_csv(out, index=False)
    return rs


def class_balance(full: bool = True) -> pd.DataFrame:
    """Windows, positives and rates per split per horizon, plus positive firms."""
    splits = load_all(full=full)
    rows = []
    for name in ("train", "val", "test"):
        d = splits[name]
        row = {"split": name, "windows": d.n,
               "firms": int(len(np.unique(d.cik))),
               "positive_firms": int(len(np.unique(d.cik[d.is_positive_firm == 1])))}
        for h in C.HORIZONS:
            yv = d.y[:, h - 1]
            row[f"pos_y{h}"] = int(yv.sum())
            row[f"rate_y{h}"] = float(yv.mean())
            row[f"pos_firms_y{h}"] = int(len(np.unique(d.cik[yv == 1])))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Embargo variant (Phase G.2) — derived, not re-piped
# --------------------------------------------------------------------------
def embargo_mask(d: SplitData, full: bool = True) -> np.ndarray:
    """Windows the dataset repo's `--embargo` flag would keep.

    It drops val windows whose input starts on or before the train cut-off and
    test windows whose input starts before the test period.  Train windows are
    untouched, so the train-fitted scaler is unchanged and the existing tensors
    can be reused directly.
    """
    man = load_manifest(full=full)
    q = man.drop_duplicates(["cik", "end_quarter"]).set_index(["cik", "end_quarter"])
    train_end_idx = int(q.loc[q["split"] == "train", "end_quarter_idx"].max())
    test_start_idx = int(q.loc[q["split"] == "test", "end_quarter_idx"].min())

    idx = pd.MultiIndex.from_arrays([d.cik, d.end_quarter])
    start_idx = q["start_quarter_idx"].reindex(idx).to_numpy()
    if d.split == "val":
        bad = start_idx <= train_end_idx
    elif d.split == "test":
        bad = start_idx < test_start_idx
    else:
        bad = np.zeros(d.n, dtype=bool)
    return ~bad
