"""Classical scoring formulas computed from levels, on the row-set contract.

Every score is evaluated on the window's **end quarter** (the `t-0` slice), so
Altman, Ohlson and Zmijewski see exactly the windows the deep models see.

Annualisation matters here and nowhere else.  The panel's flows are quarterly
(dataset `docs/DECISIONS.md` 4.1) while Altman's X3/X5, Ohlson's NI/TA and
FFO/TL and Zmijewski's NI/TA are annual flows over a stock.  Feeding quarterly
flows to fixed published coefficients would shrink those terms roughly fourfold.
We therefore build trailing-four-quarter sums, falling back to x4 only where
fewer than four contiguous prior quarters exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Published coefficients.  Verified against the original papers; see
# docs/DECISIONS.md for the citations and the Z'' / X5 discrepancy note.
# --------------------------------------------------------------------------
# Altman (1983) Z' — private firms, five variables, book-value X4.
Z_PRIME_COEF = {"X1": 0.717, "X2": 0.847, "X3": 3.107, "X4": 0.420, "X5": 0.998}
Z_PRIME_CUTS = (1.23, 2.90)          # < 1.23 distress, > 2.90 safe, between = grey

# Altman (1995) Z'' — non-manufacturers / emerging markets, four variables.
Z_DPRIME_COEF = {"X1": 6.56, "X2": 3.26, "X3": 6.72, "X4": 1.05}
Z_DPRIME_CUTS = (1.10, 2.60)

# Ohlson (1980) model 1.
OHLSON_COEF = {
    "const": -1.32, "SIZE": -0.407, "TLTA": 6.03, "WCTA": -1.43, "CLCA": 0.0757,
    "OENEG": -1.72, "NITA": -2.37, "FUTL": -1.83, "INTWO": 0.285, "CHIN": -0.521,
}

# Zmijewski (1984) probit.
ZMIJEWSKI_COEF = {"const": -4.3, "NITA": -4.5, "TLTA": 5.7, "CACL": -0.004}

TTM_COLS = ["Revenue", "NetIncomeLoss", "OCF", "EBIT"]


# --------------------------------------------------------------------------
# Trailing-four-quarter annualisation
# --------------------------------------------------------------------------
def annualise(panel: pd.DataFrame, cols: list[str] = TTM_COLS) -> pd.DataFrame:
    """Add `<col>_ttm` and `<col>_ttm_fallback` for each flow column.

    A TTM value is the sum of quarters t-3..t on a *contiguous* quarter grid.
    Where any of the four is absent the value falls back to 4 x the quarter's
    own flow, and the row is flagged so the fallback rate can be reported.
    """
    p = panel.sort_values(["cik", "quarter_idx"]).reset_index(drop=True).copy()
    q0 = int(p["quarter_idx"].min())
    grid = np.arange(q0, int(p["quarter_idx"].max()) + 1)
    q_pos = p["quarter_idx"].to_numpy(dtype=int) - q0

    for col in cols:
        # Each firm is laid out on a gap-free quarter grid, so a four-quarter
        # sum can never silently splice non-adjacent quarters together.
        wide = p.pivot_table(index="cik", columns="quarter_idx", values=col, aggfunc="first")
        wide = wide.reindex(columns=grid)
        arr = wide.to_numpy(dtype=float)
        f, q = arr.shape
        obs = np.isfinite(arr)
        cs = np.cumsum(np.where(obs, arr, 0.0), axis=1)
        cn = np.cumsum(obs.astype(np.int32), axis=1)
        pad_s = np.concatenate([np.zeros((f, 1)), cs[:, :-4]], axis=1)
        pad_n = np.concatenate([np.zeros((f, 1), dtype=np.int32), cn[:, :-4]], axis=1)
        ttm = np.full((f, q), np.nan)
        ttm[:, 3:] = cs[:, 3:] - pad_s
        complete = np.zeros((f, q), dtype=bool)
        complete[:, 3:] = (cn[:, 3:] - pad_n) == 4
        ttm[~complete] = np.nan

        firm_pos = wide.index.get_indexer(p["cik"])
        p[f"{col}_ttm"] = ttm[firm_pos, q_pos]
        fb = p[f"{col}_ttm"].isna() & p[col].notna()
        p[f"{col}_ttm_fallback"] = fb
        p.loc[fb, f"{col}_ttm"] = 4.0 * p.loc[fb, col]
    return p


def lag4(panel: pd.DataFrame, col: str, name: str) -> pd.Series:
    """The value of `col` four quarters earlier, matched on quarter index."""
    keyed = panel.set_index(["cik", "quarter_idx"])[col]
    idx = pd.MultiIndex.from_arrays([panel["cik"], panel["quarter_idx"] - 4])
    return pd.Series(keyed.reindex(idx).to_numpy(), index=panel.index, name=name)


# --------------------------------------------------------------------------
# The level table every classical model reads
# --------------------------------------------------------------------------
def _safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[ok] = a[ok] / b[ok]
    return out


def load_deflator(path: Path | None = None) -> pd.DataFrame:
    """US GDP implicit price deflator (FRED GDPDEF), quarterly."""
    path = path or (C.REPO / "data" / "gdpdef.csv")
    d = pd.read_csv(path)
    d.columns = ["date", "gdpdef"]
    d["date"] = pd.to_datetime(d["date"])
    d["quarter"] = d["date"].dt.year.astype(str) + "Q" + d["date"].dt.quarter.astype(str)
    return d[["quarter", "gdpdef"]]


@dataclass
class LevelTable:
    df: pd.DataFrame
    notes: dict = field(default_factory=dict)


def build_levels(full: bool = True) -> LevelTable:
    """Firm-quarter levels plus every variable the three formulas need."""
    cols = ["cik", "quarter", "quarter_idx", "Assets", "AssetsCurrent", "Liabilities",
            "LiabilitiesAndStockholdersEquity", "LiabilitiesCurrent", "StockholdersEquity",
            "RetainedEarnings", "Revenue", "NetIncomeLoss", "OCF", "EBIT"]
    p = pd.read_parquet(C.fundamentals_path(full=full), columns=cols)
    p["cik"] = p["cik"].astype(str)
    notes: dict = {"panel_rows": int(len(p))}

    # Total liabilities: prefer the tagged value, else the accounting identity.
    tl = p["Liabilities"].to_numpy(dtype=float)
    ident = p["LiabilitiesAndStockholdersEquity"].fillna(p["Assets"]) - p["StockholdersEquity"]
    fill = ~np.isfinite(tl) & np.isfinite(ident.to_numpy(dtype=float))
    tl = np.where(fill, ident.to_numpy(dtype=float), tl)
    p["TL"] = tl
    notes["TL_from_identity"] = int(fill.sum())

    p = annualise(p)
    for col in TTM_COLS:
        notes[f"{col}_ttm_fallback_rows"] = int(p[f"{col}_ttm_fallback"].sum())

    p["NI_ttm_lag4"] = lag4(p, "NetIncomeLoss_ttm", "NI_ttm_lag4")

    TA, CA, CL = p["Assets"], p["AssetsCurrent"], p["LiabilitiesCurrent"]
    p["WC"] = CA - CL

    # Altman's five, annualised where the formula expects an annual flow.
    p["X1"] = _safe_div(p["WC"], TA)                       # working capital / TA
    p["X2"] = _safe_div(p["RetainedEarnings"], TA)         # retained earnings / TA
    p["X3"] = _safe_div(p["EBIT_ttm"], TA)                 # EBIT (annual) / TA
    p["X4"] = _safe_div(p["StockholdersEquity"], p["TL"])  # book equity / TL  (Z'/Z'' form)
    p["X5"] = _safe_div(p["Revenue_ttm"], TA)              # sales (annual) / TA

    # Ohlson.
    defl = load_deflator()
    p = p.merge(defl, on="quarter", how="left")
    notes["deflator_missing_rows"] = int(p["gdpdef"].isna().sum())
    ta_real = _safe_div(p["Assets"], p["gdpdef"] / 100.0)
    p["SIZE"] = np.where(ta_real > 0, np.log(np.where(ta_real > 0, ta_real, np.nan)), np.nan)
    p["TLTA"] = _safe_div(p["TL"], TA)
    p["WCTA"] = p["X1"]
    p["CLCA"] = _safe_div(CL, CA)
    p["OENEG"] = np.where(np.isfinite(p["TL"]) & np.isfinite(TA), (p["TL"] > TA).astype(float), np.nan)
    p["NITA"] = _safe_div(p["NetIncomeLoss_ttm"], TA)
    p["FUTL"] = _safe_div(p["OCF_ttm"], p["TL"])           # FFO approximated by OCF
    both_neg = (p["NetIncomeLoss_ttm"] < 0) & (p["NI_ttm_lag4"] < 0)
    p["INTWO"] = np.where(p["NetIncomeLoss_ttm"].notna() & p["NI_ttm_lag4"].notna(),
                          both_neg.astype(float), np.nan)
    ni, ni4 = p["NetIncomeLoss_ttm"].to_numpy(float), p["NI_ttm_lag4"].to_numpy(float)
    p["CHIN"] = _safe_div(ni - ni4, np.abs(ni) + np.abs(ni4))

    # Zmijewski.
    p["CACL"] = _safe_div(CA, CL)

    return LevelTable(df=p, notes=notes)


# --------------------------------------------------------------------------
# Joining to the row set, with a train-fitted imputer
# --------------------------------------------------------------------------
CLASSICAL_VARS = ["X1", "X2", "X3", "X4", "X5", "SIZE", "TLTA", "WCTA", "CLCA",
                  "OENEG", "NITA", "FUTL", "INTWO", "CHIN", "CACL"]


def join_to_row_set(levels: pd.DataFrame, row_sets: pd.DataFrame) -> pd.DataFrame:
    """One row per evaluated window, carrying its end-quarter level variables."""
    keyed = levels.set_index(["cik", "quarter_idx"])
    idx = pd.MultiIndex.from_arrays([row_sets["cik"].astype(str), row_sets["end_quarter_idx"]])
    out = row_sets.copy()
    carry = CLASSICAL_VARS + ["Assets", "TL", "Revenue_ttm", "EBIT_ttm",
                              "Revenue_ttm_fallback", "EBIT_ttm_fallback",
                              "NetIncomeLoss_ttm_fallback", "OCF_ttm_fallback"]
    for col in carry:
        out[col] = keyed[col].reindex(idx).to_numpy()
    matched = out["Assets"].notna().mean()
    if matched < 0.99:
        raise AssertionError(f"only {matched:.3%} of windows joined to a panel row — check the join key")
    return out


def fit_imputer(train: pd.DataFrame, cols: list[str] = CLASSICAL_VARS) -> dict[str, float]:
    """Median of each variable on train rows only (leakage rule 1)."""
    return {c: float(np.nanmedian(train[c].to_numpy(dtype=float))) for c in cols}


def apply_imputer(df: pd.DataFrame, med: dict[str, float],
                  cols: list[str] = CLASSICAL_VARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rates = []
    for c in cols:
        v = np.array(out[c].to_numpy(dtype=float), copy=True)
        miss = ~np.isfinite(v)
        rates.append({"variable": c, "n_missing": int(miss.sum()), "rate": float(miss.mean()),
                      "train_median": med[c]})
        v[miss] = med[c]
        out[c] = v
    return out, pd.DataFrame(rates)


# --------------------------------------------------------------------------
# The three formulas
# --------------------------------------------------------------------------
def altman_z_double_prime(df: pd.DataFrame, coef: dict | None = None) -> np.ndarray:
    c = coef or Z_DPRIME_COEF
    return (c["X1"] * df["X1"] + c["X2"] * df["X2"] + c["X3"] * df["X3"] + c["X4"] * df["X4"]).to_numpy(float)


def altman_z_prime(df: pd.DataFrame, coef: dict | None = None) -> np.ndarray:
    c = coef or Z_PRIME_COEF
    return (c["X1"] * df["X1"] + c["X2"] * df["X2"] + c["X3"] * df["X3"]
            + c["X4"] * df["X4"] + c["X5"] * df["X5"]).to_numpy(float)


def ohlson_o(df: pd.DataFrame, coef: dict | None = None) -> np.ndarray:
    c = coef or OHLSON_COEF
    return (c["const"] + c["SIZE"] * df["SIZE"] + c["TLTA"] * df["TLTA"] + c["WCTA"] * df["WCTA"]
            + c["CLCA"] * df["CLCA"] + c["OENEG"] * df["OENEG"] + c["NITA"] * df["NITA"]
            + c["FUTL"] * df["FUTL"] + c["INTWO"] * df["INTWO"] + c["CHIN"] * df["CHIN"]).to_numpy(float)


def zmijewski(df: pd.DataFrame, coef: dict | None = None) -> np.ndarray:
    c = coef or ZMIJEWSKI_COEF
    return (c["const"] + c["NITA"] * df["NITA"] + c["TLTA"] * df["TLTA"]
            + c["CACL"] * df["CACL"]).to_numpy(float)


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def altman_zone(z: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    """'distress' / 'grey' / 'safe' at the variant's canonical cutoffs."""
    lo, hi = cuts
    out = np.full(len(z), "grey", dtype=object)
    out[z < lo] = "distress"
    out[z > hi] = "safe"
    return out


# --------------------------------------------------------------------------
# Model B — Altman's variables, coefficients re-estimated on train
# --------------------------------------------------------------------------
def refit_altman(train: pd.DataFrame, y: np.ndarray, variables: list[str],
                 seed: int = 0) -> tuple[object, dict]:
    """Logistic regression on the Altman variables, fitted on train rows only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=seed, solver="lbfgs")),
    ])
    Xtr = train[variables].to_numpy(dtype=float)
    pipe.fit(Xtr, y)
    lr = pipe.named_steps["lr"]
    scl = pipe.named_steps["scale"]
    # Coefficients expressed on the original variable scale, for the paper.
    raw = lr.coef_[0] / scl.scale_
    intercept = float(lr.intercept_[0] - np.sum(lr.coef_[0] * scl.mean_ / scl.scale_))
    coefs = {"intercept": intercept, **{v: float(c) for v, c in zip(variables, raw)}}
    return pipe, coefs
