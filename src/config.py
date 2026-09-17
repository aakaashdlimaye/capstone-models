"""Paths, constants and the feature vocabulary.

The dataset location is configured here once.  Everything else imports from
this module, so moving the dataset checkout is a one-line change.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent

# The frozen dataset repository.  Override with CAPSTONE_DATASET_DIR.
DATASET_REPO = Path(
    os.environ.get("CAPSTONE_DATASET_DIR", REPO.parent / "capstone-dataset")
).resolve()
DATA = DATASET_REPO / "data"
PROCESSED = DATA / "processed"
INTERIM = DATA / "interim"
EXTERNAL = DATA / "external"

# Pilot smoke runs write to a separate tree so they can never be mistaken for,
# or overwrite, a reported number.  run_all.py sets this before importing src.
RESULTS = Path(os.environ.get("CAPSTONE_RESULTS_DIR", REPO / "results")).resolve()
PREDS = RESULTS / "preds"
ATTN = RESULTS / "attn"
RUNS = RESULTS / "runs"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS_DIR = REPO / "models"
REPORTS = Path(os.environ.get("CAPSTONE_REPORTS_DIR", REPO / "reports")).resolve()
DOCS = REPO / "docs"
EXPERIMENTS = REPO / "experiments"

for _d in (RESULTS, PREDS, ATTN, RUNS, FIGURES, TABLES, MODELS_DIR, REPORTS, DOCS, EXPERIMENTS):
    _d.mkdir(parents=True, exist_ok=True)


def sequences_path(split: str, full: bool = True) -> Path:
    suffix = "_full" if full else ""
    return PROCESSED / f"sequences_{split}{suffix}.npz"


def manifest_path(full: bool = True) -> Path:
    return PROCESSED / ("split_manifest_full.csv" if full else "split_manifest.csv")


def ratios_path(full: bool = True) -> Path:
    return PROCESSED / ("ratios_panel_full.parquet" if full else "ratios_panel.parquet")


def fundamentals_path(full: bool = True) -> Path:
    return INTERIM / ("fundamentals_panel_full.parquet" if full else "fundamentals_panel.parquet")


def scaler_path(full: bool = True) -> Path:
    return PROCESSED / ("scaler_params_full.json" if full else "scaler_params.json")


# --------------------------------------------------------------------------
# Tensor contract
# --------------------------------------------------------------------------
WINDOW_LEN = 8
N_FEATURES = 29
HORIZONS = (1, 2, 3, 4)

FEATURE_NAMES = [
    "r01_current_ratio",
    "r02_quick_ratio",
    "r03_cash_ratio",
    "r04_wc_to_ta",
    "r05_net_profit_margin",
    "r06_roa",
    "r07_roe",
    "r08_ebitda_margin",
    "r09_ebit_to_ta",
    "r10_re_to_ta",
    "r11_debt_to_equity",
    "r12_debt_to_assets",
    "r13_interest_coverage",
    "r14_equity_to_liabilities",
    "r15_ltd_to_ta",
    "r16_asset_turnover",
    "r17_inventory_turnover",
    "r18_receivables_turnover",
    "r19_payables_turnover",
    "r20_cash_conversion_cycle",
    "r21_revenue_growth",
    "r22_net_income_growth",
    "r23_assets_growth",
    "r24_equity_growth",
    "r25_ocf_to_cl",
    "r26_fcf_to_ta",
    "r27_accrual_quality",
    "r28_ocf_to_debt",
    "r29_negative_equity_flag",
]

INDICATOR_NAMES = ["has_inventory", "has_debt"]

# Six ratio families for group-level SHAP; feature 29 stands alone.
FAMILIES: dict[str, list[int]] = {
    "Liquidity": [0, 1, 2, 3],
    "Profitability": [4, 5, 6, 7, 8, 9],
    "Leverage": [10, 11, 12, 13, 14],
    "Efficiency": [15, 16, 17, 18, 19],
    "Growth": [20, 21, 22, 23],
    "Cash flow": [24, 25, 26, 27],
    "Distress flag": [28],
}

# Altman's five, in X1..X5 order, as indices into axis 2 of X.
#   X1 = r04_wc_to_ta, X2 = r10_re_to_ta, X3 = r09_ebit_to_ta,
#   X4 = r14_equity_to_liabilities (book form), X5 = r16_asset_turnover
ALTMAN_IDX = [3, 9, 8, 13, 15]
ALTMAN_NAMES = ["X1_wc_to_ta", "X2_re_to_ta", "X3_ebit_to_ta", "X4_equity_to_liab", "X5_asset_turnover"]

# Expected positives per split per horizon — asserted on load.
EXPECTED_POSITIVES = {
    "train": {1: 154, 2: 410, 3: 693, 4: 986},
    "val": {1: 37, 2: 71, 3: 92, 4: 108},
    "test": {1: 60, 2: 158, 3: 260, 4: 363},
}
EXPECTED_N = {"train": 86592, "val": 20844, "test": 32216}
EXPECTED_POSITIVE_FIRMS = {"train": 450, "val": 152, "test": 127}
EXPECTED_TOTAL_WINDOWS = 139652

# --------------------------------------------------------------------------
# Experiment protocol constants
# --------------------------------------------------------------------------
SEEDS = (0, 1, 2, 3, 4)
ARCHITECTURES = ("lstm", "bilstm", "transformer", "cnn_lstm_attn")
TREATMENTS = ("none", "class_weight", "smote", "focal")
COST_RATIOS = (1, 10, 20, 50)          # Type II : Type I  (missing a bankruptcy : false alarm)
TUNING_TRIALS = 30                      # identical budget for every architecture
MAX_EPOCHS = 60
PATIENCE = 10

# Rolling-origin folds live inside train+val; test is never touched.
CV_FOLDS = [
    ("2016Q4", ("2017Q1", "2017Q4")),
    ("2017Q4", ("2018Q1", "2018Q4")),
    ("2018Q4", ("2019Q1", "2019Q4")),
    ("2019Q4", ("2020Q1", "2021Q4")),
]

TRAIN_END_QUARTER = "2019Q4"
VAL_START_QUARTER = "2020Q1"
TEST_START_QUARTER = "2022Q1"
