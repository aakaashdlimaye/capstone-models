"""The nine acceptance criteria, checked against the artefacts on disk.

These are integration checks, not unit tests: they only pass once a full run has
produced `results/`.  They are skipped when it has not, so `pytest tests/` still
works on a fresh checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

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


# --- 1. run_all.py reproduces everything, seeds fixed ----------------------
def test_run_all_summary_records_the_protocol():
    s = json.loads((C.RESULTS / "run_all_summary.json").read_text(encoding="utf-8"))
    assert s["universe"] == "full"
    assert s["tuning_trials"] == C.TUNING_TRIALS
    assert s["models_sha"] != "unknown" and s["dataset_sha"] != "unknown"


def test_every_run_record_carries_both_shas_and_a_seed():
    records = [json.loads(p.read_text(encoding="utf-8")) for p in C.RUNS.glob("*.json")]
    assert len(records) > 250
    for r in records:
        assert r["models_sha"] != "unknown", r["key"]
        assert r["dataset_sha"] != "unknown", r["key"]
        assert isinstance(r["seed"], int)


def test_every_run_has_an_input_config():
    for p in C.RUNS.glob("*.json"):
        assert (C.EXPERIMENTS / p.name).exists(), f"no experiments/{p.name}"


# --- 2. identical row set, all four horizons ------------------------------
def test_row_set_contract_holds():
    rs = _read("row_sets.csv")
    assert len(rs) == C.EXPECTED_TOTAL_WINDOWS
    assert rs.duplicated(["cik", "end_quarter"]).sum() == 0


def test_all_models_scored_on_the_same_test_rows_at_every_horizon():
    rs = _read("row_sets.csv")
    n_test = int((rs["split"] == "test").sum())
    checked = 0
    for h in C.HORIZONS:
        for model in ("altman_zdp", "altman_zp", "ohlson_o", "zmijewski",
                      "logreg", "svm_rbf", "random_forest", "xgboost", "stacking"):
            p = C.PREDS / f"{model}_{h}.parquet"
            assert p.exists(), f"missing {p.name}"
            assert len(pd.read_parquet(p)) == n_test
            checked += 1
        for arch in C.ARCHITECTURES:
            for seed in C.SEEDS:
                p = C.PREDS / f"{arch}_{h}_class_weight_{seed}.parquet"
                assert p.exists(), f"missing {p.name}"
                assert len(pd.read_parquet(p)) == n_test
                checked += 1
    assert checked == 4 * (9 + 4 * len(C.SEEDS))


# --- 3. 5-seed mean +/- std, DeLong, McNemar ------------------------------
def test_deep_results_are_five_seed_summaries():
    d = _read("deep_all.csv")
    assert (d["n_seeds"] == len(C.SEEDS)).all()
    assert d["pr_auc_std"].notna().all() and (d["pr_auc_std"] > 0).all()
    assert len(d) == len(C.ARCHITECTURES) * len(C.HORIZONS)


def test_every_decomposition_step_has_delong_and_mcnemar():
    d = _read("decomposition_all.csv")
    steps = d[d["step"].notna()]
    assert len(steps) > 0
    # DeLong and McNemar are still reported, now explicitly labelled window-level
    # because both assume independent rows; the primary intervals are the
    # firm-clustered ones checked in tests/test_acceptance_review.py.
    for col in ("delong_window_level_diff", "delong_window_level_ci_low",
                "delong_window_level_ci_high", "delong_window_level_p",
                "pr_auc_diff", "pr_auc_cluster_ci_low", "pr_auc_cluster_ci_high",
                "mcnemar_window_level_b", "mcnemar_window_level_c",
                "mcnemar_window_level_p"):
        assert col in steps.columns, f"{col} missing"
        assert steps[col].notna().all(), f"{col} has gaps"
    assert (steps["delong_window_level_ci_low"]
            <= steps["delong_window_level_diff"] + 1e-9).all()
    assert (steps["delong_window_level_diff"]
            <= steps["delong_window_level_ci_high"] + 1e-9).all()


# --- 4. ablation and protocol audit, with the >95% row --------------------
def test_imbalance_ablation_is_a_full_factorial():
    for h in (1, 4):
        t = _read(f"imbalance_ablation_h{h}.csv")
        assert set(t["arch"]) == set(C.ARCHITECTURES)
        assert set(C.TREATMENTS) <= set(t["treatment"])
        assert {f"cost_threshold_1to{r}" for r in C.COST_RATIOS} <= set(t["treatment"])
        assert (t["n_seeds"] == len(C.SEEDS)).all()
        for r in C.COST_RATIOS:
            assert f"cost_1to{r}_mean" in t.columns


def test_protocol_audit_reproduces_the_inflated_accuracy():
    a = _read("protocol_audit.csv")
    inflated = a[a["protocol"] == "inflated"]
    assert len(inflated) == 2
    assert (inflated["accuracy_mean"] > 0.95).all(), "the >95% claim is not reproduced"
    correct = a[a["protocol"] == "correct"]
    assert correct["pr_auc_mean"].notna().all(), "PR-AUC must be reported beside it"
    assert (correct["pr_auc_mean"] < inflated["accuracy_mean"].min()).all()


def test_protocol_audit_decomposition_is_bounded():
    d = _read("protocol_audit_decomposition.csv")
    for col in ("share_chronological_split", "share_resampling_inside_train"):
        assert ((d[col] >= 0) & (d[col] <= 1)).all(), f"{col} outside [0, 1]"
    assert np.allclose(d["share_chronological_split"] + d["share_resampling_inside_train"], 1.0)


# --- 5. the decomposition states its share sentence -----------------------
def test_decomposition_summary_states_the_shares_with_numbers():
    from src.phase_e import CAUSE, PRIMARY_LADDER

    txt = (C.RESULTS / "decomposition_summary.md").read_text(encoding="utf-8")
    # The old ladder called B->C "the static formulation" and credited four
    # simultaneous changes to it; it is now split into preprocessing,
    # nonlinearity and the time axis, each its own rung.
    for cause in ("coefficient drift", "nonlinearity", "the time axis",
                  "the feature-set expansion"):
        assert cause in txt, f"summary does not name {cause!r}"
    for h in C.HORIZONS:
        assert f"At h={h}, test PR-AUC moves from" in txt

    s = _read("decomposition_shares.csv")
    steps = [f"{PRIMARY_LADDER[i - 1]}->{PRIMARY_LADDER[i]}"
             for i in range(1, len(PRIMARY_LADDER))]
    assert set(steps) <= set(CAUSE), "a ladder step has no stated cause"
    for h in C.HORIZONS:
        row = s[s["horizon"] == h].iloc[0]
        shares = [row[f"pr_auc_share_{k}"] for k in steps]
        assert all(0 <= v <= 1 for v in shares), f"h={h} shares outside [0, 1]: {shares}"
        assert np.isclose(sum(shares), 1.0), f"h={h} shares do not sum to 1"


# --- 6. attention and SHAP, masked, with agreement quantified -------------
def test_attention_and_shap_exist_and_agree_is_quantified():
    a = _read("attention_by_quarter.csv")
    assert set(a["architecture"]) == {"transformer", "cnn_lstm_attn"}
    assert (a["n_windows"] > 0).all()
    assert "frac_step_observed" in a.columns, "masking evidence missing"

    x = _read("interpretability_crosscheck.csv")
    assert len(x) > 0
    for col in ("spearman_rho", "spearman_p", "kendall_tau", "agree_on_top_quarter"):
        assert col in x.columns
    assert x["spearman_rho"].notna().all()

    f = _read("shap_feature_importance.csv")
    assert set(f["feature"]) == set(C.FEATURE_NAMES)
    assert f["is_altman"].sum() == len(C.ALTMAN_IDX) * f["horizon"].nunique()

    for stem in ("attention_by_quarter", "shap_family_by_quarter"):
        for ext in (".png", ".svg"):
            assert (C.FIGURES / f"{stem}{ext}").exists(), f"{stem}{ext} missing"


# --- 7. the leakage audit passes, with evidence ---------------------------
def test_leakage_audit_passes_all_four_checks():
    txt = (C.REPORTS / "leakage_audit.md").read_text(encoding="utf-8")
    assert "**Leakage audit: PASS**" in txt
    assert txt.count("**PASS**") >= 4
    assert "FAIL" not in txt.replace("PASS or FAIL", "")
    # Each check must print evidence, not just a verdict.
    assert txt.count("```") >= 8
    assert "prediction files checked" in txt
    assert "runs whose statistics do not come from train only : 0" in txt


# --- 8. RESULTS.md is generated and complete ------------------------------
def test_results_md_is_complete_and_traceable():
    txt = (C.REPORTS / "RESULTS.md").read_text(encoding="utf-8")
    assert "not produced by this run" not in txt
    assert "Missing artefacts" not in txt
    for token in ("TODO", "TBD", "XXX", "PLACEHOLDER", "FIXME"):
        assert token not in txt
    for n in range(1, 10):
        assert f"\n## {n}. " in txt, f"section {n} missing"
    man = json.loads((C.RESULTS / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert man["models_sha"] != "unknown" and man["dataset_sha"] != "unknown"
    assert man["n_files"] == len(man["files"]) > 100


# --- 9. unit tests cover the five named areas -----------------------------
def test_the_named_unit_test_areas_exist():
    here = Path(__file__).parent
    expected = {
        "test_data.py": ["test_shapes_and_counts", "test_positive_counts_match_the_report",
                         "test_loader_rejects_a_scrambled_label_column"],
        "test_classical.py": ["test_ttm_sums_four_contiguous_quarters",
                              "test_z_double_prime_matches_hand_calculation",
                              "test_z_prime_matches_hand_calculation"],
        "test_leakage_rules.py": ["test_smote_touches_only_the_training_arrays",
                                  "test_threshold_comes_from_val_and_ignores_test"],
    }
    for fname, names in expected.items():
        txt = (here / fname).read_text(encoding="utf-8")
        for n in names:
            assert f"def {n}(" in txt, f"{fname} is missing {n}"
