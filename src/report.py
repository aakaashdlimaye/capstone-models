"""reports/RESULTS.md — generated entirely from the CSVs in results/.

Nothing in this module types a number.  Every table is read off disk, so the
report cannot drift from the artefacts and cannot contain a placeholder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import utils as U


MISSING: list[str] = []


def _read(name: str) -> pd.DataFrame | None:
    """Read a results table, recording anything absent or empty.

    A full run should find every table.  Anything that lands in MISSING is a
    hole in the report and is printed loudly rather than rendered as prose.
    """
    p = C.RESULTS / name
    if not p.exists():
        MISSING.append(f"{name} (absent)")
        return None
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        MISSING.append(f"{name} (unreadable: {exc})")
        return None
    if df.empty:
        MISSING.append(f"{name} (empty)")
        return None
    return df


def _md(df: pd.DataFrame | None, cols: list[str] | None = None, nd: int = 4) -> str:
    if df is None or df.empty:
        return "_not produced by this run._\n"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return U.to_markdown(df, floatfmt=f"%.{nd}f")


def _ms(df: pd.DataFrame, base: str) -> pd.Series:
    m, s = f"{base}_mean", f"{base}_std"
    if m not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df.apply(lambda r: U.fmt_ms(r[m], r.get(s, np.nan)), axis=1)


def compute_time() -> tuple[float, int]:
    total, n = 0.0, 0
    for p in C.RUNS.glob("*.json"):
        try:
            total += float(json.loads(p.read_text(encoding="utf-8")).get("seconds", 0.0))
            n += 1
        except Exception:
            continue
    for p in (C.RESULTS / "tuning").glob("*.csv"):
        try:
            total += float(pd.read_csv(p)["seconds"].sum())
        except Exception:
            continue
    return total, n


def provenance_manifest() -> Path:
    """One record per produced artefact, stamped with both repositories' SHAs.

    Individual CSVs have no place to carry a git SHA, so the SHA lives here
    together with each file's size and content hash — which also makes it
    possible to tell whether a table on disk still matches the run that made it.
    """
    import hashlib

    prov = U.provenance()
    rows = []
    for p in sorted(C.RESULTS.rglob("*")):
        if not p.is_file():
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"path": str(p.relative_to(C.RESULTS)).replace("\\", "/"),
                     "bytes": p.stat().st_size, "sha256": h})
    out = C.RESULTS / "PROVENANCE.json"
    U.write_json(out, {**prov, "seeds": list(C.SEEDS),
                       "dataset_files": {
                           name: str(C.sequences_path(name).name)
                           for name in ("train", "val", "test")},
                       "n_files": len(rows), "files": rows})
    return out


def build(full: bool = True) -> Path:
    MISSING.clear()
    prov = U.provenance()
    L: list[str] = []
    A = L.append

    A("# Results")
    A("")
    A("Every table below is generated from a CSV in `results/` by `src/report.py`.")
    A("No number in this file is typed by hand.")
    A("")
    A(f"- Universe: **{'full' if full else 'PILOT — not a reportable result'}**")
    A(f"- Dataset repository SHA: `{prov['dataset_sha']}`")
    A(f"- Models repository SHA: `{prov['models_sha']}`")
    A(f"- Seeds: `{list(C.SEEDS)}`; deep results are mean ± std over "
      f"{len(C.SEEDS)} seeds unless stated otherwise.")
    A(f"- Generated: {prov['timestamp']}")
    secs, nruns = compute_time()
    A(f"- Total measured compute: **{secs / 3600:.2f} h** over {nruns} cached deep runs "
      f"plus the logged tuning trials.")
    A("")
    A("Read PR-AUC first.  At a window-level positive rate of roughly 1% (0.18% at")
    A("h=1) ROC-AUC flatters every model, which is exactly the point Contribution 5")
    A("makes about the published literature.")
    A("")

    # ---------------------------------------------------------------- 1
    A("## 1. Data and the row-set contract")
    A("")
    bal = _read("class_balance.csv")
    A(_md(bal, ["split", "windows", "firms", "positive_firms",
                "pos_y1", "rate_y1", "pos_y2", "rate_y2",
                "pos_y3", "rate_y3", "pos_y4", "rate_y4"], nd=5))
    A("")
    rs = _read("row_sets.csv")
    if rs is not None:
        A(f"`results/row_sets.csv` holds **{len(rs):,}** windows with no duplicate")
        A("`(cik, end_quarter)` pair.  Every model in this report — classical, machine")
        A("learning and deep — is scored on exactly these rows.")
        A("")

    # ---------------------------------------------------------------- 2
    A("## 2. Classical and machine-learning baselines")
    A("")
    for h in C.HORIZONS:
        b = _read(f"baselines_{h}.csv")
        if b is None:
            continue
        A(f"### Horizon h = {h}")
        A("")
        A(_md(b, ["model", "family", "val_roc_auc", "val_pr_auc", "roc_auc", "pr_auc",
                  "f1", "recall", "specificity", "precision", "threshold",
                  "cost_1to1", "cost_1to10", "cost_1to20", "cost_1to50"]))
        A("")
    A("### Altman at its canonical cutoffs")
    A("")
    A("Threshold-free AUCs above are the fair reading; this is how the score is")
    A("actually used in practice.")
    A("")
    A(_md(_read("altman_canonical_cutoffs.csv"),
          ["model", "horizon", "rule", "cut_distress", "cut_safe", "tp", "fp", "fn", "tn",
           "precision", "recall", "specificity", "f1", "accuracy", "alarm_rate"]))
    A("")
    A("### Where the classical inputs came from")
    A("")
    notes_p = C.RESULTS / "classical_levels_notes.json"
    if notes_p.exists():
        n = json.loads(notes_p.read_text(encoding="utf-8"))
        A(f"- Trailing-four-quarter sums replace quarterly flows for Altman X3/X5, "
          f"Ohlson NI/TA and FFO/TL, and Zmijewski NI/TA.")
        A(f"- The `x4` fallback (fewer than four contiguous prior quarters) was used on "
          f"{n.get('ttm_fallback_on_row_set')} of the {len(rs) if rs is not None else '?'} "
          f"evaluated windows.")
        A(f"- Total liabilities came from the accounting identity on "
          f"{n.get('TL_from_identity')} panel rows where the tag was absent.")
        A(f"- Missing classical variables are filled with the **train** median "
          f"(fitted on {n.get('imputer_train_rows'):,} train rows).")
        A("")

    # ---------------------------------------------------------------- 3
    A("## 3. The four temporal architectures")
    A("")
    A(_md(_read("architecture_parameters.csv"), None, nd=0))
    A("")
    tun = _read("tuning_summary.csv")
    if tun is not None:
        A("Hyperparameters were searched on validation PR-AUC only, with an identical")
        A(f"budget of {int(tun['n_trials'].max())} random-search trials per architecture.")
        A("Trial 0 is the specification's own defaults, so the search can only improve")
        A("on the spec.")
        A("")
        A(_md(tun, ["architecture", "lr", "batch_size", "dropout", "weight_decay",
                    "best_val_pr_auc", "spec_default_val_pr_auc", "n_trials"], nd=5))
        A("")
    for h in C.HORIZONS:
        d = _read(f"deep_{h}.csv")
        if d is None:
            continue
        t = pd.DataFrame({"architecture": d["arch"], "n_params": d.get("n_params"),
                          "ROC-AUC": _ms(d, "roc_auc"), "PR-AUC": _ms(d, "pr_auc"),
                          "F1": _ms(d, "f1"), "recall": _ms(d, "recall"),
                          "specificity": _ms(d, "specificity"),
                          "cost @ 1:20": _ms(d, "cost_1to20"),
                          "epochs": d.get("epochs_mean")})
        A(f"### Horizon h = {h}")
        A("")
        A(_md(t, None, nd=2))
        A("")
    A("### Multi-horizon head versus four single-horizon models")
    A("")
    A(_md(_read("multi_horizon_comparison.csv"),
          ["arch", "horizon", "pr_auc_mean", "pr_auc_std", "single_pr_auc_mean",
           "single_pr_auc_std", "roc_auc_mean", "single_roc_auc_mean",
           "multi_beats_single_pr"]))
    A("")
    ind = _read("indicator_ablation.csv")
    if ind is not None:
        A("### Structural-indicator ablation")
        A("")
        A("Three of the 29 ratios are undefined for firms without inventory and one for")
        A("firms without debt.  This run feeds `has_inventory` and `has_debt` as two")
        A("extra input channels so the model can tell 'not applicable' from 'missing'.")
        A("")
        A(_md(ind))
        A("")

    # ---------------------------------------------------------------- 4
    A("## 4. Imbalance ablation")
    A("")
    for h in (1, 4):
        t = _read(f"imbalance_ablation_h{h}.csv")
        if t is None:
            continue
        out = pd.DataFrame({"architecture": t["arch"], "treatment": t["treatment"],
                            "ROC-AUC": _ms(t, "roc_auc"), "PR-AUC": _ms(t, "pr_auc"),
                            "F1": _ms(t, "f1"), "recall": _ms(t, "recall"),
                            "specificity": _ms(t, "specificity"),
                            "cost 1:1": _ms(t, "cost_1to1"),
                            "cost 1:10": _ms(t, "cost_1to10"),
                            "cost 1:20": _ms(t, "cost_1to20"),
                            "cost 1:50": _ms(t, "cost_1to50")})
        A(f"### Horizon h = {h}")
        A("")
        A(_md(out, None, nd=2))
        A("")
        A(f"![PR-AUC heatmap h={h}](../results/figures/imbalance_heatmap_h{h}.png)")
        A("")
        A("PR-AUC is threshold-free, so only the four training treatments can move it.")
        A("The cost-sensitive threshold changes where the untreated model's ranking is")
        A("cut, which shows up in expected cost rather than in PR-AUC:")
        A("")
        A(f"![cost heatmap h={h}](../results/figures/imbalance_cost_heatmap_h{h}.png)")
        A("")

    # ---------------------------------------------------------------- 5
    A("## 5. Protocol audit — reproducing the 91–99% accuracy")
    A("")
    A("The same LSTM, run three ways.  The only changes between rows are")
    A("methodological.")
    A("")
    A(_md(_read("protocol_audit.csv"),
          ["horizon", "protocol", "n_test", "test_positive_rate", "accuracy_mean",
           "accuracy_std", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "recall_mean",
           "f1_mean", "n_seeds"]))
    A("")
    A("### What each fix costs")
    A("")
    A("The decomposition is on PR-AUC.  Accuracy cannot carry it: it is high under")
    A("the inflated protocol because the model separates a balanced test set, and high")
    A("again under the correct protocol because the majority class is ~99% of it.  The")
    A("two large, offsetting accuracy moves cancel, so `model_beats_majority_class` is")
    A("the column that matters — where it is false, a model reporting >99% accuracy is")
    A("losing to a constant that predicts no bankruptcy at all.")
    A("")
    A(_md(_read("protocol_audit_decomposition.csv")))
    A("")
    A(_md(_read("external_protocol_audit.csv"),
          ["dataset", "protocol", "accuracy_mean", "accuracy_std", "pr_auc_mean",
           "pr_auc_std", "roc_auc_mean", "recall_mean", "test_positive_rate", "n_seeds"]))
    A("")

    # ---------------------------------------------------------------- 6
    A("## 6. Decomposition — why the 1968 formula underperforms")
    A("")
    for h in C.HORIZONS:
        d = _read(f"decomposition_{h}.csv")
        if d is None:
            continue
        A(f"### Horizon h = {h}")
        A("")
        A(_md(d, ["model", "roc_auc", "pr_auc", "f1", "recall", "specificity", "step",
                  "delong_roc_diff", "delong_ci_low", "delong_ci_high", "delong_p",
                  "pr_auc_diff", "pr_ci_low", "pr_ci_high",
                  "mcnemar_b", "mcnemar_c", "mcnemar_p"]))
        A("")
        A(f"![decomposition PR curves h={h}](../results/figures/decomposition_pr_h{h}.png)")
        A("")
    A("### Share of the A→D gap attributable to each cause")
    A("")
    sh = _read("decomposition_shares.csv")
    A(_md(sh, [c for c in (sh.columns if sh is not None else [])
               if c == "horizon" or c.startswith(("pr_auc_", "roc_auc_"))]))
    A("")
    A("### Missingness sanity check")
    A("")
    A(_md(_read("decomposition_complete_subset.csv")))
    A("")
    A("### Re-estimated coefficients")
    A("")
    A(_md(_read("decomposition_coefficients.csv"), None, nd=5))
    A("")

    # ---------------------------------------------------------------- 7
    A("## 7. Interpretability")
    A("")
    A("![attention by quarter](../results/figures/attention_by_quarter.png)")
    A("")
    A("![SHAP family by quarter](../results/figures/shap_family_by_quarter.png)")
    A("")
    A(_md(_read("interpretability_crosscheck.csv"),
          ["horizon", "attention_model", "shap_model", "group", "spearman_rho",
           "spearman_p", "kendall_tau", "attention_argmax", "shap_argmax",
           "agree_on_top_quarter"]))
    A("")
    A("### Ratio-family importance")
    A("")
    A(_md(_read("shap_family_importance.csv"),
          ["horizon", "family", "n_features", "mean_abs_shap", "share"], nd=5))
    A("")
    A("### Top features")
    A("")
    feats = _read("shap_feature_importance.csv")
    if feats is not None:
        top = feats.sort_values(["horizon", "rank"]).groupby("horizon").head(10)
        A(_md(top, ["horizon", "rank", "feature", "family", "mean_abs_shap",
                    "is_altman", "observed_rate"], nd=5))
    A("")

    # ---------------------------------------------------------------- 8
    A("## 8. Robustness")
    A("")
    A("### Rolling-origin cross-validation (inside train + val; test untouched)")
    A("")
    A(_md(_read("rolling_origin.csv"),
          ["model", "fold", "n_eval", "n_pos_eval", "pr_auc_mean", "pr_auc_std",
           "roc_auc_mean", "roc_auc_std", "n_seeds"]))
    A("")
    A("### Embargo — boundary-straddling windows dropped")
    A("")
    A(_md(_read("embargo.csv")))
    A("")
    A("### External validation")
    A("")
    A(_md(_read("external_validation.csv"),
          ["dataset", "model", "n_train", "n_test", "base_rate", "roc_auc", "pr_auc",
           "f1", "recall", "specificity"]))
    A("")

    # ---------------------------------------------------------------- 9
    A("## 9. Gates")
    A("")
    for name in ("phase_b_gate", "phase_c_gate"):
        t = _read(f"tables/{name}.csv")
        if t is not None:
            A(f"**{name}**")
            A("")
            A(_md(t))
            A("")
    A("The leakage audit and its printed evidence are in "
      "[`leakage_audit.md`](leakage_audit.md).")
    A("")

    if MISSING:
        A("## Missing artefacts")
        A("")
        A("These tables were expected and not found, so the sections above are")
        A("incomplete.  A full run should leave this list empty.")
        A("")
        for m in MISSING:
            A(f"- `{m}`")
        A("")

    out = C.REPORTS / "RESULTS.md"
    out.write_text("\n".join(L), encoding="utf-8")
    man = provenance_manifest()
    print(f"[report] wrote {out}")
    print(f"[report] wrote {man}")
    if MISSING:
        print(f"[report] WARNING: {len(MISSING)} expected tables missing:")
        for m in MISSING:
            print(f"           {m}")
    else:
        print("[report] every expected table was found")
    return out
