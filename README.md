# capstone-models

Modelling and evaluation for *"Bankruptcy Prediction using Temporal Deep
Learning: A Comparative Study of LSTM and Transformer Models."*

This repository consumes the frozen dataset in
[`capstone-dataset`](https://github.com/aakaashdlimaye/bankruptcy-prediction-dataset)
and produces every table and figure the paper needs: classical baselines, four
temporal architectures, the imbalance ablation, the protocol audit, the
decomposition experiment, interpretability, and the robustness checks.

Nothing here writes to the dataset repository. It is read-only input.

---

## 1. What this answers

| Contribution | Where it lands |
|---|---|
| 2 — first head-to-head comparison of four temporal architectures under one dataset, split, imbalance treatment and tuning budget | `results/deep_{1,2,3,4}.csv`, `results/architecture_parameters.csv` |
| 3 — decomposition of classical-formula failure into coefficient drift, static formulation and feature-set limitation | `results/decomposition_{h}.csv`, `results/decomposition_summary.md` |
| 4 — period-level interpretability, validated by two independent methods | `results/figures/attention_by_quarter.*`, `results/figures/shap_family_by_quarter.*`, `results/interpretability_summary.md` |
| 5 — methodological audit of the reported 91–99% accuracies | `results/protocol_audit.csv`, `results/imbalance_ablation_h{1,4}.csv`, `results/external_protocol_audit.csv` |

Contribution 1 (the dataset itself) is delivered by the other repository.

---

## 2. Setup

Python 3.11 or newer. A CUDA GPU is used if present and is not required — the
models are small (input 8 × 29) and the whole matrix runs on CPU, just slower.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

For GPU training, install torch from the CUDA index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

The dataset is expected at `../capstone-dataset/`. Override with:

```bash
export CAPSTONE_DATASET_DIR=/path/to/capstone-dataset
```

Two of the dataset's files are Git LFS objects. If you cloned rather than using
a local checkout, run `git lfs pull` in that repository first — otherwise
`ratios_panel_full.parquet` and `fundamentals_panel_full.parquet` arrive as
130-byte pointers and the loader will say so.

---

## 3. Reproducing everything

```bash
python run_all.py
```

That rebuilds every file in `results/` and `reports/` from the frozen dataset
with seeds fixed. Runs are cached by key, so an interrupted run resumes where it
stopped and a second invocation is close to instant.

Useful variants:

```bash
python run_all.py --pilot          # smoke run on the 18k-window pilot tensors
python run_all.py --pilot --quick  # wiring check only, a few minutes
python run_all.py --from d         # resume from a phase
python run_all.py --only f report  # rebuild interpretability and the report
python run_all.py --force          # ignore cached runs and retrain
python -m pytest tests/ -q         # 56 unit tests
```

`--pilot` writes to `results_pilot/` and `reports_pilot/`. The pilot is
deliberately positive-enriched and **is not a reportable number** — it exists to
catch shape and label bugs before a full run.

---

## 4. Phases

| Phase | What it does | Gate |
|---|---|---|
| **A** | Loads the tensors, asserts the contract, builds the row-set table | 139,652 rows, no duplicates, positives match the dataset report exactly |
| **B** | Altman Z″/Z′, Ohlson, Zmijewski from levels; logistic regression, SVM, random forest, XGBoost and a stacking ensemble on the flattened window | Altman Z″ test ROC-AUC above 0.5 and below the best tuned ML baseline, at every horizon |
| **C** | The four temporal architectures, 4 horizons × 5 seeds, plus the multi-horizon head and the structural-indicator ablation | Every architecture's mean test PR-AUC at h=4 exceeds Altman Z″'s |
| **D** | Imbalance ablation (4 architectures × 5 treatments × 2 horizons × 5 seeds) and the three-protocol audit | The "Inflated" protocol reproduces >95% accuracy |
| **E** | The A→B→C→D decomposition with DeLong CIs and McNemar tests, plus the all-complete-subset check | — |
| **F** | Attention over quarters and SHAP family × quarter importance, masked, cross-checked | — |
| **G** | Rolling-origin CV, the embargo split, external validation on UCI Taiwanese and Polish | — |
| **audit** | The four-check leakage audit with printed evidence | All four checks pass |
| **report** | Generates `reports/RESULTS.md` from the CSVs | — |

---

## 5. Layout

```
src/
  config.py        paths, the 29 feature names, families, Altman indices, protocol constants
  data.py          the loader, its assertions, the row-set contract, the embargo mask
  utils.py         seeding, git SHAs, table and figure writers
  metrics.py       ROC/PR-AUC, thresholded metrics, expected cost, DeLong, McNemar, bootstrap
  classical.py     annualisation, Altman Z''/Z', Ohlson, Zmijewski, coefficient re-estimation
  ml_baselines.py  the five ML baselines and their random search
  models.py        the four architectures in PyTorch, with attention exposed
  imbalance.py     class weights, SMOTE, focal loss
  train.py         the single training loop
  experiment.py    run-and-cache, scoring, seed aggregation
  tuning.py        the deep random search
  phase_{a..g}.py  one module per phase
  leakage.py       the four-check audit
  report.py        RESULTS.md generation
experiments/       one JSON config per phase, written by the run
results/           tables (CSV + Markdown), figures (PNG + SVG), predictions, run records
models/            saved weights (gitignored)
reports/           RESULTS.md, leakage_audit.md
docs/DECISIONS.md  every judgment call, with its rationale
tests/             56 unit tests
```

---

## 6. What every results file contains

### Tables

| File | Contents |
|---|---|
| `row_sets.csv` | One row per evaluated window: `(cik, end_quarter)`, its split, its tensor row, its panel join key, labels at all four horizons. **Every model is scored on exactly these rows.** |
| `class_balance.csv` | Windows, firms, positive firms, positives and rates per split per horizon |
| `baselines_{1,2,3,4}.csv` | All nine baselines at one horizon: val and test ROC/PR-AUC, F1, recall, specificity, precision, the val-chosen threshold, and expected cost at four cost ratios |
| `baselines_all.csv` | The same, all horizons in one file |
| `altman_canonical_cutoffs.csv` | Z″ and Z′ confusion matrices at 1.10/2.60 and 1.23/2.90, under two alarm rules |
| `classical_levels.parquet`, `classical_levels_notes.json` | The joined level table every classical model reads, and its provenance: TTM fallback counts, identity-derived liabilities, train medians |
| `architecture_parameters.csv` | Trainable parameter count per architecture |
| `tuning_summary.csv`, `tuning/*.json`, `tuning/*.csv` | The winning hyperparameters, the spec default's score, and every trial |
| `deep_{1,2,3,4}.csv`, `deep_all.csv` | Mean ± std over 5 seeds per architecture per horizon |
| `multi_horizon_comparison.csv` | The 4-output head against four single-horizon models |
| `indicator_ablation.csv` | Does feeding `has_inventory` / `has_debt` as extra channels help? |
| `imbalance_ablation_h{1,4}.csv`, `imbalance_ablation_all.csv` | 4 architectures × 8 treatment rows × mean ± std, including expected cost at every ratio |
| `protocol_audit.csv`, `protocol_audit_runs.csv` | The three protocols side by side, aggregated and per seed |
| `protocol_audit_decomposition.csv` | How much of the accuracy drop each fix accounts for |
| `decomposition_{1,2,3,4}.csv`, `decomposition_all.csv` | Each ladder rung with DeLong CIs on the ROC-AUC difference, a bootstrap CI on the PR-AUC difference, and McNemar on the thresholded decisions |
| `decomposition_shares.csv`, `decomposition_summary.md` | The share of the A→D gap attributable to each cause |
| `decomposition_coefficients.csv` | Re-estimated Altman coefficients, on the original variable scale |
| `decomposition_complete_subset.csv` | C and D retrained on windows where all five Altman ratios are observed in all eight quarters |
| `attention_by_quarter.csv` | Mean attention per quarter, per architecture, per group, with the observed fraction it was masked against |
| `shap_family_importance.csv`, `shap_feature_importance.csv`, `shap_family_by_quarter_h{1,4}.csv` | \|SHAP\| by family, by feature, and on the family × quarter grid |
| `interpretability_crosscheck.csv`, `interpretability_summary.md` | Spearman and Kendall agreement between attention and SHAP |
| `rolling_origin.csv`, `rolling_origin_runs.csv` | PR-AUC per expanding fold for the best deep model and Altman Z″ |
| `embargo.csv` | The stricter, boundary-free split against the standard one |
| `external_validation.csv`, `external_protocol_audit.csv` | UCI Taiwanese and Polish under the same protocol, and the protocol audit repeated there |
| `run_all_summary.json` | Phase outcomes, wall time, seeds, tuning budget, both repositories' SHAs |

Every table is written as both `.csv` and `.md`.

### Predictions and run records

| Path | Contents |
|---|---|
| `results/preds/{key}.parquet` | Per-window **test** predictions: `cik`, `end_quarter`, `y_true`, `p_hat` |
| `results/preds_val/{key}.parquet` | The same on validation — this is the only place a threshold is ever chosen from |
| `results/runs/{key}.json` | Metrics, hyperparameters, imbalance statistics, epoch history, both repositories' SHAs, seed |
| `results/attn/{key}.npz` | Attention weights: `(layers, n, heads, 8, 8)` for the Transformer, `(n, 4)` for CNN-LSTM-Attention |

A key is `{tag_}{arch}_{horizon}_{treatment}_{seed}`, e.g. `transformer_4_class_weight_2`
or `decompC_lstm_1_class_weight_0`.

### Figures

`results/figures/` — `attention_by_quarter`, `shap_family_by_quarter`,
`shap_family_by_quarter_h{1,4}`, `decomposition_pr_h{1,2,3,4}`,
`imbalance_heatmap_h{1,4}`. Each as both `.png` and `.svg`.

---

## 7. Protocol, in one place

- **The split is inherited, never recomputed.** Train ≤ 2019Q4, validation
  2020Q1–2021Q4, test 2022Q1–2024Q4, assigned by the window's end quarter.
- **The scaler is never refitted.** `X` arrives z-scored on train-only
  parameters and is used as is.
- **Resampling and weighting happen inside train, after the split.** Val and
  test are never resampled.
- **Thresholds are chosen on validation and applied once to test.**
- **Rolling-origin CV lives inside train + val.** Test is touched once, at the
  end, by each model, exactly once.
- **Hyperparameter search uses validation only**, with an identical 30-trial
  budget for every architecture.
- **Five seeds per configuration**, reported as mean ± std.
- **Every result file records both repositories' git SHA and the seed.**

`reports/leakage_audit.md` proves four of these with printed code output.

---

## 8. Reading the numbers

The window-level positive rate is **0.18% at h=1 and 1.1% at h=4** — not the
~7% firm-level rate — because bankrupt firms stop filing and so contribute far
fewer windows each. Two consequences:

- **PR-AUC is the honest headline.** ROC-AUC flatters every model here, and
  reporting it alone is part of what Contribution 5 criticises. Both are always
  reported.
- **Validation has 37 positives at h=1.** Any threshold tuned there is
  unstable, which is why the threshold sensitivity is reported alongside and
  why early stopping watches PR-AUC rather than the ROC-AUC the original
  specification named (`docs/DECISIONS.md` 0.5).

---

## 9. Compute

Measured on one NVIDIA T1000 8GB, Windows 11, PyTorch 2.14 + CUDA 12.6.
`results/run_all_summary.json` records the wall time of the run that produced
the current artefacts, and `reports/RESULTS.md` prints the total measured
compute across all cached runs and tuning trials.

The models are small — 31k to 71k parameters — and the binding cost is kernel
launch overhead across roughly 2,700 optimiser steps per epoch at batch 32, not
arithmetic. Larger batch sizes are in the search space for exactly that reason.
