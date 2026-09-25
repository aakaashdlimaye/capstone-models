# capstone-models

Modelling and evaluation for

> **Beyond the Z-Score: Decomposing the Failure of Static Bankruptcy Formulas
> with Sequence Models**

The capstone was registered as *"Bankruptcy Prediction using Temporal Deep
Learning: A Comparative Study of LSTM and Transformer Models"*, and that remains
its title for the university.

The paper title changed because a gradient-boosted tree on the flattened window
has the highest PR-AUC of any model here, so the contribution is the
decomposition and the methodological audit rather than a win for sequence
models.

**That framing has an open question against it.** Phase H1b found the
eight-quarter window's contribution is *masked* rather than absent: four of the
29 ratios are year-on-year growth, so a single end-quarter row already carries
four quarters of history. Drop them and the window becomes significant at three
of the four horizons, where it was significant at none with them present; and on
Altman's five ratios, which contain no growth ratio, the window helps
significantly at every horizon. "Temporal structure does not help here" is
therefore too strong a reading of the headline table. See §9 and
`results/h1b_growth_ratio_explanation.csv`.

This repository consumes the frozen dataset in
[`capstone-dataset`](https://github.com/aakaashdlimaye/bankruptcy-prediction-dataset)
and produces every table and figure the paper needs: classical baselines, four
temporal architectures, the imbalance ablation, the protocol audit, the
decomposition experiment, interpretability, the robustness checks, and the
controls and practitioner metrics in Phase H.

Nothing here writes to the dataset repository. It is read-only input.

---

## 1. What this answers

| Contribution | Where it lands |
|---|---|
| 2 — first head-to-head comparison of four temporal architectures under one dataset, split, imbalance treatment and tuning budget | `results/deep_{1,2,3,4}.csv`, `results/architecture_parameters.csv` |
| 3 — decomposition of classical-formula failure into coefficient drift, static formulation and feature-set limitation | `results/decomposition_{h}.csv`, `results/decomposition_summary.md` |
| 4 — period-level interpretability, validated by two independent methods | `results/figures/attention_by_quarter.*`, `results/figures/shap_family_by_quarter.*`, `results/interpretability_summary.md` |
| 5 — methodological audit of the reported 91–99% accuracies | `results/protocol_audit.csv`, `results/imbalance_ablation_h{1,4}.csv`, `results/external_protocol_audit.csv` |
| Controls and practitioner metrics (Phase H) — does history help the best model, and what does a credit officer get? | `results/h1_temporal_control.csv`, `results/h1b_growth_ratio_explanation.csv`, `results/h4_precision_at_k.csv`, `results/h5_calibration.csv` |

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
python run_all.py --only h report  # rebuild Phase H and the report
python run_all.py --force          # ignore cached runs and retrain
python -m pytest tests/ -q         # 113 tests
```

`--pilot` writes to `results_pilot/` and `reports_pilot/`. The pilot is
deliberately positive-enriched and **is not a reportable number** — it exists to
catch shape and label bugs before committing to a full run. Because its numbers
are never reported it defaults to a cheaper budget (3 seeds, 3 tuning trials)
so the whole pipeline finishes in about half an hour; `--pilot --seeds 5
--trials 30` runs it at the full budget.

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
| **H** | H1 the temporal control for the winning model and H1b the growth-ratio explanation for it; H2 a static MLP separating deep from temporal; H3 pairwise firm-clustered significance; H4 precision@k and capture curves at window and firm level; H5 calibration; H6 size and sector breakdowns | — |
| **audit** | The four-check leakage audit with printed evidence | All four checks pass |
| **report** | Generates `reports/RESULTS.md` from the CSVs | — |

---

## 5. Layout

```
src/
  config.py        paths, the 29 feature names, families, Altman indices, protocol constants
  data.py          the loader, its assertions, the row-set contract, the embargo mask
  utils.py         seeding, git SHAs, table and figure writers
  metrics.py       ROC/PR-AUC, expected cost, DeLong, McNemar, cluster bootstrap,
                   precision@k, capture curves, calibration, Platt scaling
  classical.py     annualisation, Altman Z''/Z', Ohlson, Zmijewski, coefficient re-estimation
  ml_baselines.py  the five ML baselines and their random search
  models.py        the four architectures in PyTorch, with attention exposed
  imbalance.py     class weights, SMOTE, focal loss
  train.py         the single training loop
  experiment.py    run-and-cache, scoring, seed aggregation
  tuning.py        the deep random search
  tabular.py       cached tabular models on any slice of the tensor
  phase_{a..h}.py  one module per phase
  leakage.py       the four-check audit
  report.py        RESULTS.md generation
experiments/       one JSON config per run: what it was asked to do, before it ran
results/           tables (CSV + Markdown), figures (PNG + SVG), predictions, run records
models/            saved weights (gitignored)
reports/           RESULTS.md, leakage_audit.md
docs/DECISIONS.md  every judgment call, with its rationale
tests/             113 tests: unit tests, the nine acceptance criteria,
                   and invariants on the artefacts themselves
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
| `decomposition_significance.csv` | Each ladder step at each horizon, stated separately: does its firm-clustered interval exclude zero? |
| `decomposition_tree_ladder.csv`, `decomposition_tree_ladder_tests.csv` | The same time-axis question walked by gradient boosting instead of a neural net, five seeds a rung, with a clustered CI on the difference |
| `decomposition_drift_diagnostic.csv` | Does refitting Altman's coefficients on train+val recover the ROC-AUC drop? (It does not; the table carries the verdict.) |
| `h1_temporal_control.csv` | XGBoost on 29 ratios at t-0 against the same learner on the flattened window — the control for whether history helps the best model |
| `h1b_growth_ratio_explanation.csv` | The same comparison with the four year-on-year growth ratios removed, which is what explains H1 |
| `h2_static_deep_control.csv` | A static MLP on 29 ratios at t-0, separating "deep" from "temporal" |
| `h3_pairwise_significance.csv` | Every headline architecture pair with firm-clustered intervals, DeLong and McNemar beside them as window-level |
| `h4_precision_at_k.csv`, `h4_capture_curves.csv` | Of the k riskiest, how many fail — at window level and at firm level |
| `h5_calibration.csv`, `h5_calibration_bins.csv` | Reliability, Brier and ECE; scores that are not probabilities are Platt-scaled on validation first |
| `h6_size_breakdown.csv`, `h6_sector_breakdown.csv`, `h6_breakdown_notes.json` | PR-AUC by total-assets tercile and by 2-digit SIC, reporting "too few" below 20 test positives |
| `results_snapshot.json` | Every watched metric from the previous report build; the "Changes from previous run" section diffs against it |
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
`imbalance_heatmap_h{1,4}`, `imbalance_cost_heatmap_h{1,4}`,
`capture_curves_{firm,window}`, `calibration`. Each as both `.png` and `.svg`.

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

## 9. What the run found

<!-- FINDINGS:START -->
_Generated from the CSVs in `results/` by `src/report.py`; do not edit by hand._

**xgboost on the flattened window has the highest PR-AUC of any model here** — 0.1510 at h=4, against the best neural model's (transformer) 0.0729, and Altman Z″'s 0.0305. Same row set, same split, same 30-trial tuning budget, and both are five-seed ensembles.

Temporal architectures at h=4:

| model | PR-AUC (ensemble) | PR-AUC (per-seed mean +/- std) | ROC-AUC (ensemble) |
|---|---|---|---|
| transformer | 0.0729 | 0.0692 ± 0.0122 | 0.8510 |
| cnn_lstm_attn | 0.0599 | 0.0440 ± 0.0060 | 0.8159 |
| lstm | 0.0466 | 0.0390 ± 0.0093 | 0.7684 |
| bilstm | 0.0461 | 0.0353 ± 0.0067 | 0.7566 |


**Does the eight-quarter window help the best model?** Not measurably. The window minus t-0 difference for xgboost is +0.0078 to +0.0553 PR-AUC across the four horizons, and the firm-clustered interval contains zero at 4 of 4 horizons.

| horizon | t0_pr_auc | window_pr_auc | window_minus_t0_pr_auc | pr_auc_cluster_ci_low | pr_auc_cluster_ci_high | pr_auc_cluster_p | window_helps |
|---|---|---|---|---|---|---|---|
| 1 | 0.1042 | 0.1595 | 0.0553 | -0.0253 | 0.1332 | 0.1670 | False |
| 2 | 0.0938 | 0.1250 | 0.0312 | -0.0048 | 0.0675 | 0.1030 | False |
| 3 | 0.1323 | 0.1401 | 0.0078 | -0.0276 | 0.0401 | 0.7490 | False |
| 4 | 0.1151 | 0.1510 | 0.0359 | -0.0024 | 0.0750 | 0.0690 | False |


**Why?** Four of the 29 ratios are year-on-year growth, so a single t-0 row already carries a four-quarter comparison. Dropping r21–r24 and repeating the comparison supports that reading at 3 of 4 horizons.

| horizon | window_gap_with_growth | window_gap_without_growth | growth_worth_to_t0 | growth_worth_to_window | supports_explanation |
|---|---|---|---|---|---|
| 1 | 0.0553 | 0.0448 | -0.0009 | 0.0096 | False |
| 2 | 0.0312 | 0.0707 | 0.0166 | -0.0229 | True |
| 3 | 0.0078 | 0.0443 | 0.0448 | 0.0083 | True |
| 4 | 0.0359 | 0.0437 | 0.0226 | 0.0149 | True |


**The decomposition.** At h=4, of the 0.1689 of total movement from Altman Z″ to the full temporal model: 1% A_zdp->B_levels (+0.0010), 2% B_levels->B_tensor (+0.0028), 24% B_tensor->B_mlp_t0 (+0.0399), 29% B_mlp_t0->C_lstm (+0.0488), 45% C_lstm->D_lstm (-0.0764).

Each step with its firm-clustered interval:

| step | cause | PR-AUC change | 95% CI | excludes 0 |
|---|---|---|---|---|
| A_zdp -> B_levels | coefficient drift | 0.0010 | [-0.0044, +0.0080] | False |
| B_levels -> B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | 0.0028 | [-0.0009, +0.0083] | False |
| B_tensor -> B_mlp_t0 | nonlinearity | 0.0399 | [+0.0204, +0.0665] | True |
| B_mlp_t0 -> C_lstm | the time axis | 0.0488 | [+0.0194, +0.0829] | True |
| C_lstm -> D_lstm | the feature-set expansion | -0.0764 | [-0.1165, -0.0412] | True |
| D_lstm -> D_transformer | architecture | 0.0264 | [+0.0039, +0.0559] | True |


**The protocol audit.** The same LSTM at h=1, scored at matched base rates: PR-AUC 0.8939 under the inflated protocol against 0.0217 under the correct one — a 41x collapse, with both scored on real windows at a 0.0019 base rate. On ROC-AUC, which is base-rate invariant, 74% of the loss is the resampling order and 26% the split.

The same model reports 0.9974 accuracy and loses to a constant predicting no bankruptcy (0.9981).

**What a practitioner gets.** Of the 50 riskiest firms at h=4:

| model | tp | n_positive | precision | recall | lift |
|---|---|---|---|---|---|
| altman_zdp | 1 | 123 | 0.0200 | 0.0081 | 0.5514 |
| xgboost | 17 | 123 | 0.3400 | 0.1382 | 9.3735 |
| transformer | 12 | 123 | 0.2400 | 0.0976 | 6.6166 |
| C_lstm | 18 | 123 | 0.3600 | 0.1463 | 9.9249 |


**Calibration** (scores are Platt-scaled on validation where they are not already probabilities):

| model | scaling | base_rate | mean_predicted | brier | ece |
|---|---|---|---|---|---|
| altman_zdp | Platt-scaled on validation | 0.01127 | 0.00522 | 0.01118 | 0.01033 |
| xgboost | native probability | 0.01127 | 0.00908 | 0.01136 | 0.00541 |
| transformer | native probability | 0.01127 | 0.04445 | 0.01878 | 0.03318 |
| C_lstm | native probability | 0.01127 | 0.07732 | 0.03196 | 0.06605 |


**Who it works for.** PR-AUC at h=4 by total-assets tercile (cut on the train period):

| model | small | mid | large |
|---|---|---|---|
| C_lstm | 0.0689 | 0.1803 | 0.2423 |
| altman_zdp | 0.0265 | 0.0709 | 0.0595 |
| transformer | 0.0427 | 0.1012 | 0.1229 |
| xgboost | 0.0618 | 0.1984 | 0.3194 |


<!-- FINDINGS:END -->

---

## 10. Compute

Measured on one NVIDIA T1000 8GB, Windows 11, PyTorch 2.14 + CUDA 12.6.

| | |
|---|---:|
| Total measured compute, full universe | **14.9 h** |
| Cached deep runs | 315 |
| Tests (113, incl. 32 acceptance checks) | ~4 min |

The pilot reproduction last measured **29.8 min**, before Phase H and the
round-2 additions; it has not been re-timed since, and the firm-clustered
bootstraps will have made it longer. `python run_all.py --pilot` re-measures it.

`reports/RESULTS.md` prints the total measured compute across all cached runs
and tuning trials, and `results/PROVENANCE.json` lists every artefact with its
SHA-256 and both repositories' git SHAs.

The models are small — 31k to 71k parameters — and the binding cost is kernel
launch overhead across roughly 2,700 optimiser steps per epoch at batch 32, not
arithmetic. Larger batch sizes are in the search space for exactly that reason;
the Transformer's search nevertheless chose batch 32, which is why it accounts
for more than half the training compute on its own.

Since the review fixes, the **firm-clustered bootstraps** are the other large
cost: Phase E spends roughly 30 minutes per horizon on them, because each of
~76,000 metric evaluations per horizon has to rebuild a resampled index from
~3,400 per-firm arrays. `metrics._cluster_resample` would vectorise well if that
becomes a problem; it has not been optimised because it is a speed issue rather
than a correctness one.
