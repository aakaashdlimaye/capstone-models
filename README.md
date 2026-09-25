# capstone-models

Modelling and evaluation for

> **Beyond the Z-Score: Decomposing the Failure of Static Bankruptcy Formulas
> with Sequence Models**

The capstone was registered as *"Bankruptcy Prediction using Temporal Deep
Learning: A Comparative Study of LSTM and Transformer Models"*, and that remains
its title for the university. The paper title changed because the results did
not support the original framing: a gradient-boosted tree on the flattened
window outperforms all four temporal architectures, so the contribution is the
decomposition and the methodological audit rather than a win for sequence
models. See §9.

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
python run_all.py --only h report  # rebuild Phase H and the report
python run_all.py --force          # ignore cached runs and retrain
python -m pytest tests/ -q         # 71 unit tests
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
| **H** | The missing controls (XGBoost at t-0 vs the window; a static MLP), pairwise firm-clustered significance, precision@k and capture curves, calibration, and size/sector breakdowns | — |
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
  tabular.py       cached tabular models on any slice of the tensor
  phase_{a..h}.py  one module per phase
  leakage.py       the four-check audit
  report.py        RESULTS.md generation
experiments/       one JSON config per run: what it was asked to do, before it ran
results/           tables (CSV + Markdown), figures (PNG + SVG), predictions, run records
models/            saved weights (gitignored)
reports/           RESULTS.md, leakage_audit.md
docs/DECISIONS.md  every judgment call, with its rationale
tests/             71 unit tests, including the nine acceptance criteria
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

## 9. What the run found

Headline numbers from the artefacts in `results/`. Every one is traceable to a
CSV and a seed; see `reports/RESULTS.md` for the full tables.

**A tuned XGBoost on the flattened window has the highest PR-AUC of any model
here** — 0.149 at h=4, against the best neural model's 0.069. Gradient boosting
on 232 flattened ratio-quarters beats all four temporal architectures by roughly
2×, on the same row set, the same split and the same 30-trial tuning budget.
That is the headline, and it is why the paper is framed around the decomposition
rather than around sequence models winning.

**Among the temporal architectures the Transformer is best**, and the only one
that does not decay as the horizon lengthens: test PR-AUC at h=4 is Transformer
0.069 ± 0.012, CNN-LSTM-Attn 0.044 ± 0.006, LSTM 0.039 ± 0.009, Bi-LSTM
0.035 ± 0.007, against Altman Z″'s 0.031. Across h=1…4 the Transformer stays in
0.069–0.079 while the others fall away. Phase H reports firm-clustered intervals
on every one of those comparisons, so the ordering can be read with its
uncertainty rather than as a ranking.

Numbers quoted here are the **per-seed mean ± std**. Every table also carries
the **seed-ensemble** value (the metric of the averaged prediction), which is
the higher of the two and is what the significance tests use. Both conventions
are named in every column; see `docs/DECISIONS.md` 4.x.

**The decomposition answers its question, and the answer is not the expected
one.** At h=4, of the total movement from Altman Z″ to a full temporal model,
**1% is coefficient drift** (+0.0010 PR-AUC, 95% CI [−0.0021, +0.0047] — not
distinguishable from zero), **54% is the static formulation** (+0.0915, CI
[+0.0657, +0.1254]) and **45% is the feature-set expansion moving the wrong way**
(−0.0764, CI [−0.1088, −0.0504]). Re-estimating Altman's sixty-year-old
coefficients on modern data buys nothing measurable; giving the same five ratios
a time axis is what helps; and widening five ratios to twenty-nine *hurts* the
LSTM significantly. The Transformer recovers part of that loss, so the last step
is partly an LSTM capacity limit rather than a clean statement about features.
The gaps survive retraining on the subset where all five Altman ratios are
observed in all eight quarters, so they are not a missingness artefact.

**The protocol audit reproduces the inflated accuracies and locates them.** The
same LSTM, three ways at h=1: PR-AUC **0.9998** under the inflated protocol
(random split, SMOTE before splitting), 0.9511 half-fixed, **0.0217** under the
correct one — a **46× collapse**, 95% of it attributable to resampling before
the split rather than to the random split itself. The diagnostic that makes the
point concrete: the correctly-evaluated model reports **99.74% accuracy and
loses to a constant that predicts no bankruptcy at all** (99.81%), while its
PR-AUC is 11.6× the base rate. The same audit on UCI Polish gives PR-AUC 0.999
inflated against 0.494 correct, and on UCI Taiwanese 0.998 against 0.319.

**No training-side imbalance treatment beats a well-chosen threshold.** At h=4
the Transformer's best expected cost at a 50:1 miss-to-false-alarm ratio comes
from cost-sensitive thresholding on the *untreated* model (0.197, recall 0.304),
not from class weights (0.217), SMOTE (0.222) or focal loss (0.221). SMOTE and
focal loss both reduce the Transformer's PR-AUC (0.041 and 0.047 against 0.066
untreated).

**Altman at its published cutoffs alarms on 44% of windows** at a ~1% base rate:
recall 0.93, precision 0.024. Threshold-free it is a reasonable ranker
(ROC-AUC 0.81–0.85); as a decision rule it is unusable at this base rate.

**Interpretability agrees on the recent quarters and disagrees on which one.**
Both methods put the most weight on the final quarters; the Transformer's
attention and SHAP agree that t-0 matters most (Spearman ρ = 0.40, not
significant over 8 points), while the CNN-LSTM-Attention weights correlate
strongly with SHAP (ρ = 0.93, p = 0.0009) but peak at t-1 rather than t-0. The
disagreement is reported rather than resolved. Three of Altman's five ratios sit
in the bottom third of the SHAP ranking.

**The 2020–2021 COVID fold is not an outlier for the deep model** — rolling-origin
PR-AUC is 0.087, 0.111, 0.107, 0.102 across the four expanding folds — but it is
for Altman Z″, which drops to 0.013 from ~0.030 elsewhere.

---

## 10. Compute

Measured on one NVIDIA T1000 8GB, Windows 11, PyTorch 2.14 + CUDA 12.6.

| | |
|---|---:|
| Total measured compute, full universe | **13.7 h** |
| Cached deep runs | 275 |
| Full pipeline on the pilot, every gate | 29.8 min |
| Unit tests (71, incl. 15 acceptance checks) | ~3 min |

`reports/RESULTS.md` prints the total measured compute across all cached runs
and tuning trials, and `results/PROVENANCE.json` lists every artefact with its
SHA-256 and both repositories' git SHAs.

The models are small — 31k to 71k parameters — and the binding cost is kernel
launch overhead across roughly 2,700 optimiser steps per epoch at batch 32, not
arithmetic. Larger batch sizes are in the search space for exactly that reason;
the Transformer's search nevertheless chose batch 32, which is why it accounts
for more than half the total compute on its own.
