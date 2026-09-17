# Decisions — modelling stage

Every judgment call made while building the modelling and evaluation stage, with
a one-line rationale.  The dataset repository's own `docs/DECISIONS.md` covers
everything upstream of the tensors; this file starts where that one stops.

Where a decision deviates from the letter of the specification, it says so
explicitly.

---

## 0. Repository and protocol

| # | Decision | Rationale |
|---|---|---|
| 0.1 | The modelling code lives in its own repository (`capstone-models`), reading the dataset read-only from `../capstone-dataset/` | Spec rule. The dataset is a frozen artefact; nothing here writes to it, and `src/config.py` is the single place the path is configured. |
| 0.2 | **PyTorch**, not Keras | Spec rule. Each architecture is implemented layer for layer against the Keras notation in `Capstone Summary` §6, and `results/architecture_parameters.csv` logs the parameter count so the translation is auditable. |
| 0.3 | Every deep run is cached as `results/preds/<key>.parquet` + `results/runs/<key>.json` and is not repeated if both exist | Makes `run_all.py` resumable, and guarantees the test set is scored exactly once per model — check (c) of the leakage audit rests on it. |
| 0.4 | Pilot runs write to `results_pilot/` and `reports_pilot/` via `CAPSTONE_RESULTS_DIR` | A smoke number can then never overwrite, or be mistaken for, a reported one. The pilot is positive-enriched and is not a result. |
| 0.5 | Early stopping watches **validation PR-AUC**, not validation ROC-AUC as `Capstone Summary` §6 says | With 37 validation positives at h=1, ROC-AUC is dominated by the ranking of 20,807 negatives and moves very little; PR-AUC responds to the positives, which is what is being learned. Logged as a deliberate deviation. |
| 0.6 | Five seeds per configuration; every reported deep number is mean ± std | Spec rule, and necessary at this positive count — single-seed differences at h=1 are smaller than seed noise. |
| 0.7 | Deep runs use an epoch cap of 60 with patience 10 | Patience is the spec's. The cap is ours, as a guard against a run that never triggers early stopping; `results/runs/*.json` records `epochs_run` so any run that hit the cap is visible. |
| 0.8 | Training keeps all tensors resident on the GPU and indexes them with a permutation instead of using a `DataLoader` | 86,592 × 8 × 29 float32 is 80 MB. At batch 32 an epoch is ~2,700 steps, all launch-bound; removing the host round-trip per batch is what makes several hundred 5-seed runs tractable on one GPU. |

---

## 1. Architecture translation (Keras → PyTorch)

| # | Decision | Rationale |
|---|---|---|
| 1.1 | **HEAD-ACT.** The spec writes the Transformer head as `Dense(32)` → `Dense(1, sigmoid)` and the CNN-LSTM-Attention head as `Dense(16)` → … → `Dense(1, sigmoid)`, with no activation named. Both are implemented with **ReLU**. | Keras's `Dense` defaults to a linear activation, and a linear `Dense(32)` feeding a linear `Dense(1)` composes to a single linear map — the layer would be inert. ReLU also matches the LSTM and Bi-LSTM heads, which the spec *does* mark ReLU, so the four models keep comparable head capacity. |
| 1.2 | **CONV-PAD.** `Conv1D(32, kernel=3)` uses `padding='same'`, not Keras's default `'valid'` | The spec does not state padding, and it decides what the attention weights mean. `'valid'` gives 6 steps → pooled to 3, which map onto ragged, overlapping receptive fields. `'same'` gives 8 → pooled to 4, so each attention position is exactly one contiguous quarter pair: (t-7,t-6), (t-5,t-4), (t-3,t-2), (t-1,t-0). Phase F needs that to put attention on a quarter axis. |
| 1.3 | The CNN-LSTM-Attention weights are reported both raw (4 pooled positions) and expanded to 8 quarters by splitting each pooled weight evenly across its two source quarters | The prompt asks for "additive weights over the 8 time steps"; pooling makes 8 unavailable directly. The expansion is stated rather than presented as a native 8-step attention. |
| 1.4 | `Bidirectional(LSTM(64))` with `return_sequences=False` is reproduced as the forward pass's **last** step concatenated with the backward pass's **first** time index | That is what Keras concatenates. Taking `h[:, -1, :]` from PyTorch's bidirectional output would feed the backward direction its *first* processed step, which is the wrong end of the sequence. |
| 1.5 | Transformer blocks are post-norm (`LayerNorm(x + sublayer(x))`) | The spec says "layer norm + residual", which is the original Vaswani et al. ordering. |
| 1.6 | `nn.MultiheadAttention` is called with `average_attn_weights=False` so per-head, per-layer weights survive | Phase F needs them; averaging at source would make the head-level analysis impossible. Weights are requested only at inference, so training is unaffected. |
| 1.7 | Gradient-norm clipping at 1.0 on every architecture | Uniform across the four models, so it cannot advantage one. Guards the recurrent models at high class weights (`pos_weight` is 562 at h=1). |

---

## 2. Hyperparameter search

| # | Decision | Rationale |
|---|---|---|
| 2.1 | 30 random-search trials per architecture and per ML baseline, scored on **validation PR-AUC** | Spec rule: the same budget for every architecture, so differences are attributable to architecture. Logged in `results/tuning/`. |
| 2.2 | **Trial 0 is always the specification's own defaults** (lr 1e-3, batch 32, dropout 0.3, no weight decay) | The search can then only improve on the spec, never silently replace it with something unrelated. `results/tuning_summary.csv` reports the spec default's score next to the winner's. |
| 2.3 | Search space: lr log-uniform [1e-4, 5e-3], batch ∈ {32, 64, 128, 256}, dropout ∈ {0.1…0.5}, weight decay ∈ {0, 1e-5, 1e-4, 1e-3}. The **architecture** — layer count and width — is never searched. | The spec fixes the architecture; it gives training defaults. Searching only training hyperparameters keeps "same input tensor, split, imbalance treatment and tuning budget" true while leaving the architectures exactly as specified. |
| 2.4 | Trials run with a 15-epoch cap and patience 5; the winning configuration is then retrained to convergence (60 / 10) for every reported run | The usual two-stage budget. Recorded in each `results/tuning/<arch>.json` under `budget`. |
| 2.5 | Tuning is done **once per architecture, at h=4, under class weights**, and the winning configuration is reused at every horizon and treatment | h=4 has the most positives, so the selection signal is least noisy. Reusing one configuration across treatments is what makes the Phase D ablation a clean comparison — if each cell were separately tuned, the ablation would measure tuning effort as much as treatment. |
| 2.6 | Tuning trials do produce test predictions internally (the training loop always scores all three splits) but they are **discarded and never written to disk** | Only `run_deep` writes a test prediction file, and it writes one per run key. |

---

## 3. Classical baselines

| # | Decision | Rationale |
|---|---|---|
| 3.1 | **Z″ (1995, four-variable) is the primary Altman baseline**; Z′ (1983, five-variable) is reported alongside | Resolves the discrepancy the prompt flags: `Capstone Summary` §5 says "X₁ through X₅ of the Z″ variant", but Z″ has no X₅. Z″ is the right variant for a mixed non-financial sample, and it is the spec's own guidance in §4. Both are in `results/baselines_{h}.csv`. |
| 3.2 | Coefficients verified against the original papers before use: Z′ = 0.717/0.847/3.107/0.420/0.998 (cuts 1.23 / 2.90); Z″ = 6.56/3.26/6.72/1.05 (cuts 1.10 / 2.60); Ohlson model 1 = −1.32/−0.407/6.03/−1.43/0.0757/−1.72/−2.37/−1.83/0.285/−0.521; Zmijewski = −4.3/−4.5/5.7/−0.004. Unit-tested against a hand-computed worked example. | A coefficient typo would be invisible in an AUC. `tests/test_classical.py` fails if any coefficient moves. |
| 3.3 | The Z″ **no-constant** form is used, paired with the 1.10 / 2.60 cutoffs | Altman's emerging-market "EM Score" adds +3.25 to map onto bond ratings. The constant is additive, so it cannot change any ranking or AUC, but it does shift the cutoffs; the no-constant form and the 1.10 / 2.60 cuts are the internally consistent pair. |
| 3.4 | **Annualisation.** X₃ and X₅, Ohlson's NI/TA and FFO/TL, and Zmijewski's NI/TA use trailing-four-quarter sums over end-quarter total assets | Dataset `docs/DECISIONS.md` 4.1: the panel's flows are quarterly, and these terms are annual flows over a stock in the published formulas. Without this, the fixed coefficients see roughly a quarter of what they expect. X₁, X₂ and X₄ are stock/stock and are untouched. |
| 3.5 | The TTM sum is computed on a **gap-free quarter grid** and requires all four quarters present; otherwise it falls back to 4 × the quarter's own flow and the row is flagged | A positional `shift(4)` would splice non-adjacent quarters wherever the panel has a hole. The fallback rate on the evaluated row set is recorded in `results/classical_levels_notes.json` and reported in `RESULTS.md`. |
| 3.6 | Annualisation is applied to Model B as well, even though re-estimated coefficients would absorb the scale | Prompt rule, and it is what keeps A→B a pure coefficient comparison. |
| 3.7 | Ohlson's SIZE uses the **US GDP implicit price deflator** (FRED `GDPDEF`), natural log, with the series committed to `data/gdpdef.csv` | Ohlson's original index is the GNP price level with 1968 = 100; the current GDPDEF vintage is 2017 = 100. The base year enters as an additive constant inside a log, so it shifts the intercept only and cannot change any ranking. Committing the series keeps the pipeline reproducible offline. |
| 3.8 | Total liabilities falls back to `LiabilitiesAndStockholdersEquity − StockholdersEquity` where the `Liabilities` tag is absent | The accounting identity, used on 446 of 284,915 panel rows and counted in the notes file. A NaN here would drop a window from three of the four formulas. |
| 3.9 | Missing classical variables are filled with the **train median**, and the per-variable missing rate is reported | Every model must score the identical row set, so a row cannot be dropped for one model and kept for another. The median is a fitted statistic, so it is fitted on train rows only (leakage rule 1) and saved. Coverage on the row set is 97–100% for every variable, so the fill changes little. |
| 3.10 | Altman is reported **both** threshold-free (as a ranked score, AUC) and at its canonical cutoffs, under two alarm rules (`distress` only, and `distress or grey`) | Spec rule. Reporting only the cutoffs would look rigged; reporting only the AUC would hide that the published cutoffs alarm on ~44% of windows at a ~1% base rate. |
| 3.11 | Altman scores are negated before ranking | Z is a health score: low means distress. Feeding the raw Z to an AUC would report 1 − AUC. |

---

## 4. Machine-learning baselines

| # | Decision | Rationale |
|---|---|---|
| 4.1 | Input is the 8 × 29 window flattened to 232 features plus the two structural indicators at t-0 (234) | Spec rule. The ML baselines then see exactly the information the deep models see, without the time axis being modelled. |
| 4.2 | The RBF SVM is approximated with a **Nystroem feature map + a linear smoothed-hinge (modified Huber) classifier**, not `SVC(kernel='rbf')` | An exact kernel SVC is O(n²) in the 86,592 training rows and `probability=True` adds a 5-fold Platt calibration on top; it is not tractable here. Modified Huber is in the SVM loss family and emits the probability outputs the spec requires. The **exact** RBF SVC is used on the external UCI sets in Phase G, which are small enough for it — so the exact model is still reported somewhere. |
| 4.3 | Class imbalance in the ML baselines is handled with `class_weight='balanced'` (and `scale_pos_weight` for XGBoost) | Matches the deep models' default treatment, so the baseline-versus-deep comparison is not confounded by a different imbalance strategy. |
| 4.4 | Stacking uses 3-fold **out-of-fold** predictions computed inside train for the meta-learner, with the base models then refit on all of train | Fitting the meta-learner on validation predictions would consume the split that the threshold is later chosen on. Nothing in the stack sees val or test. |
| 4.5 | Each ML baseline is tuned separately **per horizon** | They are cheap (1–10 s per fit), so the reason for reusing one configuration across horizons in the deep models does not apply. |

---

## 5. Imbalance ablation

| # | Decision | Rationale |
|---|---|---|
| 5.1 | SMOTE resamples to **full balance** on the flattened window, using `imbalanced-learn` | That is the standard practice the literature applies and the practice Contribution 5 is auditing, so it is what should be reproduced. Its artefact is stated in the code and in `RESULTS.md`: an interpolated row is a blend of two real firms' eight-quarter histories, so its quarter-to-quarter dynamics are a linear mix, not an observed path. |
| 5.2 | Class weights are inverse-frequency, computed on train, passed as `pos_weight` to `BCEWithLogitsLoss` | Spec rule. The statistics are recorded in every run's JSON, which is what leakage check (b) reads. |
| 5.3 | Focal loss uses γ = 2, α = 0.25 | Spec rule (Chang, Liu & Deng 2025). |
| 5.4 | The **cost-sensitive threshold is applied on top of the untreated (`none`) model**, not trained as a fifth model | It is a decision rule, not a training treatment. Applying it to the untreated baseline isolates what a better threshold alone buys, which is the more useful comparison; expected cost at all four ratios is also reported for every other treatment at its F1-chosen threshold. |
| 5.5 | Cost ratios are read as **Type II : Type I** — a missed bankruptcy costs 1, 10, 20 or 50 times a false alarm | The direction that matches the literature: missing a failure is the expensive error. Reported as expected cost **per window** so the numbers are comparable across splits of different size. |
| 5.6 | The ablation runs at h = 1 and h = 4 only | The prompt's choice: the two extremes. h=1 is the hardest (154 train positives) and h=4 the most stable. |

---

## 6. Protocol audit

| # | Decision | Rationale |
|---|---|---|
| 6.1 | A textbook SMOTE is implemented in `imbalance.smote_with_parents` for the audit only, alongside the `imbalanced-learn` version used everywhere else | To reproduce "SMOTE before the split" faithfully, a synthetic row must inherit the split of the real firm it was interpolated from; `imbalanced-learn` does not expose that parent index. The algorithm is Chawla et al. (2002), and the main ablation still uses the library implementation. |
| 6.2 | The **Inflated** protocol reports accuracy at a fixed 0.5 threshold on a random 80/20 split of the balanced, pooled data, with the inner validation slice taken from its own training portion | That is the procedure the criticised papers describe. Reporting anything else would be attacking a straw man. |
| 6.3 | The **Half-fixed** protocol keeps the chronological split but resamples first, so synthetic positives cross the split boundary | Isolates the two mistakes from each other: Inflated → Half-fixed is the cost of the random split, Half-fixed → Correct the cost of resampling before splitting. |
| 6.4 | All three protocols report accuracy **and** PR-AUC | The point is not that accuracy is high, it is that accuracy is uninformative at a 1% base rate while PR-AUC is not. Showing only one metric per protocol would prove nothing. |
| 6.5 | The audit is repeated on the UCI Taiwanese and Polish sets in Phase G | Those are the datasets seven of the twenty-one reviewed papers use, so reproducing the inflation there is the more direct demonstration. |

---

## 7. Decomposition (A → B → C → D)

| # | Decision | Rationale |
|---|---|---|
| 7.1 | **Model B is fitted twice**: `B_levels` on the annualised panel levels, and `B_tensor` on the tensor's own five ratios at t-0 | A→B must differ only in coefficients, which requires B to use A's variable definitions; B→C must differ only in the time axis, which requires B to use C's. Fitting both and reporting both keeps each gap attributable instead of quietly absorbing a change of variable definition. `B_levels` is the primary ladder; `B_tensor` sits in `decomposition_all.csv`. |
| 7.2 | Model C uses tensor feature indices [3, 9, 8, 13, 15] = r04, r10, r09, r14, r16 — Altman's X₁…X₅ in order | Dataset `SESSION_SUMMARY.md` §5. Asserted in `tests/test_data.py`. |
| 7.3 | Models B and C are compared against the **Z″ four-variable** A rung, while C and D see five features | Z″ is the primary A. `A_zp` (the five-variable Z′) is carried in the full ladder table so the five-variable comparison is also available. |
| 7.4 | DeLong and McNemar run on the **seed-averaged** predicted probability of the deep rungs | Both tests take a single prediction vector. Averaging five seeds is a standard ensemble and is more stable than picking one seed arbitrarily; per-seed run records remain on disk. |
| 7.5 | PR-AUC differences get a **stratified bootstrap** CI, not DeLong | DeLong's estimator is defined for ROC-AUC only. Both are reported: DeLong on ROC-AUC differences, bootstrap on PR-AUC differences. |
| 7.6 | Gap shares are reported even when a step is negative (a rung that loses ground) | A negative share is a finding — it says that step did not help — and suppressing it would misstate the decomposition. Shares are on the total A→D gap, so they sum to 1 by construction. |
| 7.7 | The missingness sanity check **retrains** C and D on the all-complete subset rather than only re-scoring | Re-scoring would leave the models trained on the fuller data; retraining is the stricter check the prompt asks for. Run at h=1 and h=4. |
| 7.8 | Model D is repeated with the Transformer | Prompt rule: the finding must not be LSTM-specific. |

---

## 8. Interpretability

| # | Decision | Rationale |
|---|---|---|
| 8.1 | Transformer "importance of quarter t" is the attention that position t **receives**, averaged over query positions, heads and both layers | The classifier reads a global average over positions, so what matters is how much each key position is attended to, not what it attends to. |
| 8.2 | A quarter is excluded from every attention average when its **whole row of the mask is zero** — the quarter was forward-filled and carries no information of its own | Prompt rule ("masking imputed cells"). The per-step observed fraction is reported next to every attention number so the reader can see how much was excluded. |
| 8.3 | Attention is reported for true positives, true negatives, all positives and all negatives separately | The prompt asks for TP versus TN; the unconditional groups are added because TP/TN depend on the chosen threshold and the unconditional split does not. |
| 8.4 | SHAP uses `shap.GradientExplainer` (expected gradients) | The prompt allows KernelSHAP or DeepSHAP. Expected gradients is the gradient-based member of the DeepSHAP family; KernelSHAP over 232 inputs for thousands of windows is not tractable, and DeepExplainer's layer support does not cover cuDNN LSTM cleanly. Stated in `results/interpretability_summary.md`. |
| 8.5 | SHAP explains **all** test positives plus a random 1,500 negatives, against a 200-window train background | Explaining every window is unnecessary and slow; including all positives means the positive-class attribution is not a sample. |
| 8.6 | Masked cells are dropped from every |SHAP| average | A cell whose input was set to the train mean produces an attribution about the imputation, not about the firm. |
| 8.7 | Agreement between the two methods is quantified with **Spearman ρ and Kendall τ** over the 8 quarters, plus whether they agree on the single top quarter | Eight points is too few for a correlation to be decisive on its own, so the argmax agreement is reported next to it. Disagreement is reported as a finding. |

---

## 9. Robustness

| # | Decision | Rationale |
|---|---|---|
| 9.1 | Rolling-origin folds hold the **last four quarters of each fold's training range** back as an inner validation set for early stopping and thresholding | A fold has to stop and threshold on something, and using the fold's evaluation period would leak it. Keeping the inner split chronological keeps the fold internally consistent with the main protocol. |
| 9.2 | Rolling-origin CV runs at **h = 4 only** | At h=1 a single fold carries roughly ten positives, so a per-fold PR-AUC would be noise. h=4 is the horizon where the per-fold comparison is meaningful. |
| 9.3 | The embargo split is **derived from the manifest** rather than by re-running the dataset pipeline | The dataset repo's `--embargo` drops val windows whose input starts on or before the train cut-off and test windows whose input starts before the test period; it leaves train untouched, so the train-fitted scaler is bit-identical and masking the existing tensors reproduces the split exactly. This also honours the instruction not to write to the dataset repository. |
| 9.4 | The embargo check runs at h = 4 and **records h = 1 as not evaluable** | The embargoed validation split retains only windows ending 2021Q4 — 2,642 of 20,844 — and contains **zero** h=1 positives. There is nothing to early-stop or threshold on. Reported as a limitation of the stricter protocol rather than worked around. |
| 9.5 | The embargo comparison also scores the **standard model on the embargoed rows** | Otherwise the comparison confounds "stricter split" with "smaller, later test set". Both numbers are in `results/embargo.csv`. |
| 9.6 | External validation uses a **stratified random** 70/15/15 split | The UCI sets are single-snapshot and carry no time axis, so a chronological split is impossible. Stated as such rather than presented as equivalent to the main protocol. |
| 9.7 | Polish `.arff` missing values are filled with the column median | The set is 1.3% missing overall; dropping rows would discard positives. |

---

## 10. Reporting

| # | Decision | Rationale |
|---|---|---|
| 10.1 | `reports/RESULTS.md` is generated entirely by `src/report.py` from the CSVs in `results/` | Spec rule. No number in the report is typed, so it cannot drift from the artefacts or contain a placeholder. |
| 10.2 | Every table is written as **both** CSV and Markdown | The CSV is the machine-readable artefact the report reads; the Markdown sits next to it for reading in a browser. |
| 10.3 | Every figure is written as **both** PNG and SVG | Spec rule; the SVG is what goes into the paper. |
| 10.4 | Every run record and every tuning study carries the git SHA of **both** repositories and the seed | Spec rule. |
| 10.5 | PR-AUC is presented first everywhere, with ROC-AUC beside it | At a 0.18–1.1% positive rate, ROC-AUC flatters every model; leading with it would repeat the mistake the paper is criticising. |
