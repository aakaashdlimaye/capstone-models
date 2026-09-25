# Decomposition summary

Every model here scores the identical row set.  **A** applies Altman's published
Z″ coefficients to the window's end quarter; **B_levels** re-estimates those
coefficients on train; **B_tensor** refits them on the tensor's own z-scored
quarterly ratios; **B_mlp_t0** makes that nonlinear but still static;
**C_lstm** adds the eight-quarter time axis; **D_lstm** adds the other 24 ratios.

The ladder walks one change at a time on purpose.  The earlier B→C step moved
four things at once — linear to nonlinear, static to sequential, annualised
levels to z-scored quarterly ratios, and one fit to a five-seed ensemble — and
credited all of it to the time axis.

**Intervals are firm-clustered bootstraps.**  Stride-1 windowing gives one firm
up to ~50 overlapping windows sharing seven of eight quarters and one event, so a
window-level interval is too narrow.  DeLong and the window-level bootstrap are
carried in `decomposition_all.csv` under `window_level_` names for comparison.

Shares are of the **total absolute movement** across the steps, not of the net
A→D gap: a step moves backwards, and dividing by a small net total would report
an ordinary step as several hundred percent.

## Horizon h = 1

> At h=1, test PR-AUC moves from 0.0066 for Altman Z″ to 0.0847 for the full temporal model, a net +0.0781.  Of the 0.1241 of total movement across the 5 steps, 1% is coefficient drift (+0.0015, improves), 0% is preprocessing (annualised levels to z-scored quarterly ratios) (+0.0001, improves), 20% is nonlinearity (+0.0246, improves), 60% is the time axis (+0.0749, improves), 19% is the feature-set expansion (-0.0230, degrades).

| step | cause | PR-AUC change | PR-AUC cluster 95% CI | excludes 0 | share of movement | ROC-AUC change | ROC-AUC cluster 95% CI | ROC excludes 0 | DeLong p (window-level) | McNemar p (window-level) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0015 | [-0.0003, +0.0037] | False | 0.0122 | -0.0166 | [-0.0660, +0.0242] | False | 0.4622 | 0.0000 |
| B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | 0.0001 | [-0.0032, +0.0023] | False | 0.0005 | -0.0225 | [-0.0603, +0.0075] | False | 0.2205 | 0.0000 |
| B_tensor->B_mlp_t0 | nonlinearity | 0.0246 | [+0.0078, +0.0759] | True | 0.1983 | 0.0283 | [-0.0065, +0.0598] | False | 0.1055 | 0.0000 |
| B_mlp_t0->C_lstm | the time axis | 0.0749 | [+0.0144, +0.1486] | True | 0.6035 | 0.0399 | [-0.0011, +0.0843] | False | 0.0808 | 0.2148 |
| C_lstm->D_lstm | the feature-set expansion | -0.0230 | [-0.1012, +0.0523] | False | 0.1855 | -0.0116 | [-0.0488, +0.0229] | False | 0.5261 | 0.6985 |


## Horizon h = 2

> At h=2, test PR-AUC moves from 0.0152 for Altman Z″ to 0.0683 for the full temporal model, a net +0.0531.  Of the 0.1707 of total movement across the 5 steps, 1% is coefficient drift (+0.0012, improves), 1% is preprocessing (annualised levels to z-scored quarterly ratios) (+0.0010, improves), 22% is nonlinearity (+0.0384, improves), 42% is the time axis (+0.0713, improves), 34% is the feature-set expansion (-0.0588, degrades).

| step | cause | PR-AUC change | PR-AUC cluster 95% CI | excludes 0 | share of movement | ROC-AUC change | ROC-AUC cluster 95% CI | ROC excludes 0 | DeLong p (window-level) | McNemar p (window-level) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0012 | [-0.0019, +0.0045] | False | 0.0071 | -0.0250 | [-0.0656, +0.0096] | False | 0.1047 | 0.0000 |
| B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | 0.0010 | [-0.0008, +0.0031] | False | 0.0058 | -0.0016 | [-0.0210, +0.0137] | False | 0.8167 | 0.0000 |
| B_tensor->B_mlp_t0 | nonlinearity | 0.0384 | [+0.0173, +0.0854] | True | 0.2248 | 0.0830 | [+0.0569, +0.1129] | True | 0.0000 | 0.0000 |
| B_mlp_t0->C_lstm | the time axis | 0.0713 | [+0.0268, +0.1135] | True | 0.4178 | -0.0043 | [-0.0256, +0.0155] | False | 0.6479 | 0.0004 |
| C_lstm->D_lstm | the feature-set expansion | -0.0588 | [-0.1186, -0.0038] | True | 0.3445 | -0.0156 | [-0.0421, +0.0104] | False | 0.1524 | 0.0000 |


## Horizon h = 3

> At h=3, test PR-AUC moves from 0.0233 for Altman Z″ to 0.0384 for the full temporal model, a net +0.0150.  Of the 0.1958 of total movement across the 5 steps, 1% is coefficient drift (+0.0013, improves), 1% is preprocessing (annualised levels to z-scored quarterly ratios) (+0.0020, improves), 20% is nonlinearity (+0.0389, improves), 32% is the time axis (+0.0633, improves), 46% is the feature-set expansion (-0.0904, degrades).

| step | cause | PR-AUC change | PR-AUC cluster 95% CI | excludes 0 | share of movement | ROC-AUC change | ROC-AUC cluster 95% CI | ROC excludes 0 | DeLong p (window-level) | McNemar p (window-level) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0013 | [-0.0030, +0.0064] | False | 0.0066 | -0.0311 | [-0.0691, +0.0039] | False | 0.0147 | 0.0000 |
| B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | 0.0020 | [-0.0006, +0.0054] | False | 0.0101 | -0.0010 | [-0.0186, +0.0141] | False | 0.8612 | 0.0000 |
| B_tensor->B_mlp_t0 | nonlinearity | 0.0389 | [+0.0195, +0.0713] | True | 0.1985 | 0.0857 | [+0.0657, +0.1088] | True | 0.0000 | 0.0000 |
| B_mlp_t0->C_lstm | the time axis | 0.0633 | [+0.0277, +0.1003] | True | 0.3232 | 0.0157 | [+0.0024, +0.0277] | True | 0.0019 | 0.2853 |
| C_lstm->D_lstm | the feature-set expansion | -0.0904 | [-0.1324, -0.0513] | True | 0.4616 | -0.1161 | [-0.1523, -0.0796] | True | 0.0000 | 0.4477 |


## Horizon h = 4

> At h=4, test PR-AUC moves from 0.0305 for Altman Z″ to 0.0466 for the full temporal model, a net +0.0161.  Of the 0.1689 of total movement across the 5 steps, 1% is coefficient drift (+0.0010, improves), 2% is preprocessing (annualised levels to z-scored quarterly ratios) (+0.0028, improves), 24% is nonlinearity (+0.0399, improves), 29% is the time axis (+0.0488, improves), 45% is the feature-set expansion (-0.0764, degrades).

| step | cause | PR-AUC change | PR-AUC cluster 95% CI | excludes 0 | share of movement | ROC-AUC change | ROC-AUC cluster 95% CI | ROC excludes 0 | DeLong p (window-level) | McNemar p (window-level) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0010 | [-0.0044, +0.0080] | False | 0.0062 | -0.0376 | [-0.0776, -0.0010] | True | 0.0011 | 0.0000 |
| B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | 0.0028 | [-0.0009, +0.0083] | False | 0.0168 | -0.0015 | [-0.0192, +0.0142] | False | 0.7779 | 0.0000 |
| B_tensor->B_mlp_t0 | nonlinearity | 0.0399 | [+0.0204, +0.0665] | True | 0.2360 | 0.0967 | [+0.0757, +0.1204] | True | 0.0000 | 0.0000 |
| B_mlp_t0->C_lstm | the time axis | 0.0488 | [+0.0194, +0.0829] | True | 0.2886 | 0.0068 | [-0.0062, +0.0190] | False | 0.1582 | 0.0000 |
| C_lstm->D_lstm | the feature-set expansion | -0.0764 | [-0.1165, -0.0412] | True | 0.4524 | -0.1082 | [-0.1419, -0.0731] | True | 0.0000 | 0.2888 |


## Significance, stated per horizon

Each step is reported at each horizon separately.  A step that clears zero
at h=4 need not clear it at h=1, where there are 60 test positives rather
than 363, and the table below does not generalise across horizons.

**h = 1**

- A_zdp->B_levels: PR-AUC +0.0015 [-0.0003, +0.0037], ROC-AUC -0.0166 [-0.0660, +0.0242] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- B_levels->B_tensor: PR-AUC +0.0001 [-0.0032, +0.0023], ROC-AUC -0.0225 [-0.0603, +0.0075] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- B_tensor->B_mlp_t0: PR-AUC +0.0246 [+0.0078, +0.0759], ROC-AUC +0.0283 [-0.0065, +0.0598] — PR-AUC interval excludes zero; ROC-AUC interval contains zero.
- B_mlp_t0->C_lstm: PR-AUC +0.0749 [+0.0144, +0.1486], ROC-AUC +0.0399 [-0.0011, +0.0843] — PR-AUC interval excludes zero; ROC-AUC interval contains zero.
- C_lstm->D_lstm: PR-AUC -0.0230 [-0.1012, +0.0523], ROC-AUC -0.0116 [-0.0488, +0.0229] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- D_lstm->D_transformer: PR-AUC +0.0164 [-0.0212, +0.0532], ROC-AUC -0.0249 [-0.0692, +0.0173] — PR-AUC interval contains zero; ROC-AUC interval contains zero.

**h = 2**

- A_zdp->B_levels: PR-AUC +0.0012 [-0.0019, +0.0045], ROC-AUC -0.0250 [-0.0656, +0.0096] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- B_levels->B_tensor: PR-AUC +0.0010 [-0.0008, +0.0031], ROC-AUC -0.0016 [-0.0210, +0.0137] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- B_tensor->B_mlp_t0: PR-AUC +0.0384 [+0.0173, +0.0854], ROC-AUC +0.0830 [+0.0569, +0.1129] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.
- B_mlp_t0->C_lstm: PR-AUC +0.0713 [+0.0268, +0.1135], ROC-AUC -0.0043 [-0.0256, +0.0155] — PR-AUC interval excludes zero; ROC-AUC interval contains zero.
- C_lstm->D_lstm: PR-AUC -0.0588 [-0.1186, -0.0038], ROC-AUC -0.0156 [-0.0421, +0.0104] — PR-AUC interval excludes zero; ROC-AUC interval contains zero.
- D_lstm->D_transformer: PR-AUC +0.0313 [-0.0077, +0.0818], ROC-AUC -0.0058 [-0.0332, +0.0223] — PR-AUC interval contains zero; ROC-AUC interval contains zero.

**h = 3**

- A_zdp->B_levels: PR-AUC +0.0013 [-0.0030, +0.0064], ROC-AUC -0.0311 [-0.0691, +0.0039] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- B_levels->B_tensor: PR-AUC +0.0020 [-0.0006, +0.0054], ROC-AUC -0.0010 [-0.0186, +0.0141] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- B_tensor->B_mlp_t0: PR-AUC +0.0389 [+0.0195, +0.0713], ROC-AUC +0.0857 [+0.0657, +0.1088] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.
- B_mlp_t0->C_lstm: PR-AUC +0.0633 [+0.0277, +0.1003], ROC-AUC +0.0157 [+0.0024, +0.0277] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.
- C_lstm->D_lstm: PR-AUC -0.0904 [-0.1324, -0.0513], ROC-AUC -0.1161 [-0.1523, -0.0796] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.
- D_lstm->D_transformer: PR-AUC +0.0517 [+0.0226, +0.0954], ROC-AUC +0.0925 [+0.0604, +0.1264] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.

**h = 4**

- A_zdp->B_levels: PR-AUC +0.0010 [-0.0044, +0.0080], ROC-AUC -0.0376 [-0.0776, -0.0010] — PR-AUC interval contains zero; ROC-AUC interval excludes zero.
- B_levels->B_tensor: PR-AUC +0.0028 [-0.0009, +0.0083], ROC-AUC -0.0015 [-0.0192, +0.0142] — PR-AUC interval contains zero; ROC-AUC interval contains zero.
- B_tensor->B_mlp_t0: PR-AUC +0.0399 [+0.0204, +0.0665], ROC-AUC +0.0967 [+0.0757, +0.1204] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.
- B_mlp_t0->C_lstm: PR-AUC +0.0488 [+0.0194, +0.0829], ROC-AUC +0.0068 [-0.0062, +0.0190] — PR-AUC interval excludes zero; ROC-AUC interval contains zero.
- C_lstm->D_lstm: PR-AUC -0.0764 [-0.1165, -0.0412], ROC-AUC -0.1082 [-0.1419, -0.0731] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.
- D_lstm->D_transformer: PR-AUC +0.0264 [+0.0039, +0.0559], ROC-AUC +0.0825 [+0.0539, +0.1112] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero.

## Is the A→B ROC-AUC drop stale coefficients or distribution shift?

Re-estimating Altman's coefficients leaves PR-AUC flat but *lowers* ROC-AUC
at the longer horizons, so 'worth nothing' understates it: on the ranking
metric the refit is actively worse.  One explanation is that the training
period is no longer representative of the test period.  Refitting on
train + val — two years closer to the test window, and still strictly before
it — tests that: if shift were the cause, the drop should shrink.  This is a
diagnostic only; nothing fitted on val scores a headline number.

| horizon | published_roc_auc | train_only_roc_auc | train_plus_val_roc_auc | roc_drop_train_only | roc_drop_train_plus_val | shift_explains_drop | roc_drop_cluster_ci_low | roc_drop_cluster_ci_high |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.84504 | 0.82844 | 0.82910 | -0.01660 | -0.01594 | True | -0.06533 | 0.02460 |
| 2 | 0.82988 | 0.80484 | 0.80486 | -0.02504 | -0.02502 | True | -0.06380 | 0.00955 |
| 3 | 0.82244 | 0.79136 | 0.79118 | -0.03108 | -0.03125 | False | -0.06884 | 0.00428 |
| 4 | 0.81225 | 0.77460 | 0.77464 | -0.03765 | -0.03760 | True | -0.07887 | -0.00092 |


## The same question without a neural network

If the time axis is what pays, it should pay for a gradient-boosted tree
too.  B_xgb_t0 sees Altman's five ratios at t-0; C_xgb sees the same five
across all eight quarters (40 inputs).  Same tuning budget, same row set.

| name | horizon | n_inputs | roc_auc | pr_auc | recall | note |
|---|---|---|---|---|---|---|
| decompB_xgb_t0 | 1 | 5 | 0.8621 | 0.0665 | 0.0833 | Altman's five ratios at t-0 only |
| decompC_xgb | 1 | 40 | 0.8683 | 0.1655 | 0.2333 | Altman's five ratios across all eight quarters (40 inputs) |
| decompB_xgb_t0 | 2 | 5 | 0.8690 | 0.0478 | 0.0316 | Altman's five ratios at t-0 only |
| decompC_xgb | 2 | 40 | 0.8668 | 0.1414 | 0.1899 | Altman's five ratios across all eight quarters (40 inputs) |
| decompB_xgb_t0 | 3 | 5 | 0.8583 | 0.0570 | 0.1115 | Altman's five ratios at t-0 only |
| decompC_xgb | 3 | 40 | 0.8599 | 0.1272 | 0.1192 | Altman's five ratios across all eight quarters (40 inputs) |
| decompB_xgb_t0 | 4 | 5 | 0.8563 | 0.0648 | 0.0303 | Altman's five ratios at t-0 only |
| decompC_xgb | 4 | 40 | 0.8423 | 0.0984 | 0.0523 | Altman's five ratios across all eight quarters (40 inputs) |


## Is the finding LSTM-specific?

Model D is repeated with the Transformer.  Where the LSTM loses ground on the
full 29-ratio feature set, the Transformer recovers part of it, so C→D is
partly an LSTM capacity limitation rather than a pure statement about the
feature set.  Both are in `decomposition_all.csv`.

| horizon | C_lstm PR-AUC | D_lstm PR-AUC | D_transformer PR-AUC | C_lstm ROC-AUC | D_lstm ROC-AUC | D_transformer ROC-AUC |
|---|---|---|---|---|---|---|
| 1.0000 | 0.1077 | 0.0847 | 0.1011 | 0.8741 | 0.8625 | 0.8376 |
| 2.0000 | 0.1271 | 0.0683 | 0.0996 | 0.8820 | 0.8663 | 0.8606 |
| 3.0000 | 0.1288 | 0.0384 | 0.0901 | 0.8918 | 0.7757 | 0.8681 |
| 4.0000 | 0.1230 | 0.0466 | 0.0729 | 0.8766 | 0.7684 | 0.8510 |


## Missingness sanity check

Models C and D retrained on the subset of windows where all five Altman ratios
are observed in all eight quarters.  If the gaps were a missingness artefact
they would close here.

| horizon | model | subset | n_test | n_pos | roc_auc | pr_auc | train_frac_complete | test_frac_complete |
|---|---|---|---|---|---|---|---|---|
| 1 | C_lstm | all-five-complete | 30678 | 57 | 0.8750 | 0.1142 | 0.9167 | 0.9523 |
| 1 | D_lstm | all-five-complete | 30678 | 57 | 0.8776 | 0.0980 | 0.9167 | 0.9523 |
| 4 | C_lstm | all-five-complete | 30678 | 346 | 0.8803 | 0.1317 | 0.9167 | 0.9523 |
| 4 | D_lstm | all-five-complete | 30678 | 346 | 0.7558 | 0.0451 | 0.9167 | 0.9523 |

