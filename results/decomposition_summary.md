# Decomposition summary

Four models on the identical row set.  **A** applies Altman's published Z″
coefficients to the window's end quarter; **B** re-estimates those coefficients
on train; **C** gives the same variables an 8-quarter LSTM; **D** gives the LSTM
all 29 ratios.  A→B isolates coefficient drift, B→C the static formulation,
C→D the feature set.

Shares are of the **total absolute movement** across the three steps, not of the
net A→D gap.  One step moves backwards, and dividing by a small net total would
report an ordinary step as several hundred percent.  Each step's signed change
and its 95% interval sit beside its share.

## Horizon h = 1

> At h=1, test PR-AUC moves from 0.0066 for Altman Z″ to 0.0847 for the full temporal model, a net +0.0781.  Of the 0.1241 of total movement across the three steps, 1% is coefficient drift (+0.0015, improves), 80% is the static formulation (+0.0996, improves), 19% is the feature-set expansion (-0.0230, degrades).  Re-estimating Altman's coefficients on modern training data is worth +0.0015 PR-AUC, 95% CI [-0.0001, +0.0036].

| step | cause | PR-AUC change | PR-AUC 95% CI | share of movement | direction | ROC-AUC change | DeLong 95% CI | DeLong p | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0015 | [-0.0001, +0.0036] | 0.0122 | improves | -0.0166 | [-0.0608, +0.0277] | 0.4622 | 0.0000 |
| B_levels->C_lstm | the static formulation | 0.0996 | [+0.0419, +0.1931] | 0.8024 | improves | 0.0456 | [+0.0033, +0.0880] | 0.0346 | 0.0000 |
| C_lstm->D_lstm | the feature-set expansion | -0.0230 | [-0.1053, +0.0521] | 0.1855 | degrades | -0.0116 | [-0.0474, +0.0242] | 0.5261 | 0.6985 |


## Horizon h = 2

> At h=2, test PR-AUC moves from 0.0152 for Altman Z″ to 0.0683 for the full temporal model, a net +0.0531.  Of the 0.1707 of total movement across the three steps, 1% is coefficient drift (+0.0012, improves), 65% is the static formulation (+0.1107, improves), 34% is the feature-set expansion (-0.0588, degrades).  Re-estimating Altman's coefficients on modern training data is worth +0.0012 PR-AUC, 95% CI [-0.0010, +0.0038].

| step | cause | PR-AUC change | PR-AUC 95% CI | share of movement | direction | ROC-AUC change | DeLong 95% CI | DeLong p | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0012 | [-0.0010, +0.0038] | 0.0071 | improves | -0.0250 | [-0.0553, +0.0052] | 0.1047 | 0.0000 |
| B_levels->C_lstm | the static formulation | 0.1107 | [+0.0672, +0.1693] | 0.6484 | improves | 0.0771 | [+0.0520, +0.1022] | 0.0000 | 0.0000 |
| C_lstm->D_lstm | the feature-set expansion | -0.0588 | [-0.1094, -0.0142] | 0.3445 | degrades | -0.0156 | [-0.0371, +0.0058] | 0.1524 | 0.0000 |


## Horizon h = 3

> At h=3, test PR-AUC moves from 0.0233 for Altman Z″ to 0.0384 for the full temporal model, a net +0.0150.  Of the 0.1958 of total movement across the three steps, 1% is coefficient drift (+0.0013, improves), 53% is the static formulation (+0.1041, improves), 46% is the feature-set expansion (-0.0904, degrades).  Re-estimating Altman's coefficients on modern training data is worth +0.0013 PR-AUC, 95% CI [-0.0014, +0.0045].

| step | cause | PR-AUC change | PR-AUC 95% CI | share of movement | direction | ROC-AUC change | DeLong 95% CI | DeLong p | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0013 | [-0.0014, +0.0045] | 0.0066 | improves | -0.0311 | [-0.0561, -0.0061] | 0.0147 | 0.0000 |
| B_levels->C_lstm | the static formulation | 0.1041 | [+0.0714, +0.1447] | 0.5317 | improves | 0.1004 | [+0.0805, +0.1203] | 0.0000 | 0.0000 |
| C_lstm->D_lstm | the feature-set expansion | -0.0904 | [-0.1264, -0.0584] | 0.4616 | degrades | -0.1161 | [-0.1408, -0.0914] | 0.0000 | 0.4477 |


## Horizon h = 4

> At h=4, test PR-AUC moves from 0.0305 for Altman Z″ to 0.0466 for the full temporal model, a net +0.0161.  Of the 0.1689 of total movement across the three steps, 1% is coefficient drift (+0.0010, improves), 54% is the static formulation (+0.0915, improves), 45% is the feature-set expansion (-0.0764, degrades).  Re-estimating Altman's coefficients on modern training data is worth +0.0010 PR-AUC, 95% CI [-0.0021, +0.0047].

| step | cause | PR-AUC change | PR-AUC 95% CI | share of movement | direction | ROC-AUC change | DeLong 95% CI | DeLong p | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| A_zdp->B_levels | coefficient drift | 0.0010 | [-0.0021, +0.0047] | 0.0062 | improves | -0.0376 | [-0.0603, -0.0150] | 0.0011 | 0.0000 |
| B_levels->C_lstm | the static formulation | 0.0915 | [+0.0657, +0.1254] | 0.5414 | improves | 0.1020 | [+0.0854, +0.1186] | 0.0000 | 0.0000 |
| C_lstm->D_lstm | the feature-set expansion | -0.0764 | [-0.1088, -0.0504] | 0.4524 | degrades | -0.1082 | [-0.1302, -0.0863] | 0.0000 | 0.2888 |


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

