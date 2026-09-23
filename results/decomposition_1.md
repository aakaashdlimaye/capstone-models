| model | horizon | roc_auc | pr_auc | threshold | precision | recall | specificity | f1 | step | delong_roc_diff | delong_ci_low | delong_ci_high | delong_p | pr_auc_diff | pr_ci_low | pr_ci_high | pr_boot_p | mcnemar_b | mcnemar_c | mcnemar_p | mcnemar_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp | 1 | 0.8450 | 0.0066 | 2.0156 | 0.0061 | 0.8667 | 0.7353 | 0.0121 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B_levels | 1 | 0.8284 | 0.0081 | 0.5997 | 0.0094 | 0.6500 | 0.8717 | 0.0185 | A_zdp -> B_levels | -0.0166 | -0.0608 | 0.0277 | 0.4622 | 0.0015 | -0.0001 | 0.0036 | 0.0700 | 984.0000 | 5357.0000 | 0.0000 | chi2 (continuity corrected) |
| C_lstm | 1 | 0.8741 | 0.1077 | 0.7694 | 0.2927 | 0.2000 | 0.9991 | 0.2376 | B_levels -> C_lstm | 0.0456 | 0.0033 | 0.0880 | 0.0346 | 0.0996 | 0.0419 | 0.1931 | 0.0000 | 33.0000 | 4104.0000 | 0.0000 | chi2 (continuity corrected) |
| D_lstm | 1 | 0.8625 | 0.0847 | 0.5960 | 0.2162 | 0.1333 | 0.9991 | 0.1649 | C_lstm -> D_lstm | -0.0116 | -0.0474 | 0.0242 | 0.5261 | -0.0230 | -0.1053 | 0.0521 | 0.5630 | 32.0000 | 28.0000 | 0.6985 | chi2 (continuity corrected) |
