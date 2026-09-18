| model | horizon | roc_auc | pr_auc | threshold | precision | recall | specificity | f1 | step | delong_roc_diff | delong_ci_low | delong_ci_high | delong_p | pr_auc_diff | pr_ci_low | pr_ci_high | pr_boot_p | mcnemar_b | mcnemar_c | mcnemar_p | mcnemar_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp | 4 | 0.8122 | 0.0305 | 2.0156 | 0.0334 | 0.7879 | 0.7401 | 0.0641 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B_levels | 4 | 0.7746 | 0.0315 | 0.6228 | 0.0398 | 0.4601 | 0.8734 | 0.0732 | A_zdp -> B_levels | -0.0376 | -0.0603 | -0.0150 | 0.0011 | 0.0010 | -0.0021 | 0.0047 | 0.5040 | 1119.0000 | 5245.0000 | 0.0000 | chi2 (continuity corrected) |
| C_lstm | 4 | 0.8766 | 0.1230 | 0.8690 | 0.2431 | 0.0964 | 0.9966 | 0.1381 | B_levels -> C_lstm | 0.1020 | 0.0854 | 0.1186 | 0.0000 | 0.0915 | 0.0657 | 0.1254 | 0.0000 | 169.0000 | 3962.0000 | 0.0000 | chi2 (continuity corrected) |
| D_lstm | 4 | 0.7684 | 0.0466 | 0.5307 | 0.1548 | 0.0358 | 0.9978 | 0.0582 | C_lstm -> D_lstm | -0.1082 | -0.1302 | -0.0863 | 0.0000 | -0.0764 | -0.1088 | -0.0504 | 0.0000 | 92.0000 | 108.0000 | 0.2888 | chi2 (continuity corrected) |
