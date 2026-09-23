| model | horizon | roc_auc | pr_auc | threshold | precision | recall | specificity | f1 | step | delong_roc_diff | delong_ci_low | delong_ci_high | delong_p | pr_auc_diff | pr_ci_low | pr_ci_high | pr_boot_p | mcnemar_b | mcnemar_c | mcnemar_p | mcnemar_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp | 2 | 0.8299 | 0.0152 | 2.0156 | 0.0154 | 0.8354 | 0.7369 | 0.0303 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B_levels | 2 | 0.8048 | 0.0164 | 0.6168 | 0.0207 | 0.5506 | 0.8719 | 0.0400 | A_zdp -> B_levels | -0.0250 | -0.0553 | 0.0052 | 0.1047 | 0.0012 | -0.0010 | 0.0038 | 0.2750 | 1020.0000 | 5300.0000 | 0.0000 | chi2 (continuity corrected) |
| C_lstm | 2 | 0.8820 | 0.1271 | 0.6243 | 0.1674 | 0.2278 | 0.9944 | 0.1930 | B_levels -> C_lstm | 0.0771 | 0.0520 | 0.1022 | 0.0000 | 0.1107 | 0.0672 | 0.1693 | 0.0000 | 106.0000 | 3984.0000 | 0.0000 | chi2 (continuity corrected) |
| D_lstm | 2 | 0.8663 | 0.0683 | 0.6412 | 0.2083 | 0.0633 | 0.9988 | 0.0971 | C_lstm -> D_lstm | -0.0156 | -0.0371 | 0.0058 | 0.1524 | -0.0588 | -0.1094 | -0.0142 | 0.0090 | 59.0000 | 174.0000 | 0.0000 | chi2 (continuity corrected) |
