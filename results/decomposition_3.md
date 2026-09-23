| model | horizon | roc_auc | pr_auc | threshold | precision | recall | specificity | f1 | step | delong_roc_diff | delong_ci_low | delong_ci_high | delong_p | pr_auc_diff | pr_ci_low | pr_ci_high | pr_boot_p | mcnemar_b | mcnemar_c | mcnemar_p | mcnemar_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_zdp | 3 | 0.8224 | 0.0233 | 2.0156 | 0.0249 | 0.8192 | 0.7386 | 0.0483 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B_levels | 3 | 0.7914 | 0.0246 | 0.6218 | 0.0313 | 0.5038 | 0.8733 | 0.0590 | A_zdp -> B_levels | -0.0311 | -0.0561 | -0.0061 | 0.0147 | 0.0013 | -0.0014 | 0.0045 | 0.3240 | 1056.0000 | 5276.0000 | 0.0000 | chi2 (continuity corrected) |
| C_lstm | 3 | 0.8918 | 0.1288 | 0.8475 | 0.3023 | 0.1000 | 0.9981 | 0.1503 | B_levels -> C_lstm | 0.1004 | 0.0805 | 0.1203 | 0.0000 | 0.1041 | 0.0714 | 0.1447 | 0.0000 | 124.0000 | 4009.0000 | 0.0000 | chi2 (continuity corrected) |
| D_lstm | 3 | 0.7757 | 0.0384 | 0.6003 | 0.1429 | 0.0192 | 0.9991 | 0.0339 | C_lstm -> D_lstm | -0.1161 | -0.1408 | -0.0914 | 0.0000 | -0.0904 | -0.1264 | -0.0584 | 0.0000 | 51.0000 | 60.0000 | 0.4477 | chi2 (continuity corrected) |
