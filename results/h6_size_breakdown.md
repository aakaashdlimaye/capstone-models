| horizon | model | group_kind | group | n | n_positive | base_rate | n_firms | pr_auc | roc_auc | pr_auc_cluster_ci_low | pr_auc_cluster_ci_high | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | altman_zdp | size tercile | small | 9370 | 26 | 0.0028 | 1262 | 0.0062 | 0.7631 | 0.0039 | 0.0112 | reported |
| 1 | altman_zdp | size tercile | mid | 10548 | 22 | 0.0021 | 1234 | 0.0201 | 0.8921 | 0.0087 | 0.1009 | reported |
| 1 | altman_zdp | size tercile | large | 12298 | 12 | 0.0010 | 1202 |  |  |  |  | too few (12 test positives, need 20) |
| 1 | xgboost | size tercile | small | 9370 | 26 | 0.0028 | 1262 | 0.0370 | 0.7916 | 0.0076 | 0.1663 | reported |
| 1 | xgboost | size tercile | mid | 10548 | 22 | 0.0021 | 1234 | 0.2842 | 0.8901 | 0.1094 | 0.4779 | reported |
| 1 | xgboost | size tercile | large | 12298 | 12 | 0.0010 | 1202 |  |  |  |  | too few (12 test positives, need 20) |
| 1 | transformer | size tercile | small | 9370 | 26 | 0.0028 | 1262 | 0.0184 | 0.7678 | 0.0066 | 0.0634 | reported |
| 1 | transformer | size tercile | mid | 10548 | 22 | 0.0021 | 1234 | 0.1894 | 0.8496 | 0.0574 | 0.3573 | reported |
| 1 | transformer | size tercile | large | 12298 | 12 | 0.0010 | 1202 |  |  |  |  | too few (12 test positives, need 20) |
| 1 | C_lstm | size tercile | small | 9370 | 26 | 0.0028 | 1262 | 0.0304 | 0.8343 | 0.0110 | 0.0953 | reported |
| 1 | C_lstm | size tercile | mid | 10548 | 22 | 0.0021 | 1234 | 0.3292 | 0.9029 | 0.1477 | 0.5378 | reported |
| 1 | C_lstm | size tercile | large | 12298 | 12 | 0.0010 | 1202 |  |  |  |  | too few (12 test positives, need 20) |
| 2 | altman_zdp | size tercile | small | 9370 | 73 | 0.0078 | 1262 | 0.0135 | 0.7167 | 0.0095 | 0.0193 | reported |
| 2 | altman_zdp | size tercile | mid | 10548 | 58 | 0.0055 | 1234 | 0.0420 | 0.8856 | 0.0232 | 0.1271 | reported |
| 2 | altman_zdp | size tercile | large | 12298 | 27 | 0.0022 | 1202 | 0.0608 | 0.8709 | 0.0252 | 0.2252 | reported |
| 2 | xgboost | size tercile | small | 9370 | 73 | 0.0078 | 1262 | 0.0386 | 0.8055 | 0.0217 | 0.0902 | reported |
| 2 | xgboost | size tercile | mid | 10548 | 58 | 0.0055 | 1234 | 0.2055 | 0.9262 | 0.1005 | 0.3268 | reported |
| 2 | xgboost | size tercile | large | 12298 | 27 | 0.0022 | 1202 | 0.2094 | 0.9551 | 0.1021 | 0.3834 | reported |
| 2 | transformer | size tercile | small | 9370 | 73 | 0.0078 | 1262 | 0.0308 | 0.7742 | 0.0148 | 0.0973 | reported |
| 2 | transformer | size tercile | mid | 10548 | 58 | 0.0055 | 1234 | 0.1679 | 0.8697 | 0.0724 | 0.2894 | reported |
| 2 | transformer | size tercile | large | 12298 | 27 | 0.0022 | 1202 | 0.1912 | 0.8993 | 0.0562 | 0.4230 | reported |
| 2 | C_lstm | size tercile | small | 9370 | 73 | 0.0078 | 1262 | 0.0554 | 0.8583 | 0.0313 | 0.1106 | reported |
| 2 | C_lstm | size tercile | mid | 10548 | 58 | 0.0055 | 1234 | 0.2273 | 0.8823 | 0.1176 | 0.3484 | reported |
| 2 | C_lstm | size tercile | large | 12298 | 27 | 0.0022 | 1202 | 0.2277 | 0.9010 | 0.0896 | 0.4312 | reported |
| 3 | altman_zdp | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0204 | 0.6987 | 0.0148 | 0.0277 | reported |
| 3 | altman_zdp | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.0591 | 0.8757 | 0.0350 | 0.1402 | reported |
| 3 | altman_zdp | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.0630 | 0.8579 | 0.0274 | 0.1880 | reported |
| 3 | xgboost | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0537 | 0.8254 | 0.0342 | 0.0914 | reported |
| 3 | xgboost | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.1985 | 0.9161 | 0.1078 | 0.3075 | reported |
| 3 | xgboost | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.2758 | 0.9453 | 0.1222 | 0.4418 | reported |
| 3 | transformer | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0409 | 0.7751 | 0.0252 | 0.0912 | reported |
| 3 | transformer | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.1426 | 0.8819 | 0.0736 | 0.2531 | reported |
| 3 | transformer | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.1892 | 0.8974 | 0.0503 | 0.4152 | reported |
| 3 | C_lstm | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0623 | 0.8375 | 0.0392 | 0.1119 | reported |
| 3 | C_lstm | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.1919 | 0.8980 | 0.1121 | 0.2869 | reported |
| 3 | C_lstm | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.3077 | 0.9087 | 0.1278 | 0.5116 | reported |
| 4 | altman_zdp | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0265 | 0.6821 | 0.0194 | 0.0356 | reported |
| 4 | altman_zdp | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.0709 | 0.8609 | 0.0440 | 0.1471 | reported |
| 4 | altman_zdp | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.0595 | 0.8497 | 0.0274 | 0.1575 | reported |
| 4 | xgboost | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0618 | 0.8074 | 0.0420 | 0.0956 | reported |
| 4 | xgboost | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.1984 | 0.9068 | 0.1144 | 0.2994 | reported |
| 4 | xgboost | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.3194 | 0.9463 | 0.1481 | 0.5023 | reported |
| 4 | transformer | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0427 | 0.7542 | 0.0292 | 0.0759 | reported |
| 4 | transformer | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.1012 | 0.8773 | 0.0593 | 0.1729 | reported |
| 4 | transformer | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.1229 | 0.8901 | 0.0366 | 0.2892 | reported |
| 4 | C_lstm | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0689 | 0.8228 | 0.0481 | 0.1064 | reported |
| 4 | C_lstm | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.1803 | 0.8730 | 0.1119 | 0.2670 | reported |
| 4 | C_lstm | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.2423 | 0.9093 | 0.1024 | 0.4094 | reported |
