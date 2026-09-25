| horizon | model | group_kind | group | n | n_positive | base_rate | n_firms | pr_auc | roc_auc | pr_auc_cluster_ci_low | pr_auc_cluster_ci_high | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | altman_zdp | size tercile | small | 9370 | 26 | 0.0028 | 1262 | 0.0062 | 0.7631 | 0.0039 | 0.0112 | reported |
| 1 | altman_zdp | size tercile | mid | 10548 | 22 | 0.0021 | 1234 | 0.0201 | 0.8921 | 0.0087 | 0.1009 | reported |
| 1 | altman_zdp | size tercile | large | 12298 | 12 | 0.0010 | 1202 |  |  |  |  | too few (12 test positives, need 20) |
| 1 | xgboost | size tercile | small | 9370 | 26 | 0.0028 | 1262 | 0.0282 | 0.7597 | 0.0074 | 0.1136 | reported |
| 1 | xgboost | size tercile | mid | 10548 | 22 | 0.0021 | 1234 | 0.3077 | 0.8876 | 0.1301 | 0.5037 | reported |
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
| 2 | xgboost | size tercile | small | 9370 | 73 | 0.0078 | 1262 | 0.0345 | 0.8032 | 0.0205 | 0.0658 | reported |
| 2 | xgboost | size tercile | mid | 10548 | 58 | 0.0055 | 1234 | 0.2014 | 0.9265 | 0.0979 | 0.3202 | reported |
| 2 | xgboost | size tercile | large | 12298 | 27 | 0.0022 | 1202 | 0.2042 | 0.9550 | 0.1044 | 0.3556 | reported |
| 2 | transformer | size tercile | small | 9370 | 73 | 0.0078 | 1262 | 0.0308 | 0.7742 | 0.0148 | 0.0973 | reported |
| 2 | transformer | size tercile | mid | 10548 | 58 | 0.0055 | 1234 | 0.1679 | 0.8697 | 0.0724 | 0.2894 | reported |
| 2 | transformer | size tercile | large | 12298 | 27 | 0.0022 | 1202 | 0.1912 | 0.8993 | 0.0562 | 0.4230 | reported |
| 2 | C_lstm | size tercile | small | 9370 | 73 | 0.0078 | 1262 | 0.0554 | 0.8583 | 0.0313 | 0.1106 | reported |
| 2 | C_lstm | size tercile | mid | 10548 | 58 | 0.0055 | 1234 | 0.2273 | 0.8823 | 0.1176 | 0.3484 | reported |
| 2 | C_lstm | size tercile | large | 12298 | 27 | 0.0022 | 1202 | 0.2277 | 0.9010 | 0.0896 | 0.4312 | reported |
| 3 | altman_zdp | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0204 | 0.6987 | 0.0148 | 0.0277 | reported |
| 3 | altman_zdp | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.0591 | 0.8757 | 0.0350 | 0.1402 | reported |
| 3 | altman_zdp | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.0630 | 0.8579 | 0.0274 | 0.1880 | reported |
| 3 | xgboost | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0556 | 0.8200 | 0.0338 | 0.0997 | reported |
| 3 | xgboost | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.1902 | 0.9134 | 0.1003 | 0.3026 | reported |
| 3 | xgboost | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.2745 | 0.9429 | 0.1078 | 0.4425 | reported |
| 3 | transformer | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0409 | 0.7751 | 0.0252 | 0.0912 | reported |
| 3 | transformer | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.1426 | 0.8819 | 0.0736 | 0.2531 | reported |
| 3 | transformer | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.1892 | 0.8974 | 0.0503 | 0.4152 | reported |
| 3 | C_lstm | size tercile | small | 9370 | 121 | 0.0129 | 1262 | 0.0623 | 0.8375 | 0.0392 | 0.1119 | reported |
| 3 | C_lstm | size tercile | mid | 10548 | 98 | 0.0093 | 1234 | 0.1919 | 0.8980 | 0.1121 | 0.2869 | reported |
| 3 | C_lstm | size tercile | large | 12298 | 41 | 0.0033 | 1202 | 0.3077 | 0.9087 | 0.1278 | 0.5116 | reported |
| 4 | altman_zdp | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0265 | 0.6821 | 0.0194 | 0.0356 | reported |
| 4 | altman_zdp | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.0709 | 0.8609 | 0.0440 | 0.1471 | reported |
| 4 | altman_zdp | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.0595 | 0.8497 | 0.0274 | 0.1575 | reported |
| 4 | xgboost | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0656 | 0.8071 | 0.0425 | 0.1035 | reported |
| 4 | xgboost | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.1922 | 0.9019 | 0.1114 | 0.2963 | reported |
| 4 | xgboost | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.3091 | 0.9469 | 0.1419 | 0.4882 | reported |
| 4 | transformer | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0427 | 0.7542 | 0.0292 | 0.0759 | reported |
| 4 | transformer | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.1012 | 0.8773 | 0.0593 | 0.1729 | reported |
| 4 | transformer | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.1229 | 0.8901 | 0.0366 | 0.2892 | reported |
| 4 | C_lstm | size tercile | small | 9370 | 168 | 0.0179 | 1262 | 0.0689 | 0.8228 | 0.0481 | 0.1064 | reported |
| 4 | C_lstm | size tercile | mid | 10548 | 141 | 0.0134 | 1234 | 0.1803 | 0.8730 | 0.1119 | 0.2670 | reported |
| 4 | C_lstm | size tercile | large | 12298 | 54 | 0.0044 | 1202 | 0.2423 | 0.9093 | 0.1024 | 0.4094 | reported |
