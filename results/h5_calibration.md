| horizon | model | is_probability | scaling | base_rate | mean_predicted | brier | ece | n_bins_used | strategy |
|---|---|---|---|---|---|---|---|---|---|
| 1 | altman_zdp | False | Platt-scaled on validation | 0.00186 | 0.00183 | 0.00186 | 0.00228 | 10 | quantile |
| 1 | xgboost | True | native probability | 0.00186 | 0.00105 | 0.00172 | 0.00082 | 10 | quantile |
| 1 | transformer | True | native probability | 0.00186 | 0.00317 | 0.00193 | 0.00191 | 10 | quantile |
| 1 | C_lstm | True | native probability | 0.00186 | 0.00919 | 0.00359 | 0.00749 | 10 | quantile |
| 2 | altman_zdp | False | Platt-scaled on validation | 0.00490 | 0.00342 | 0.00488 | 0.00497 | 10 | quantile |
| 2 | xgboost | True | native probability | 0.00490 | 0.00270 | 0.00482 | 0.00221 | 10 | quantile |
| 2 | transformer | True | native probability | 0.00490 | 0.01395 | 0.00569 | 0.00906 | 10 | quantile |
| 2 | C_lstm | True | native probability | 0.00490 | 0.03847 | 0.01207 | 0.03356 | 10 | quantile |
| 3 | altman_zdp | False | Platt-scaled on validation | 0.00807 | 0.00448 | 0.00802 | 0.00779 | 10 | quantile |
| 3 | xgboost | True | native probability | 0.00807 | 0.00791 | 0.00836 | 0.00430 | 10 | quantile |
| 3 | transformer | True | native probability | 0.00807 | 0.02518 | 0.01097 | 0.01711 | 10 | quantile |
| 3 | C_lstm | True | native probability | 0.00807 | 0.04748 | 0.01765 | 0.03940 | 10 | quantile |
| 4 | altman_zdp | False | Platt-scaled on validation | 0.01127 | 0.00522 | 0.01118 | 0.01033 | 10 | quantile |
| 4 | xgboost | True | native probability | 0.01127 | 0.00908 | 0.01136 | 0.00541 | 10 | quantile |
| 4 | transformer | True | native probability | 0.01127 | 0.04445 | 0.01878 | 0.03318 | 10 | quantile |
| 4 | C_lstm | True | native probability | 0.01127 | 0.07732 | 0.03196 | 0.06605 | 10 | quantile |
