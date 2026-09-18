# Interpretability summary

SHAP model: **transformer** (best mean test PR-AUC at h=4). Estimator: shap.GradientExplainer (expected gradients), 200 background windows, 1560 explained windows per horizon.

Masked cells — imputed after forward fill, or structurally undefined — are
excluded from every average in this file.

## Horizon h = 1

### Which quarters attention points at

- **cnn_lstm_attn** on true positives: heaviest on t-1 (0.2383), t-0 (0.2383), t-3 (0.1458) (uniform would be 0.1250).
- **transformer** on true positives: heaviest on t-0 (0.1742), t-1 (0.1279), t-4 (0.1176) (uniform would be 0.1250).

### Which quarters SHAP points at

- t-0 (0.0457), t-1 (0.0171), t-3 (0.0109) carry the most |SHAP|.

### Agreement between the two methods

| attention_model | group | spearman_rho | spearman_p | kendall_tau | attention_argmax | shap_argmax | agree_on_top_quarter |
|---|---|---|---|---|---|---|---|
| transformer | true_positive | 0.4048 | 0.3199 | 0.3571 | t-0 | t-0 | True |
| transformer | all_positive | 0.2857 | 0.4927 | 0.1429 | t-0 | t-0 | True |
| cnn_lstm_attn | true_positive | 0.9271 | 0.0009 | 0.8487 | t-1 | t-0 | False |
| cnn_lstm_attn | all_positive | 0.9271 | 0.0009 | 0.8487 | t-1 | t-0 | False |


### Top-10 features by mean |SHAP|

| rank | feature | family | mean_abs_shap | is_altman | observed_rate |
|---|---|---|---|---|---|
| 1 | r15_ltd_to_ta | Leverage | 0.0432 | False | 0.6742 |
| 2 | r22_net_income_growth | Growth | 0.0429 | False | 0.9833 |
| 3 | r24_equity_growth | Growth | 0.0413 | False | 0.9808 |
| 4 | r01_current_ratio | Liquidity | 0.0344 | False | 0.9993 |
| 5 | r16_asset_turnover | Efficiency | 0.0291 | True | 0.9974 |
| 6 | r02_quick_ratio | Liquidity | 0.0264 | False | 0.7688 |
| 7 | r29_negative_equity_flag | Distress flag | 0.0242 | False | 0.9955 |
| 8 | r11_debt_to_equity | Leverage | 0.0158 | False | 0.6220 |
| 9 | r14_equity_to_liabilities | Leverage | 0.0141 | True | 0.9954 |
| 10 | r25_ocf_to_cl | Cash flow | 0.0135 | False | 0.9994 |


- Altman's five rank [5, 9, 23, 28, 29] of 29.
- `r29_negative_equity_flag` ranks 7 of 29.

## Horizon h = 4

### Which quarters attention points at

- **cnn_lstm_attn** on true positives: heaviest on t-1 (0.2536), t-0 (0.2536), t-3 (0.1424) (uniform would be 0.1250).
- **transformer** on true positives: heaviest on t-0 (0.2302), t-1 (0.1608), t-2 (0.1198) (uniform would be 0.1250).

### Which quarters SHAP points at

- t-0 (0.0857), t-1 (0.0488), t-7 (0.0343) carry the most |SHAP|.

### Agreement between the two methods

| attention_model | group | spearman_rho | spearman_p | kendall_tau | attention_argmax | shap_argmax | agree_on_top_quarter |
|---|---|---|---|---|---|---|---|
| transformer | true_positive | 0.9524 | 0.0003 | 0.8571 | t-0 | t-0 | True |
| transformer | all_positive | 0.9286 | 0.0009 | 0.8571 | t-0 | t-0 | True |
| cnn_lstm_attn | true_positive | 0.4880 | 0.2199 | 0.3858 | t-1 | t-0 | False |
| cnn_lstm_attn | all_positive | 0.5092 | 0.1975 | 0.4158 | t-1 | t-0 | False |


### Top-10 features by mean |SHAP|

| rank | feature | family | mean_abs_shap | is_altman | observed_rate |
|---|---|---|---|---|---|
| 1 | r25_ocf_to_cl | Cash flow | 0.1204 | False | 0.9983 |
| 2 | r01_current_ratio | Liquidity | 0.0851 | False | 0.9986 |
| 3 | r20_cash_conversion_cycle | Efficiency | 0.0621 | False | 0.5613 |
| 4 | r19_payables_turnover | Efficiency | 0.0534 | False | 0.7266 |
| 5 | r16_asset_turnover | Efficiency | 0.0494 | True | 0.9979 |
| 6 | r24_equity_growth | Growth | 0.0481 | False | 0.9808 |
| 7 | r15_ltd_to_ta | Leverage | 0.0473 | False | 0.6856 |
| 8 | r22_net_income_growth | Growth | 0.0471 | False | 0.9815 |
| 9 | r26_fcf_to_ta | Cash flow | 0.0462 | False | 0.9125 |
| 10 | r14_equity_to_liabilities | Leverage | 0.0423 | True | 0.9955 |


- Altman's five rank [5, 10, 22, 23, 29] of 29.
- `r29_negative_equity_flag` ranks 11 of 29.

## Family importance

| horizon | family | n_features | mean_abs_shap | share |
|---|---|---|---|---|
| 1 | Distress flag | 1 | 0.0242 | 0.0614 |
| 1 | Growth | 4 | 0.0233 | 0.2360 |
| 1 | Liquidity | 4 | 0.0180 | 0.1825 |
| 1 | Leverage | 5 | 0.0167 | 0.2121 |
| 1 | Efficiency | 5 | 0.0118 | 0.1497 |
| 1 | Cash flow | 4 | 0.0105 | 0.1063 |
| 1 | Profitability | 6 | 0.0034 | 0.0520 |
| 4 | Cash flow | 4 | 0.0552 | 0.2231 |
| 4 | Distress flag | 1 | 0.0422 | 0.0427 |
| 4 | Liquidity | 4 | 0.0411 | 0.1659 |
| 4 | Efficiency | 5 | 0.0374 | 0.1890 |
| 4 | Leverage | 5 | 0.0352 | 0.1777 |
| 4 | Growth | 4 | 0.0315 | 0.1272 |
| 4 | Profitability | 6 | 0.0123 | 0.0745 |

