# Interpretability summary

SHAP model: **transformer** (best mean test PR-AUC at h=4). Estimator: shap.GradientExplainer (expected gradients), 200 background windows, 1560 explained windows per horizon.

Masked cells — imputed after forward fill, or structurally undefined — are
excluded from every average in this file.

## Horizon h = 1

### Which quarters attention points at

- **cnn_lstm_attn** on true positives: heaviest on t-1 (0.2383), t-0 (0.2383), t-3 (0.1458) (uniform would be 0.1250).
- **transformer** on true positives: heaviest on t-0 (0.1742), t-1 (0.1279), t-4 (0.1176) (uniform would be 0.1250).

### Which quarters SHAP points at

- t-0 (0.0300), t-1 (0.0108), t-7 (0.0059) carry the most |SHAP|.

### Agreement between the two methods

| attention_model | group | spearman_rho | spearman_p | kendall_tau | attention_argmax | shap_argmax | agree_on_top_quarter |
|---|---|---|---|---|---|---|---|
| transformer | true_positive | 0.7619 | 0.0280 | 0.6429 | t-0 | t-0 | True |
| transformer | all_positive | 0.7619 | 0.0280 | 0.5714 | t-0 | t-0 | True |
| cnn_lstm_attn | true_positive | 0.4880 | 0.2199 | 0.3858 | t-1 | t-0 | False |
| cnn_lstm_attn | all_positive | 0.4880 | 0.2199 | 0.3858 | t-1 | t-0 | False |


### Top-10 features by mean |SHAP|

| rank | feature | family | mean_abs_shap | is_altman | observed_rate |
|---|---|---|---|---|---|
| 1 | r22_net_income_growth | Growth | 0.0191 | False | 0.9833 |
| 2 | r29_negative_equity_flag | Distress flag | 0.0190 | False | 0.9955 |
| 3 | r24_equity_growth | Growth | 0.0186 | False | 0.9808 |
| 4 | r15_ltd_to_ta | Leverage | 0.0171 | False | 0.6742 |
| 5 | r27_accrual_quality | Cash flow | 0.0116 | False | 0.9997 |
| 6 | r16_asset_turnover | Efficiency | 0.0106 | True | 0.9974 |
| 7 | r12_debt_to_assets | Leverage | 0.0093 | False | 0.6260 |
| 8 | r01_current_ratio | Liquidity | 0.0085 | False | 0.9993 |
| 9 | r02_quick_ratio | Liquidity | 0.0074 | False | 0.7688 |
| 10 | r26_fcf_to_ta | Cash flow | 0.0070 | False | 0.9092 |


- Altman's five rank [6, 12, 19, 27, 28] of 29.
- `r29_negative_equity_flag` ranks 2 of 29.

## Horizon h = 4

### Which quarters attention points at

- **cnn_lstm_attn** on true positives: heaviest on t-1 (0.2536), t-0 (0.2536), t-3 (0.1424) (uniform would be 0.1250).
- **transformer** on true positives: heaviest on t-0 (0.2302), t-1 (0.1608), t-2 (0.1198) (uniform would be 0.1250).

### Which quarters SHAP points at

- t-0 (0.1220), t-1 (0.0849), t-7 (0.0619) carry the most |SHAP|.

### Agreement between the two methods

| attention_model | group | spearman_rho | spearman_p | kendall_tau | attention_argmax | shap_argmax | agree_on_top_quarter |
|---|---|---|---|---|---|---|---|
| transformer | true_positive | 0.8571 | 0.0065 | 0.7143 | t-0 | t-0 | True |
| transformer | all_positive | 0.6905 | 0.0580 | 0.5714 | t-0 | t-0 | True |
| cnn_lstm_attn | true_positive | 0.6831 | 0.0618 | 0.6172 | t-1 | t-0 | False |
| cnn_lstm_attn | all_positive | 0.7395 | 0.0360 | 0.6425 | t-1 | t-0 | False |


### Top-10 features by mean |SHAP|

| rank | feature | family | mean_abs_shap | is_altman | observed_rate |
|---|---|---|---|---|---|
| 1 | r25_ocf_to_cl | Cash flow | 0.1651 | False | 0.9983 |
| 2 | r01_current_ratio | Liquidity | 0.1039 | False | 0.9986 |
| 3 | r19_payables_turnover | Efficiency | 0.0994 | False | 0.7266 |
| 4 | r16_asset_turnover | Efficiency | 0.0985 | True | 0.9979 |
| 5 | r24_equity_growth | Growth | 0.0985 | False | 0.9808 |
| 6 | r22_net_income_growth | Growth | 0.0955 | False | 0.9815 |
| 7 | r02_quick_ratio | Liquidity | 0.0930 | False | 0.7566 |
| 8 | r14_equity_to_liabilities | Leverage | 0.0839 | True | 0.9955 |
| 9 | r29_negative_equity_flag | Distress flag | 0.0796 | False | 0.9955 |
| 10 | r03_cash_ratio | Liquidity | 0.0775 | False | 0.9232 |


- Altman's five rank [4, 8, 23, 25, 26] of 29.
- `r29_negative_equity_flag` ranks 9 of 29.

## Family importance

| horizon | family | n_features | mean_abs_shap | share |
|---|---|---|---|---|
| 1 | Distress flag | 1 | 0.0190 | 0.0984 |
| 1 | Growth | 4 | 0.0109 | 0.2256 |
| 1 | Leverage | 5 | 0.0080 | 0.2076 |
| 1 | Cash flow | 4 | 0.0073 | 0.1518 |
| 1 | Liquidity | 4 | 0.0057 | 0.1178 |
| 1 | Efficiency | 5 | 0.0052 | 0.1351 |
| 1 | Profitability | 6 | 0.0021 | 0.0637 |
| 4 | Cash flow | 4 | 0.0885 | 0.1956 |
| 4 | Distress flag | 1 | 0.0796 | 0.0440 |
| 4 | Efficiency | 5 | 0.0744 | 0.2054 |
| 4 | Liquidity | 4 | 0.0733 | 0.1619 |
| 4 | Growth | 4 | 0.0616 | 0.1361 |
| 4 | Leverage | 5 | 0.0599 | 0.1654 |
| 4 | Profitability | 6 | 0.0276 | 0.0915 |

