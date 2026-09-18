| horizon | attention_model | shap_model | group | n_steps | spearman_rho | spearman_p | kendall_tau | kendall_p | attention_argmax | shap_argmax | agree_on_top_quarter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | transformer | transformer | true_positive | 8 | 0.4048 | 0.3199 | 0.3571 | 0.2751 | t-0 | t-0 | True |
| 1 | transformer | transformer | all_positive | 8 | 0.2857 | 0.4927 | 0.1429 | 0.7195 | t-0 | t-0 | True |
| 1 | cnn_lstm_attn | transformer | true_positive | 8 | 0.9271 | 0.0009 | 0.8487 | 0.0050 | t-1 | t-0 | False |
| 1 | cnn_lstm_attn | transformer | all_positive | 8 | 0.9271 | 0.0009 | 0.8487 | 0.0050 | t-1 | t-0 | False |
| 4 | transformer | transformer | true_positive | 8 | 0.9524 | 0.0003 | 0.8571 | 0.0017 | t-0 | t-0 | True |
| 4 | transformer | transformer | all_positive | 8 | 0.9286 | 0.0009 | 0.8571 | 0.0017 | t-0 | t-0 | True |
| 4 | cnn_lstm_attn | transformer | true_positive | 8 | 0.4880 | 0.2199 | 0.3858 | 0.2016 | t-1 | t-0 | False |
| 4 | cnn_lstm_attn | transformer | all_positive | 8 | 0.5092 | 0.1975 | 0.4158 | 0.1635 | t-1 | t-0 | False |
