| horizon | attention_model | shap_model | group | n_steps | spearman_rho | spearman_p | kendall_tau | kendall_p | attention_argmax | shap_argmax | agree_on_top_quarter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | transformer | transformer | true_positive | 8 | 0.7619 | 0.0280 | 0.6429 | 0.0312 | t-0 | t-0 | True |
| 1 | transformer | transformer | all_positive | 8 | 0.7619 | 0.0280 | 0.5714 | 0.0610 | t-0 | t-0 | True |
| 1 | cnn_lstm_attn | transformer | true_positive | 8 | 0.4880 | 0.2199 | 0.3858 | 0.2016 | t-1 | t-0 | False |
| 1 | cnn_lstm_attn | transformer | all_positive | 8 | 0.4880 | 0.2199 | 0.3858 | 0.2016 | t-1 | t-0 | False |
| 4 | transformer | transformer | true_positive | 8 | 0.8810 | 0.0039 | 0.7143 | 0.0141 | t-0 | t-0 | True |
| 4 | transformer | transformer | all_positive | 8 | 0.7619 | 0.0280 | 0.5714 | 0.0610 | t-0 | t-0 | True |
| 4 | cnn_lstm_attn | transformer | true_positive | 8 | 0.6831 | 0.0618 | 0.6172 | 0.0411 | t-1 | t-0 | False |
| 4 | cnn_lstm_attn | transformer | all_positive | 8 | 0.7395 | 0.0360 | 0.6425 | 0.0313 | t-1 | t-0 | False |
