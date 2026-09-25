| horizon | step | cause | statement |
|---|---|---|---|
| 1 | A_zdp->B_levels | coefficient drift | A_zdp->B_levels: PR-AUC +0.0015 [-0.0003, +0.0037], ROC-AUC -0.0166 [-0.0660, +0.0242] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 1 | B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | B_levels->B_tensor: PR-AUC +0.0001 [-0.0032, +0.0023], ROC-AUC -0.0225 [-0.0603, +0.0075] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 1 | B_tensor->B_mlp_t0 | nonlinearity | B_tensor->B_mlp_t0: PR-AUC +0.0246 [+0.0078, +0.0759], ROC-AUC +0.0283 [-0.0065, +0.0598] — PR-AUC interval excludes zero; ROC-AUC interval contains zero. |
| 1 | B_mlp_t0->C_lstm | the time axis | B_mlp_t0->C_lstm: PR-AUC +0.0749 [+0.0144, +0.1486], ROC-AUC +0.0399 [-0.0011, +0.0843] — PR-AUC interval excludes zero; ROC-AUC interval contains zero. |
| 1 | C_lstm->D_lstm | the feature-set expansion | C_lstm->D_lstm: PR-AUC -0.0230 [-0.1012, +0.0523], ROC-AUC -0.0116 [-0.0488, +0.0229] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 1 | D_lstm->D_transformer | architecture | D_lstm->D_transformer: PR-AUC +0.0164 [-0.0212, +0.0532], ROC-AUC -0.0249 [-0.0692, +0.0173] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 2 | A_zdp->B_levels | coefficient drift | A_zdp->B_levels: PR-AUC +0.0012 [-0.0019, +0.0045], ROC-AUC -0.0250 [-0.0656, +0.0096] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 2 | B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | B_levels->B_tensor: PR-AUC +0.0010 [-0.0008, +0.0031], ROC-AUC -0.0016 [-0.0210, +0.0137] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 2 | B_tensor->B_mlp_t0 | nonlinearity | B_tensor->B_mlp_t0: PR-AUC +0.0384 [+0.0173, +0.0854], ROC-AUC +0.0830 [+0.0569, +0.1129] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
| 2 | B_mlp_t0->C_lstm | the time axis | B_mlp_t0->C_lstm: PR-AUC +0.0713 [+0.0268, +0.1135], ROC-AUC -0.0043 [-0.0256, +0.0155] — PR-AUC interval excludes zero; ROC-AUC interval contains zero. |
| 2 | C_lstm->D_lstm | the feature-set expansion | C_lstm->D_lstm: PR-AUC -0.0588 [-0.1186, -0.0038], ROC-AUC -0.0156 [-0.0421, +0.0104] — PR-AUC interval excludes zero; ROC-AUC interval contains zero. |
| 2 | D_lstm->D_transformer | architecture | D_lstm->D_transformer: PR-AUC +0.0313 [-0.0077, +0.0818], ROC-AUC -0.0058 [-0.0332, +0.0223] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 3 | A_zdp->B_levels | coefficient drift | A_zdp->B_levels: PR-AUC +0.0013 [-0.0030, +0.0064], ROC-AUC -0.0311 [-0.0691, +0.0039] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 3 | B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | B_levels->B_tensor: PR-AUC +0.0020 [-0.0006, +0.0054], ROC-AUC -0.0010 [-0.0186, +0.0141] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 3 | B_tensor->B_mlp_t0 | nonlinearity | B_tensor->B_mlp_t0: PR-AUC +0.0389 [+0.0195, +0.0713], ROC-AUC +0.0857 [+0.0657, +0.1088] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
| 3 | B_mlp_t0->C_lstm | the time axis | B_mlp_t0->C_lstm: PR-AUC +0.0633 [+0.0277, +0.1003], ROC-AUC +0.0157 [+0.0024, +0.0277] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
| 3 | C_lstm->D_lstm | the feature-set expansion | C_lstm->D_lstm: PR-AUC -0.0904 [-0.1324, -0.0513], ROC-AUC -0.1161 [-0.1523, -0.0796] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
| 3 | D_lstm->D_transformer | architecture | D_lstm->D_transformer: PR-AUC +0.0517 [+0.0226, +0.0954], ROC-AUC +0.0925 [+0.0604, +0.1264] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
| 4 | A_zdp->B_levels | coefficient drift | A_zdp->B_levels: PR-AUC +0.0010 [-0.0044, +0.0080], ROC-AUC -0.0376 [-0.0776, -0.0010] — PR-AUC interval contains zero; ROC-AUC interval excludes zero. |
| 4 | B_levels->B_tensor | preprocessing (annualised levels to z-scored quarterly ratios) | B_levels->B_tensor: PR-AUC +0.0028 [-0.0009, +0.0083], ROC-AUC -0.0015 [-0.0192, +0.0142] — PR-AUC interval contains zero; ROC-AUC interval contains zero. |
| 4 | B_tensor->B_mlp_t0 | nonlinearity | B_tensor->B_mlp_t0: PR-AUC +0.0399 [+0.0204, +0.0665], ROC-AUC +0.0967 [+0.0757, +0.1204] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
| 4 | B_mlp_t0->C_lstm | the time axis | B_mlp_t0->C_lstm: PR-AUC +0.0488 [+0.0194, +0.0829], ROC-AUC +0.0068 [-0.0062, +0.0190] — PR-AUC interval excludes zero; ROC-AUC interval contains zero. |
| 4 | C_lstm->D_lstm | the feature-set expansion | C_lstm->D_lstm: PR-AUC -0.0764 [-0.1165, -0.0412], ROC-AUC -0.1082 [-0.1419, -0.0731] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
| 4 | D_lstm->D_transformer | architecture | D_lstm->D_transformer: PR-AUC +0.0264 [+0.0039, +0.0559], ROC-AUC +0.0825 [+0.0539, +0.1112] — PR-AUC interval excludes zero; ROC-AUC interval excludes zero. |
