# Leakage audit — modelling stage

The dataset repository proved its own four checks in
`reports/leakage_audit_full.md`.  This file proves the four the modelling
stage is responsible for.  Every claim prints the output it rests on.

Dataset repo SHA `b16eb17`, models repo SHA `85c3df3`.

## (a) no window index appears in more than one split in any result file

```
row_sets.csv rows                        : 139,652
distinct (cik, end_quarter) window keys  : 139,652
window keys mapped to >1 split           : 0
split sizes                              : {'test': 32216, 'train': 86592, 'val': 20844}
prediction files checked                 : 758
files containing a row from another split: 0
```

**PASS**

## (b) train-split resampling and weighting statistics come from train rows only

```
train split size                         : 86,592
class-weight runs inspected              : 195
SMOTE runs inspected                     : 40
runs whose statistics do not come from train only : 0

example class-weight runs (n = train size, n_pos = train positives):
  {"key": "bilstm_1_class_weight_0", "n": 86592, "n_pos": 154, "pos_weight": 561.29, "matches_train": true}
  {"key": "bilstm_1_class_weight_1", "n": 86592, "n_pos": 154, "pos_weight": 561.29, "matches_train": true}
  {"key": "bilstm_1_class_weight_2", "n": 86592, "n_pos": 154, "pos_weight": 561.29, "matches_train": true}
  {"key": "bilstm_1_class_weight_3", "n": 86592, "n_pos": 154, "pos_weight": 561.29, "matches_train": true}

example SMOTE runs (n_in = train size before resampling):
  {"key": "bilstm_1_smote_0", "n_in": 86592, "n_pos_in": 154, "n_out": 172876, "fitted_on_train_rows_only": true}
  {"key": "bilstm_1_smote_1", "n_in": 86592, "n_pos_in": 154, "n_out": 172876, "fitted_on_train_rows_only": true}
  {"key": "bilstm_1_smote_2", "n_in": 86592, "n_pos_in": 154, "n_out": 172876, "fitted_on_train_rows_only": true}
  {"key": "bilstm_1_smote_3", "n_in": 86592, "n_pos_in": 154, "n_out": 172876, "fitted_on_train_rows_only": true}

Val and test tensors are never passed to a resampler: `train_one` applies
SMOTE to Xtr/ytr only, and `make_loss` is built from train labels only.

Untagged runs must weight or resample exactly the train split; tagged runs
(embargo, complete-subset, CV folds) train on a subset of it, so the test
there is that the statistic never exceeds the train split, which it would
have to if it had reached val or test.

The three protocol-audit runs deliberately break this rule -- that is what
they exist to measure -- and are excluded here by construction: they call
the training loop directly and write no run record, so nothing they produce
reaches any reported table other than results/protocol_audit*.csv.
```

**PASS**

## (c) the final test evaluation was executed exactly once per model

```
run records on disk                      : 315
distinct run keys                        : 315
keys with more than one record           : 0
test prediction files                    : 379
test prediction files per key (max)      : 1

One run key = one test prediction file.  `run_deep` writes the file once and
reuses it on every later call, so a model's test set is scored once and every
downstream table reads that same file rather than re-running the model.
```

**PASS**

## (d) all five seeds used identical split membership

```
multi-seed run families checked          : 63
families whose seeds saw different rows  : 0

example families (row count and positive count identical across seeds):
  {"family": "bilstm/1/class_weight", "n_seeds": 5, "n_test_rows": 32216, "n_test_positives": 60}
  {"family": "bilstm/1/focal", "n_seeds": 5, "n_test_rows": 32216, "n_test_positives": 60}
  {"family": "bilstm/1/none", "n_seeds": 5, "n_test_rows": 32216, "n_test_positives": 60}
  {"family": "bilstm/1/smote", "n_seeds": 5, "n_test_rows": 32216, "n_test_positives": 60}
  {"family": "bilstm/2/class_weight", "n_seeds": 5, "n_test_rows": 32216, "n_test_positives": 158}
  {"family": "bilstm/3/class_weight", "n_seeds": 5, "n_test_rows": 32216, "n_test_positives": 260}

Seeds differ only in initialisation, shuffling and any resampling inside
train.  Split membership is a function of the window's end quarter and is
fixed by the dataset, so it cannot vary by seed.
```

**PASS**

## Supporting evidence

### The scaler is never refitted

```
scaler file                              : scaler_params_full.json
scaler parameter entries                 : 6
This repository never refits it.  `data.load_split` reads the already-scaled
`X` and asserts it, and no module calls a `fit` method on a scaler over `X`.
mean of observed train cells per feature  : min -0.000, max +0.000 (0 by construction)
```

### Thresholds are chosen on validation only

```
Every threshold in this repository is produced by one of two functions, both
of which are called on validation predictions and never on test:
  metrics.best_f1_threshold(y_val, p_val)
  metrics.best_cost_threshold(y_val, p_val, fn_cost)
experiment.score() reads results/preds_val/<key>.parquet to choose the
threshold and results/preds/<key>.parquet to apply it, in that order.
```

## Verdict

| Check | Result |
|---|---|
| (a) no window index appears in more than one split in any result file | PASS |
| (b) train-split resampling and weighting statistics come from train rows only | PASS |
| (c) the final test evaluation was executed exactly once per model | PASS |
| (d) all five seeds used identical split membership | PASS |

**Leakage audit: PASS**
