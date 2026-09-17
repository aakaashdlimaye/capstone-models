"""Architectures match the specification, and the statistics are correct."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src import metrics as M
from src import models as Models


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------
@pytest.mark.parametrize("arch", C.ARCHITECTURES)
def test_forward_shapes(arch):
    m = Models.build(arch)
    x = torch.randn(7, C.WINDOW_LEN, C.N_FEATURES)
    logits, aux = Models.forward_with_attention(m, x)
    assert logits.shape == (7, 1)
    if arch == "transformer":
        assert aux.shape == (2, 7, 8, C.WINDOW_LEN, C.WINDOW_LEN), \
            "2 encoder blocks x 8 heads, attention over the 8 quarters"
    elif arch == "cnn_lstm_attn":
        assert aux.shape == (7, C.WINDOW_LEN // 2), "attention over the pooled positions"
    else:
        assert aux is None


@pytest.mark.parametrize("arch", C.ARCHITECTURES)
def test_multi_horizon_head(arch):
    m = Models.build(arch, n_out=4)
    assert m(torch.randn(3, C.WINDOW_LEN, C.N_FEATURES))[0].shape == (3, 4)


def test_transformer_attention_rows_are_distributions():
    m = Models.build("transformer").eval()
    _, a = m(torch.randn(5, C.WINDOW_LEN, C.N_FEATURES), need_weights=True)
    assert torch.allclose(a.sum(dim=-1), torch.ones_like(a.sum(dim=-1)), atol=1e-5)


def test_additive_attention_weights_sum_to_one():
    m = Models.build("cnn_lstm_attn").eval()
    _, a = m(torch.randn(5, C.WINDOW_LEN, C.N_FEATURES), need_weights=True)
    assert torch.allclose(a.sum(dim=1), torch.ones(5), atol=1e-5)


def test_spec_layer_sizes():
    lstm = Models.build("lstm")
    assert lstm.lstm1.hidden_size == 64 and lstm.lstm2.hidden_size == 32
    assert lstm.fc1.out_features == 16 and lstm.out.out_features == 1

    bi = Models.build("bilstm")
    assert bi.bilstm.hidden_size == 64 and bi.bilstm.bidirectional
    assert bi.fc1.in_features == 128 and bi.fc1.out_features == 32

    tr = Models.build("transformer")
    assert tr.embed.out_features == 64 and len(tr.blocks) == 2
    assert tr.blocks[0].attn.num_heads == 8
    assert tr.blocks[0].ff[0].out_features == 128
    assert tr.fc1.out_features == 32

    cn = Models.build("cnn_lstm_attn")
    assert cn.conv.out_channels == 32 and cn.conv.kernel_size == (3,)
    assert cn.pool.kernel_size == 2 and cn.lstm.hidden_size == 64
    assert cn.fc1.out_features == 16


def test_sinusoidal_positional_encoding_is_fixed_not_learned():
    pe = Models.SinusoidalPE(64)
    assert not any(p.requires_grad for p in pe.parameters())
    x = torch.zeros(1, 8, 64)
    a, b = pe(x), pe(x)
    assert torch.equal(a, b)
    assert not torch.equal(a[0, 0], a[0, 1]), "positions must be distinguishable"


def test_parameter_counts_are_stable():
    counts = {a: Models.n_params(Models.build(a)) for a in C.ARCHITECTURES}
    assert all(v > 0 for v in counts.values())
    assert counts["bilstm"] > counts["lstm"], "a bidirectional layer has more parameters"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _scores(n=3000, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(int)
    return y, rng.random(n) + sep * y


def test_auc_matches_sklearn():
    y, p = _scores()
    assert M.roc_auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert M.pr_auc(y, p) == pytest.approx(average_precision_score(y, p))


def test_delong_auc_matches_sklearn_and_is_zero_against_itself():
    y, p1 = _scores(seed=1, sep=0.6)
    _, p2 = _scores(seed=1, sep=1.4)
    d = M.delong_test(y, p1, p2)
    assert d["auc1"] == pytest.approx(roc_auc_score(y, p1), abs=1e-9)
    assert d["auc2"] == pytest.approx(roc_auc_score(y, p2), abs=1e-9)
    assert d["ci_low"] < d["diff"] < d["ci_high"]
    same = M.delong_test(y, p1, p1)
    assert same["diff"] == pytest.approx(0.0) and same["se"] == pytest.approx(0.0)


def test_delong_handles_ties():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    p1 = np.array([0.5] * 8)
    p2 = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.2, 0.1])
    d = M.delong_test(y, p1, p2)
    assert d["auc1"] == pytest.approx(0.5), "all-tied scores give AUC 0.5"
    assert d["auc2"] == pytest.approx(roc_auc_score(y, p2))


def test_mcnemar_counts_correctness_not_firing():
    y = np.array([1, 1, 0, 0, 0, 0])
    p1 = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1])     # right, wrong, right x4
    p2 = np.array([0.1, 0.9, 0.1, 0.1, 0.1, 0.9])     # wrong, right, right x3, wrong
    r = M.mcnemar_test(y, p1, p2, 0.5, 0.5)
    assert r["b"] == 2 and r["c"] == 1                # b: model 1 right & 2 wrong
    assert r["test"] == "exact binomial"


def test_mcnemar_is_one_when_models_agree():
    y, p = _scores(seed=3)
    r = M.mcnemar_test(y, p, p, 0.5, 0.5)
    assert r["n_discordant"] == 0 and r["p_value"] == 1.0


def test_threshold_metrics_arithmetic():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    m = M.threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_expected_cost_weights_the_two_error_types():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.9, 0.1])       # one FN, one FP
    assert M.expected_cost(y, p, 0.5, fn_cost=1.0) == pytest.approx(2 / 4)
    assert M.expected_cost(y, p, 0.5, fn_cost=20.0) == pytest.approx(21 / 4)


def test_pr_auc_is_undefined_without_positives():
    assert np.isnan(M.pr_auc(np.zeros(10, int), np.random.random(10)))
    assert np.isnan(M.roc_auc(np.zeros(10, int), np.random.random(10)))


def test_bootstrap_ci_brackets_the_point_estimate():
    y, p1 = _scores(seed=7, sep=0.4)
    _, p2 = _scores(seed=7, sep=1.6)
    b = M.bootstrap_diff(y, p1, p2, n_boot=400, seed=0)
    assert b["ci_low"] <= b["diff"] <= b["ci_high"]
    assert b["diff"] > 0
