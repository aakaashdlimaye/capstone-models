"""The four temporal architectures, in PyTorch, matching `Capstone Summary` §6.

The spec is written in Keras notation.  Each module below reproduces it layer
for layer; the two places where a literal reading would be inert or ambiguous
are marked HEAD-ACT and CONV-PAD and are logged in docs/DECISIONS.md.

Every model returns `(logits, aux)`.  `aux` carries the attention weights the
interpretability phase reads:
  * Transformer          -> (n_layers, B, n_heads, L, L)
  * CNN-LSTM-Attention   -> (B, L_pooled)
and is None for the two plain recurrent models.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. LSTM (baseline)
#    LSTM(64, return_sequences) -> LSTM(32) -> Dense(16, ReLU)
#    -> Dropout(0.3) -> Dense(1, sigmoid)
# --------------------------------------------------------------------------
class LSTMNet(nn.Module):
    name = "lstm"

    def __init__(self, n_features: int = 29, n_out: int = 1, hidden1: int = 64,
                 hidden2: int = 32, dense: int = 16, dropout: float = 0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, dense)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dense, n_out)

    def forward(self, x):
        h, _ = self.lstm1(x)              # return_sequences=True
        h, _ = self.lstm2(h)
        h = h[:, -1, :]                   # return_sequences=False
        h = self.drop(F.relu(self.fc1(h)))
        return self.out(h), None


# --------------------------------------------------------------------------
# 2. Bi-LSTM
#    BiLSTM(64) -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(1, sigmoid)
# --------------------------------------------------------------------------
class BiLSTMNet(nn.Module):
    name = "bilstm"

    def __init__(self, n_features: int = 29, n_out: int = 1, hidden: int = 64,
                 dense: int = 32, dropout: float = 0.3):
        super().__init__()
        self.bilstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden, dense)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dense, n_out)

    def forward(self, x):
        h, _ = self.bilstm(x)
        # Keras Bidirectional(LSTM(64)) with return_sequences=False concatenates
        # the forward pass's last step with the backward pass's last step,
        # which is its *first* time index.
        fwd = h[:, -1, :h.shape[2] // 2]
        bwd = h[:, 0, h.shape[2] // 2:]
        h = torch.cat([fwd, bwd], dim=1)
        h = self.drop(F.relu(self.fc1(h)))
        return self.out(h), None


# --------------------------------------------------------------------------
# 3. Transformer encoder (primary)
#    Linear embed -> 64 -> sinusoidal PE -> 2 blocks (8 heads, 128 FFN,
#    LN + residual) -> global average pool -> Dense(32) -> Dense(1, sigmoid)
# --------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class EncoderBlock(nn.Module):
    """Post-norm block, as Vaswani et al. and the Keras spec describe it."""

    def __init__(self, d_model: int, n_heads: int, ffn: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, need_weights: bool = False):
        a, w = self.attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = self.ln1(x + self.drop(a))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x, w


class TransformerNet(nn.Module):
    name = "transformer"

    def __init__(self, n_features: int = 29, n_out: int = 1, d_model: int = 64,
                 n_heads: int = 8, ffn: int = 128, n_layers: int = 2,
                 dense: int = 32, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pe = SinusoidalPE(d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, ffn, dropout) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(d_model, dense)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dense, n_out)

    def forward(self, x, need_weights: bool = False):
        h = self.pe(self.embed(x))
        ws = []
        for blk in self.blocks:
            h, w = blk(h, need_weights=need_weights)
            if need_weights:
                ws.append(w)
        h = h.mean(dim=1)                                  # GlobalAveragePooling1D
        h = self.drop(F.relu(self.fc1(h)))                 # HEAD-ACT
        aux = torch.stack(ws) if need_weights and ws else None
        return self.out(h), aux


# --------------------------------------------------------------------------
# 4. CNN-LSTM-Attention (hybrid)
#    Conv1D(32, k=3, ReLU) -> MaxPool1D(2) -> LSTM(64, return_sequences)
#    -> additive attention -> Dense(16) -> Dropout(0.3) -> Dense(1, sigmoid)
# --------------------------------------------------------------------------
class AdditiveAttention(nn.Module):
    """Bahdanau-style scoring over time: alpha = softmax(v^T tanh(W h))."""

    def __init__(self, hidden: int, units: int = 32):
        super().__init__()
        self.W = nn.Linear(hidden, units)
        self.v = nn.Linear(units, 1, bias=False)

    def forward(self, h):
        score = self.v(torch.tanh(self.W(h))).squeeze(-1)   # (B, L)
        alpha = torch.softmax(score, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class CNNLSTMAttnNet(nn.Module):
    name = "cnn_lstm_attn"

    def __init__(self, n_features: int = 29, n_out: int = 1, filters: int = 32,
                 kernel: int = 3, hidden: int = 64, dense: int = 16,
                 attn_units: int = 32, dropout: float = 0.3):
        super().__init__()
        # CONV-PAD: 'same' padding keeps 8 steps so that after MaxPool1D(2) the
        # four attention positions map onto contiguous quarter pairs.
        self.conv = nn.Conv1d(n_features, filters, kernel, padding=kernel // 2)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.attn = AdditiveAttention(hidden, attn_units)
        self.fc1 = nn.Linear(hidden, dense)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dense, n_out)

    def forward(self, x, need_weights: bool = False):
        h = F.relu(self.conv(x.transpose(1, 2)))
        h = self.pool(h).transpose(1, 2)                    # (B, L/2, filters)
        h, _ = self.lstm(h)
        ctx, alpha = self.attn(h)
        z = self.drop(F.relu(self.fc1(ctx)))                # HEAD-ACT
        return self.out(z), (alpha if need_weights else None)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 5. MLP — the control that separates "deep" from "temporal"
# --------------------------------------------------------------------------
class MLPNet(nn.Module):
    """Feed-forward net over the flattened input, with no time modelling at all.

    Fed a single-quarter tensor (T = 1) this is a static nonlinear classifier,
    which is what isolates the LSTM's nonlinearity from its time axis: without
    it, B (linear, static) -> C (nonlinear, sequential) cannot say which of the
    two changes did the work.  Fed the full 8-quarter tensor it becomes the
    flattened-window control instead.

    Widths mirror the recurrent heads (64 -> 32 -> 16) so the comparison is not
    a capacity comparison in disguise.
    """

    name = "mlp"

    def __init__(self, n_features: int = 29, n_out: int = 1, n_steps: int = 1,
                 hidden: tuple[int, ...] = (64, 32, 16), dropout: float = 0.3):
        super().__init__()
        self.n_steps = n_steps
        dims = [n_features * n_steps, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], n_out)

    def forward(self, x):
        return self.out(self.body(x.flatten(start_dim=1))), None


REGISTRY = {
    "lstm": LSTMNet,
    "bilstm": BiLSTMNet,
    "transformer": TransformerNet,
    "cnn_lstm_attn": CNNLSTMAttnNet,
    "mlp": MLPNet,
}

HAS_ATTENTION = {"transformer", "cnn_lstm_attn"}


def build(name: str, n_features: int = 29, n_out: int = 1, **kwargs) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown architecture {name!r}; have {sorted(REGISTRY)}")
    if name != "mlp":
        kwargs.pop("n_steps", None)     # only the MLP flattens, so only it needs this
    return REGISTRY[name](n_features=n_features, n_out=n_out, **kwargs)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def forward_with_attention(model: nn.Module, x: torch.Tensor):
    """Forward pass that asks for attention when the architecture has any."""
    if model.name in HAS_ATTENTION:
        return model(x, need_weights=True)
    return model(x)
