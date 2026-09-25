"""One training loop, parameterised by model, horizon, imbalance treatment and
seed.  Every deep result in the repository comes through here.

The tensors are small enough (86,592 x 8 x 29 float32 = 80 MB) to live on the
device for the whole run, so there is no DataLoader and no host round-trip per
batch — which is what makes a few hundred 5-seed runs tractable.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import config as C
from . import imbalance as IMB
from . import metrics as M
from . import models as Models
from . import utils as U


@dataclass
class HParams:
    lr: float = 1e-3
    batch_size: int = 32
    dropout: float = 0.3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_epochs: int = C.MAX_EPOCHS
    patience: int = C.PATIENCE
    arch_kwargs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["arch_kwargs"] = dict(self.arch_kwargs)
        return d


@dataclass
class RunResult:
    arch: str
    horizon: int | str
    treatment: str
    seed: int
    n_params: int
    epochs_run: int
    best_epoch: int
    best_val_pr_auc: float
    history: list[dict]
    val_pred: np.ndarray
    test_pred: np.ndarray
    train_pred: np.ndarray | None
    attention: np.ndarray | None
    imbalance_stats: dict
    hparams: dict
    state_dict: dict
    seconds: float
    device: str


def _to_device(a: np.ndarray, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


@torch.no_grad()
def predict(model, X: torch.Tensor, batch: int = 4096,
            want_attention: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    outs, attns = [], []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch]
        if want_attention and model.name in Models.HAS_ATTENTION:
            logit, aux = model(xb, need_weights=True)
            if aux is not None:
                attns.append(aux.detach().float().cpu().numpy())
        else:
            logit, _ = model(xb)
        outs.append(torch.sigmoid(logit).detach().float().cpu().numpy())
    p = np.concatenate(outs, axis=0)
    if p.shape[1] == 1:
        p = p.ravel()
    if not attns:
        return p, None
    axis = 1 if attns[0].ndim == 5 else 0      # transformer stacks layers first
    return p, np.concatenate(attns, axis=axis)


def train_one(
    arch: str,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xval: np.ndarray, yval: np.ndarray,
    Xte: np.ndarray, yte: np.ndarray,
    treatment: str = "class_weight",
    seed: int = 0,
    hp: HParams | None = None,
    horizon: int | str = 4,
    device=None,
    want_attention: bool = False,
    keep_train_pred: bool = False,
    verbose: bool = False,
) -> RunResult:
    """Train one configuration and score val and test exactly once each."""
    t0 = time.perf_counter()
    torch.backends.cudnn.benchmark = True
    hp = hp or HParams()
    U.set_seed(seed)
    device = device or U.device()

    multi = ytr.ndim > 1 and ytr.shape[1] > 1
    n_out = ytr.shape[1] if multi else 1

    # --- imbalance treatment, fitted on train rows only ------------------
    if treatment == "smote":
        if multi:
            raise ValueError("SMOTE is defined for a single binary target, not the multi-horizon head")
        Xtr_f, ytr_f, imb_stats = IMB.smote_sequences(Xtr, ytr, seed=seed)
    else:
        Xtr_f, ytr_f, imb_stats = Xtr, ytr, {"applied": False}

    loss_src = ytr_f if not multi else ytr_f[:, -1]
    loss_fn, loss_stats = IMB.make_loss(treatment, loss_src, device=device)
    imb_stats = {**imb_stats, **loss_stats}

    # --- everything resident on the device -------------------------------
    Xtr_t = _to_device(Xtr_f, device)
    ytr_t = _to_device(np.asarray(ytr_f, dtype=np.float32).reshape(len(ytr_f), n_out), device)
    Xval_t = _to_device(Xval, device)
    Xte_t = _to_device(Xte, device)

    model = Models.build(arch, n_features=Xtr.shape[2], n_out=n_out,
                         n_steps=Xtr.shape[1], dropout=hp.dropout,
                         **hp.arch_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    # Early stopping watches validation PR-AUC.  With 37 positives at h=1 the
    # ROC-AUC the original spec named is far too noisy to stop on.
    yval_sel = yval[:, -1] if multi else yval
    n = Xtr_t.shape[0]
    best, best_epoch, best_state, history = -np.inf, -1, None, []
    g = torch.Generator(device="cpu"); g.manual_seed(seed)

    for epoch in range(hp.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        # The running loss stays a device tensor: reading it per step would
        # force a host sync on every one of the ~2,700 batches an epoch.
        total = torch.zeros((), device=device)
        for i in range(0, n, hp.batch_size):
            idx = perm[i:i + hp.batch_size]
            logits, _ = model(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if hp.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step()
            total += loss.detach() * idx.numel()
        train_loss = float(total) / n

        pval, _ = predict(model, Xval_t)
        pval_sel = pval[:, -1] if multi else pval
        vpr = M.pr_auc(yval_sel, pval_sel)
        vroc = M.roc_auc(yval_sel, pval_sel)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_pr_auc": vpr, "val_roc_auc": vroc})
        if verbose:
            print(f"    epoch {epoch:3d} loss {train_loss:.5f} val PR-AUC {vpr:.4f}", flush=True)
        if np.isfinite(vpr) and vpr > best:
            best, best_epoch = vpr, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= hp.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pval, _ = predict(model, Xval_t)
    pte, attn = predict(model, Xte_t, want_attention=want_attention)
    ptr = predict(model, Xtr_t)[0] if keep_train_pred else None

    return RunResult(
        arch=arch, horizon=horizon, treatment=treatment, seed=seed,
        n_params=Models.n_params(model), epochs_run=len(history), best_epoch=best_epoch,
        best_val_pr_auc=float(best) if np.isfinite(best) else float("nan"),
        history=history, val_pred=pval, test_pred=pte, train_pred=ptr, attention=attn,
        imbalance_stats=imb_stats, hparams=hp.as_dict(),
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        seconds=time.perf_counter() - t0, device=str(device),
    )
