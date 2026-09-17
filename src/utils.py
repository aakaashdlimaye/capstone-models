"""Seeding, provenance, timing and small IO helpers."""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import config as C

_GIT_CACHE: dict[str, str] = {}


def git_sha(repo: Path) -> str:
    """Short SHA of `repo`, or 'unknown' if it is not a git checkout."""
    key = str(repo)
    if key in _GIT_CACHE:
        return _GIT_CACHE[key]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        sha = out.stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"
    _GIT_CACHE[key] = sha
    return sha


def provenance() -> dict:
    """The stamp every result file carries."""
    return {
        "models_sha": git_sha(C.REPO),
        "dataset_sha": git_sha(C.DATASET_REPO),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def set_seed(seed: int) -> None:
    """Seed every source of randomness we use."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:  # pragma: no cover - dataloader hook
    import torch

    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def device() -> "torch.device":  # noqa: F821
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    import torch

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"cuda: {p.name}, {p.total_memory / 1024 ** 3:.1f} GB, torch {torch.__version__}"
    return f"cpu, torch {torch.__version__}"


@contextmanager
def timed(label: str, sink: list | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    msg = f"[time] {label}: {dt:.1f}s"
    print(msg, flush=True)
    if sink is not None:
        sink.append({"label": label, "seconds": dt})


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_table(df, stem: Path, index: bool = False, floatfmt: str = "%.4f") -> None:
    """Write a DataFrame as both CSV and Markdown, alongside each other."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=index)
    stem.with_suffix(".md").write_text(to_markdown(df, index=index, floatfmt=floatfmt), encoding="utf-8")


def to_markdown(df, index: bool = False, floatfmt: str = "%.4f") -> str:
    """Minimal GitHub-flavoured Markdown table (avoids a tabulate dependency)."""
    import pandas as pd

    d = df.reset_index() if index else df
    cols = list(d.columns)

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return floatfmt % v
        return str(v)

    head = "| " + " | ".join(str(c) for c in cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = [
        "| " + " | ".join(fmt(r[c]) for c in cols) + " |"
        for _, r in d.iterrows()
    ]
    return "\n".join([head, rule, *body]) + "\n"


def savefig(fig, stem: Path) -> None:
    """Every figure lands as both PNG and SVG."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def mean_std(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def fmt_ms(mean: float, std: float, nd: int = 4) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.{nd}f} ± {std:.{nd}f}"
