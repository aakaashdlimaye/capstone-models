"""Phase A — load the tensors, assert the contract, build the row-set table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import utils as U


def main(full: bool = True) -> dict:
    print("=" * 70)
    print("PHASE A — DATA CONTRACT AND ROW SETS")
    print("=" * 70)
    print(f"[phase A] device: {U.describe_device()}")

    splits = D.load_all(full=full)
    for name, d in splits.items():
        print(f"[phase A] {name:5s} X{d.X.shape} y{d.y.shape} mask{d.mask.shape} "
              f"indicators{d.indicators.shape} — all assertions passed")

    rs = D.build_row_sets(full=full)
    dup = int(rs.duplicated(["cik", "end_quarter"]).sum())
    print(f"[phase A] results/row_sets.csv: {len(rs):,} rows, {dup} duplicates")

    bal = D.class_balance(full=full)
    U.write_table(bal, C.RESULTS / "class_balance", floatfmt="%.5f")
    print(bal.to_string(index=False))

    masked = {name: float((d.mask == 0).mean()) for name, d in splits.items()}
    print(f"[phase A] fraction of cells imputed or structurally undefined: "
          f"{ {k: round(v, 4) for k, v in masked.items()} }")

    ok = (len(rs) == (C.EXPECTED_TOTAL_WINDOWS if full else len(rs))) and dup == 0
    print(f"[phase A] GATE {'PASS' if ok else 'FAIL'}")
    return {"gate_pass": bool(ok), "n_rows": int(len(rs))}
