#!/usr/bin/env python
"""Reproduce every file in results/ from the frozen dataset.

    python run_all.py                 # full universe, the reported numbers
    python run_all.py --pilot         # smoke run on the 18k-window pilot
    python run_all.py --from c        # resume at a phase
    python run_all.py --only b d      # run just these phases

Runs are cached by key: a configuration whose prediction and run files already
exist is not retrained, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

PHASES = ["a", "b", "c", "d", "e", "f", "g", "h", "audit", "report"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true",
                    help="run on the positive-enriched pilot tensors; writes to "
                         "results_pilot/ and is never a reportable number")
    ap.add_argument("--from", dest="start", choices=PHASES, default="a")
    ap.add_argument("--only", nargs="+", choices=PHASES)
    ap.add_argument("--seeds", type=int, default=None,
                    help="seeds per configuration (default 5, or 3 in pilot mode)")
    ap.add_argument("--trials", type=int, default=None,
                    help="tuning trials per model (default 30, or 3 in pilot mode)")
    ap.add_argument("--force", action="store_true", help="ignore cached runs")
    ap.add_argument("--quick", action="store_true",
                    help="cap epochs hard; for wiring checks only")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    # Headless plotting, and paths, must both be set before src is imported.
    os.environ.setdefault("MPLBACKEND", "Agg")
    if a.pilot:
        os.environ["CAPSTONE_RESULTS_DIR"] = str(REPO / "results_pilot")
        os.environ["CAPSTONE_REPORTS_DIR"] = str(REPO / "reports_pilot")
    sys.path.insert(0, str(REPO))

    from src import config as C
    from src import utils as U

    full = not a.pilot
    # The pilot is a wiring check whose numbers are never reported, so it runs a
    # cheaper budget by default and fits inside a coffee break.  Every reported
    # number comes from the full universe at 5 seeds and 30 trials.  Both are
    # overridable: `--pilot --seeds 5 --trials 30` reproduces the full budget.
    seeds = tuple(range(a.seeds if a.seeds is not None else (3 if a.pilot else len(C.SEEDS))))
    trials = a.trials if a.trials is not None else (3 if a.pilot else C.TUNING_TRIALS)
    if a.quick:
        C.MAX_EPOCHS, C.PATIENCE = 4, 2
        from src import tuning as Tune
        Tune.SEARCH_EPOCHS, Tune.SEARCH_PATIENCE = 2, 2

    todo = a.only or PHASES[PHASES.index(a.start):]
    print("=" * 70)
    print(f"RUN_ALL — universe={'FULL' if full else 'PILOT'}  seeds={seeds}  "
          f"tuning trials={trials}")
    print(f"results -> {C.RESULTS}")
    print(f"device  -> {U.describe_device()}")
    print(f"phases  -> {todo}")
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    out: dict = {}
    if "a" in todo:
        from src import phase_a
        out["a"] = phase_a.main(full=full)
    if "b" in todo:
        from src import phase_b
        out["b"] = phase_b.main(full=full, n_trials=trials, seeds=seeds, force=a.force)
    if "c" in todo:
        from src import phase_c
        out["c"] = phase_c.main(full=full, seeds=seeds, n_trials=trials, force=a.force)
    if "d" in todo:
        from src import phase_d
        out["d"] = phase_d.main(full=full, seeds=seeds, force=a.force)
    if "e" in todo:
        from src import phase_e
        out["e"] = phase_e.main(full=full, seeds=seeds, force=a.force)
    if "f" in todo:
        from src import phase_f
        out["f"] = phase_f.main(full=full, seeds=seeds)
    if "g" in todo:
        from src import phase_g
        out["g"] = phase_g.main(full=full, seeds=seeds, n_trials=trials, force=a.force)
    if "h" in todo:
        from src import phase_h
        out["h"] = phase_h.main(full=full, seeds=seeds, n_trials=trials, force=a.force)
    if "audit" in todo:
        from src import leakage
        out["audit"] = leakage.main(full=full)
    if "report" in todo:
        from src import report
        out["report"] = str(report.build(full=full))

    dt = time.perf_counter() - t0
    print("=" * 70)
    print(f"RUN_ALL finished in {dt / 60:.1f} min")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    # A resumed run must not erase the phases an earlier one completed: merge
    # into whatever is already there so the summary describes the artefacts on
    # disk rather than only the last invocation.
    summary_path = C.RESULTS / "run_all_summary.json"
    prev = {}
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    phases = {**prev.get("phases", {}), **out}
    invocations = prev.get("invocations", [])
    invocations.append({"phases": todo, "wall_seconds": dt, "seeds": list(seeds),
                        "tuning_trials": trials, **U.provenance()})
    U.write_json(summary_path,
                 {"phases": phases, "universe": "full" if full else "pilot",
                  "seeds": list(seeds), "tuning_trials": trials,
                  "wall_seconds_total": sum(i["wall_seconds"] for i in invocations),
                  "invocations": invocations, **U.provenance()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
