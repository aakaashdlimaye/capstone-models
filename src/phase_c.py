"""Phase C — the four temporal architectures.

All four share the input tensor (8 x 29), the split, the imbalance treatment
and the tuning budget, so any difference between them is attributable to the
architecture.  Every reported number is a 5-seed mean +/- std.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import experiment as E
from . import models as Models
from . import tuning as Tune
from . import utils as U


def parameter_counts(n_features: int = 29, n_out: int = 1) -> pd.DataFrame:
    rows = []
    for arch in C.ARCHITECTURES:
        m = Models.build(arch, n_features=n_features, n_out=n_out)
        rows.append({"architecture": arch, "n_parameters": Models.n_params(m),
                     "n_features": n_features, "n_outputs": n_out})
    return pd.DataFrame(rows)


def run_tuning(splits: dict[str, D.SplitData], n_trials: int = C.TUNING_TRIALS,
               force: bool = False) -> pd.DataFrame:
    """One random-search study per architecture, identical budget, val only."""
    b = E.bundle(splits, horizon=Tune.TUNE_HORIZON)
    rows = []
    for arch in C.ARCHITECTURES:
        print(f"[phase C] tuning {arch} ({n_trials} trials, val PR-AUC, h={Tune.TUNE_HORIZON})",
              flush=True)
        res = Tune.tune(arch, b, n_trials=n_trials, force=force)
        rows.append({"architecture": arch, **res["best"],
                     "best_val_pr_auc": res["best_val_pr_auc"],
                     "spec_default_val_pr_auc": res["spec_default_val_pr_auc"],
                     "n_trials": res["n_trials"]})
    df = pd.DataFrame(rows)
    U.write_table(df, C.RESULTS / "tuning_summary", floatfmt="%.5f")
    return df


def run_single_horizon(splits, horizons=C.HORIZONS, seeds=C.SEEDS,
                       treatment: str = "class_weight", force: bool = False):
    """4 architectures x 4 horizons x 5 seeds, under the spec's class weights."""
    records, scores = [], []
    for h in horizons:
        b = E.bundle(splits, horizon=h)
        for arch in C.ARCHITECTURES:
            hp = Tune.best_hparams(arch)
            for seed in seeds:
                spec = E.RunSpec(arch=arch, horizon=h, treatment=treatment, seed=seed,
                                 hp=hp, want_attention=(arch in Models.HAS_ATTENTION))
                rec = E.run_deep(spec, b, force=force)
                records.append(rec)
                scores.append(E.score(spec.key))
                print(f"[phase C] {spec.key:42s} test PR-AUC {scores[-1]['pr_auc']:.4f} "
                      f"ROC {scores[-1]['roc_auc']:.4f} ({rec['epochs_run']} ep, "
                      f"{rec['seconds']:.0f}s)", flush=True)
    return records, scores


def run_multi_horizon(splits, seeds=C.SEEDS, treatment: str = "class_weight",
                      force: bool = False):
    """The Korangi et al. (2023) term-structure head: one model, four outputs."""
    b = E.bundle(splits, horizon="multi")
    records, scores = [], []
    for arch in C.ARCHITECTURES:
        hp = Tune.best_hparams(arch)
        for seed in seeds:
            spec = E.RunSpec(arch=arch, horizon="multi", treatment=treatment, seed=seed,
                             hp=hp, note="multi-horizon 4-unit sigmoid head")
            rec = E.run_deep(spec, b, force=force)
            records.append(rec)
            for h in C.HORIZONS:
                s = E.score(spec.key, horizon_col=f"_h{h}")
                s.update({"arch": arch, "seed": seed, "horizon": h, "head": "multi"})
                scores.append(s)
            print(f"[phase C] {spec.key:42s} done ({rec['epochs_run']} ep, {rec['seconds']:.0f}s)",
                  flush=True)
    return records, scores


def run_indicator_ablation(splits, arch: str, horizons=(1, 4), seeds=C.SEEDS,
                           force: bool = False):
    """Does feeding has_inventory / has_debt as extra channels help?

    Three of the 29 ratios are structurally undefined for firms without
    inventory and one for firms without debt; the indicators are what tells a
    model that a zero there means 'not applicable' rather than 'missing'.
    """
    records, scores = [], []
    for h in horizons:
        b = E.bundle(splits, horizon=h, with_indicators=True)
        hp = Tune.best_hparams(arch)
        for seed in seeds:
            spec = E.RunSpec(arch=arch, horizon=h, treatment="class_weight", seed=seed,
                             tag="ind", hp=hp, note="29 ratios + 2 structural indicators")
            rec = E.run_deep(spec, b, force=force)
            records.append(rec)
            s = E.score(spec.key)
            s.update({"arch": arch, "seed": seed, "horizon": h, "inputs": "29+2"})
            scores.append(s)
    return records, scores


# --------------------------------------------------------------------------
def main(full: bool = True, seeds=C.SEEDS, horizons=C.HORIZONS,
         n_trials: int = C.TUNING_TRIALS, force: bool = False) -> dict:
    print("=" * 70)
    print("PHASE C — FOUR TEMPORAL ARCHITECTURES")
    print("=" * 70)
    splits = D.load_all(full=full)

    params = parameter_counts()
    U.write_table(params, C.RESULTS / "architecture_parameters", floatfmt="%.0f")
    print(params.to_string(index=False))

    run_tuning(splits, n_trials=n_trials, force=force)

    recs, scores = run_single_horizon(splits, horizons=horizons, seeds=seeds, force=force)
    summary = E.summarise(recs, scores, by=["arch", "horizon"])
    for h in horizons:
        sub = summary[summary["horizon"] == h].copy()
        U.write_table(sub, C.RESULTS / f"deep_{h}")
    summary.to_csv(C.RESULTS / "deep_all.csv", index=False)

    mrecs, mscores = run_multi_horizon(splits, seeds=seeds, force=force)
    ms = pd.DataFrame(mscores)
    magg = (ms.groupby(["arch", "horizon"])
              .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                   pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                   f1_mean=("f1", "mean"), recall_mean=("recall", "mean"),
                   n_seeds=("seed", "count")).reset_index())
    single = summary[["arch", "horizon", "pr_auc_mean", "pr_auc_std",
                      "roc_auc_mean", "roc_auc_std"]].rename(
        columns={c: f"single_{c}" for c in ("pr_auc_mean", "pr_auc_std",
                                            "roc_auc_mean", "roc_auc_std")})
    cmp = magg.merge(single, on=["arch", "horizon"], how="left")
    cmp["multi_beats_single_pr"] = cmp["pr_auc_mean"] > cmp["single_pr_auc_mean"]
    U.write_table(cmp, C.RESULTS / "multi_horizon_comparison", floatfmt="%.4f")

    best_arch = (summary[summary["horizon"] == 4]
                 .sort_values("pr_auc_mean", ascending=False)["arch"].iloc[0])
    irecs, iscores = run_indicator_ablation(splits, best_arch, seeds=seeds, force=force)
    isc = pd.DataFrame(iscores)
    iagg = (isc.groupby(["arch", "horizon"])
            .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                 roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                 n_seeds=("seed", "count")).reset_index())
    iagg["inputs"] = "29 ratios + 2 indicators"
    base29 = summary[summary["arch"] == best_arch][
        ["arch", "horizon", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std"]].rename(
        columns={c: f"ratios_only_{c}" for c in
                 ("pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std")})
    iagg = iagg.merge(base29, on=["arch", "horizon"], how="left")
    iagg["indicators_help_pr"] = iagg["pr_auc_mean"] > iagg["ratios_only_pr_auc_mean"]
    U.write_table(iagg, C.RESULTS / "indicator_ablation")

    # Gate: every architecture's mean test PR-AUC at h=4 must exceed Altman Z''.
    base = pd.read_csv(C.RESULTS / "baselines_all.csv")
    z = float(base.query("model == 'altman_zdp' and horizon == 4")["pr_auc"].iloc[0])
    g4 = summary[summary["horizon"] == 4][["arch", "pr_auc_mean", "pr_auc_std"]].copy()
    g4["altman_zdp_pr_auc"] = z
    g4["beats_altman"] = g4["pr_auc_mean"] > z
    U.write_table(g4, C.TABLES / "phase_c_gate")
    print(g4.to_string(index=False))
    ok = bool(g4["beats_altman"].all())
    print(f"[phase C] GATE {'PASS' if ok else 'FAIL'}")
    return {"gate_pass": ok, "best_arch": best_arch,
            "n_runs": len(recs) + len(mrecs) + len(irecs)}
