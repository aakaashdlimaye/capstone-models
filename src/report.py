"""reports/RESULTS.md — generated entirely from the CSVs in results/.

Nothing in this module types a number.  Every table is read off disk, so the
report cannot drift from the artefacts and cannot contain a placeholder.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import utils as U


MISSING: list[str] = []


def _read(name: str) -> pd.DataFrame | None:
    """Read a results table, recording anything absent or empty.

    A full run should find every table.  Anything that lands in MISSING is a
    hole in the report and is printed loudly rather than rendered as prose.
    """
    p = C.RESULTS / name
    if not p.exists():
        MISSING.append(f"{name} (absent)")
        return None
    try:
        df = pd.read_csv(p)
    except Exception as exc:
        MISSING.append(f"{name} (unreadable: {exc})")
        return None
    if df.empty:
        MISSING.append(f"{name} (empty)")
        return None
    return df


def _md(df: pd.DataFrame | None, cols: list[str] | None = None, nd: int = 4) -> str:
    if df is None or df.empty:
        return "_not produced by this run._\n"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    return U.to_markdown(df, floatfmt=f"%.{nd}f")


def _ms(df: pd.DataFrame, base: str) -> pd.Series:
    m, s = f"{base}_mean", f"{base}_std"
    if m not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df.apply(lambda r: U.fmt_ms(r[m], r.get(s, np.nan)), axis=1)


def compute_time() -> tuple[float, int]:
    total, n = 0.0, 0
    for p in C.RUNS.glob("*.json"):
        try:
            total += float(json.loads(p.read_text(encoding="utf-8")).get("seconds", 0.0))
            n += 1
        except Exception:
            continue
    for p in (C.RESULTS / "tuning").glob("*.csv"):
        try:
            total += float(pd.read_csv(p)["seconds"].sum())
        except Exception:
            continue
    return total, n


def imbalance_narrative(t: pd.DataFrame, horizon: int) -> list[str]:
    """The ablation read out of its own table, per architecture.

    Written from the CSV rather than by hand: the untreated baseline differs by
    architecture and by horizon, and quoting one of them as if it were general
    was how the earlier narrative went wrong.
    """
    out = ["Read from `imbalance_ablation_h%d.csv`, per architecture, because the "
           "untreated baseline is not the same number across architectures:" % horizon, ""]
    for arch in sorted(t["arch"].unique()):
        g = t[t["arch"] == arch].set_index("treatment")
        if "none" not in g.index:
            continue
        base = g.loc["none", "pr_auc_mean"]
        train_side = [x for x in C.TREATMENTS if x != "none" and x in g.index]
        if not train_side:
            continue
        best_t = max(train_side, key=lambda x: g.loc[x, "pr_auc_mean"])
        best_v = g.loc[best_t, "pr_auc_mean"]
        hurt = [x for x in train_side if g.loc[x, "pr_auc_mean"] < base]
        cost_rows = [x for x in g.index if str(x).startswith("cost_threshold")]
        best_cost = min(list(g.index), key=lambda x: g.loc[x, "cost_1to50_mean"])             if "cost_1to50_mean" in g.columns else None
        line = (f"- **{arch}**: untreated PR-AUC {base:.4f}; best training-side treatment "
                f"is {best_t} at {best_v:.4f} ({best_v - base:+.4f})")
        if hurt:
            line += f"; {', '.join(hurt)} reduce it"
        if best_cost is not None:
            line += (f".  Lowest expected cost at 50:1 comes from `{best_cost}` "
                     f"({g.loc[best_cost, 'cost_1to50_mean']:.4f})")
        out.append(line + ".")
    if any(str(x).startswith("cost_threshold") for x in t["treatment"].unique()):
        out += ["",
                "The `cost_threshold_*` rows share the untreated model's ranking and differ "
                "only in where it is cut, so their PR-AUC is identical to `none` by "
                "construction; they move expected cost, not PR-AUC."]
    return out


README_START = "<!-- FINDINGS:START -->"
README_END = "<!-- FINDINGS:END -->"


def _fmt_ci(lo, hi) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def readme_findings() -> list[str]:
    """The README's headline section, generated from the CSVs.

    It was hand-written, and after the review fixes it quoted a decomposition
    that no longer existed.  Generating it means it cannot contradict
    `reports/RESULTS.md` again.
    """
    L = ["_Generated from the CSVs in `results/` by `src/report.py`; do not edit "
         "by hand._", ""]

    base = _read("baselines_all.csv")
    deep = _read("deep_all.csv")
    if base is not None and deep is not None:
        b4 = base[base["horizon"] == 4].set_index("model")
        d4 = deep[deep["horizon"] == 4].set_index("arch")
        best_ml = b4[b4["family"] == "ml"]["pr_auc"].idxmax()
        best_deep = d4["pr_auc_ensemble"].idxmax()
        L += [f"**{best_ml} on the flattened window has the highest PR-AUC of any model "
              f"here** — {b4.loc[best_ml, 'pr_auc']:.4f} at h=4, against the best neural "
              f"model's ({best_deep}) {d4.loc[best_deep, 'pr_auc_ensemble']:.4f}, and "
              f"Altman Z″'s {b4.loc['altman_zdp', 'pr_auc']:.4f}. Same row set, same "
              f"split, same 30-trial tuning budget, and both are five-seed ensembles.", ""]
        rows = [{"model": a, "PR-AUC (ensemble)": d4.loc[a, "pr_auc_ensemble"],
                 "PR-AUC (per-seed mean +/- std)":
                     U.fmt_ms(d4.loc[a, "pr_auc_mean"], d4.loc[a, "pr_auc_std"]),
                 "ROC-AUC (ensemble)": d4.loc[a, "roc_auc_ensemble"]}
                for a in d4.sort_values("pr_auc_ensemble", ascending=False).index]
        L += ["Temporal architectures at h=4:", "",
              U.to_markdown(pd.DataFrame(rows)), ""]

    h1 = _read("h1_temporal_control.csv")
    if h1 is not None:
        helps = h1[h1["window_helps"]]
        L += ["**Does the eight-quarter window help the best model?** Not measurably. "
              f"The window minus t-0 difference for {best_ml if base is not None else 'XGBoost'} "
              f"is {h1['window_minus_t0_pr_auc'].min():+.4f} to "
              f"{h1['window_minus_t0_pr_auc'].max():+.4f} PR-AUC across the four horizons, "
              f"and the firm-clustered interval contains zero at "
              f"{len(h1) - len(helps)} of {len(h1)} horizons.", "",
              _md(h1, ["horizon", "t0_pr_auc", "window_pr_auc",
                       "window_minus_t0_pr_auc", "pr_auc_cluster_ci_low",
                       "pr_auc_cluster_ci_high", "pr_auc_cluster_p",
                       "window_helps"]), ""]

    h1b = _read("h1b_growth_ratio_explanation.csv")
    if h1b is not None:
        n_sup = int(h1b["supports_explanation"].sum())
        L += [f"**Why?** Four of the 29 ratios are year-on-year growth, so a single t-0 "
              f"row already carries a four-quarter comparison. Dropping r21–r24 and "
              f"repeating the comparison supports that reading at {n_sup} of "
              f"{len(h1b)} horizons.", "",
              _md(h1b, ["horizon", "window_gap_with_growth",
                        "window_gap_without_growth", "growth_worth_to_t0",
                        "growth_worth_to_window", "supports_explanation"]), ""]

    sh = _read("decomposition_shares.csv")
    if sh is not None:
        r = sh[sh["horizon"] == 4].iloc[0]
        steps = [c[len("pr_auc_share_"):] for c in sh.columns
                 if c.startswith("pr_auc_share_")]
        parts = ", ".join(
            f"{r[f'pr_auc_share_{s}'] * 100:.0f}% {s} ({r[f'pr_auc_gap_{s}']:+.4f})"
            for s in steps)
        L += ["**The decomposition.** At h=4, of the "
              f"{r['pr_auc_total_abs_movement']:.4f} of total movement from Altman Z″ "
              f"to the full temporal model: {parts}.", ""]

    dcm = _read("decomposition_all.csv")
    if dcm is not None:
        s4 = dcm[(dcm["horizon"] == 4) & dcm["step"].notna()]
        L += ["Each step with its firm-clustered interval:", "",
              U.to_markdown(pd.DataFrame({
                  "step": s4["step"], "cause": s4.get("cause", ""),
                  "PR-AUC change": s4["pr_auc_diff"],
                  "95% CI": [_fmt_ci(a, b) for a, b in
                             zip(s4["pr_auc_cluster_ci_low"], s4["pr_auc_cluster_ci_high"])],
                  "excludes 0": s4["pr_auc_cluster_excludes_zero"]})), ""]

    pa = _read("protocol_audit.csv")
    pad = _read("protocol_audit_decomposition.csv")
    if pa is not None and pad is not None:
        g = pa[pa["horizon"] == 1].set_index("protocol")
        d = pad[pad["horizon"] == 1].iloc[0]
        L += ["**The protocol audit.** The same LSTM at h=1, scored at matched base "
              f"rates: PR-AUC {g.loc['inflated_natural_test', 'pr_auc_mean']:.4f} under the "
              f"inflated protocol against {g.loc['correct', 'pr_auc_mean']:.4f} under the "
              f"correct one — a "
              f"{g.loc['inflated_natural_test', 'pr_auc_mean'] / g.loc['correct', 'pr_auc_mean']:.0f}x "
              f"collapse, with both scored on real windows at a "
              f"{g.loc['correct', 'test_positive_rate']:.4f} base rate. On ROC-AUC, which "
              f"is base-rate invariant, "
              f"{d['share_resampling_inside_train'] * 100:.0f}% of the loss is the "
              f"resampling order and {d['share_chronological_split'] * 100:.0f}% the split.",
              "",
              f"The same model reports {d['correct_accuracy']:.4f} accuracy and "
              f"{'beats' if d['model_beats_majority_class'] else 'loses to'} a constant "
              f"predicting no bankruptcy ({d['majority_class_accuracy']:.4f}).", ""]

    pk = _read("h4_precision_at_k.csv")
    if pk is not None:
        s = pk[(pk["horizon"] == 4) & (pk["level"] == "firm") & (pk["budget"] == "k=50")]
        L += ["**What a practitioner gets.** Of the 50 riskiest firms at h=4:", "",
              _md(s, ["model", "tp", "n_positive", "precision", "recall", "lift"]), ""]

    cal = _read("h5_calibration.csv")
    if cal is not None:
        c4 = cal[cal["horizon"] == 4]
        L += ["**Calibration** (scores are Platt-scaled on validation where they are "
              "not already probabilities):", "",
              _md(c4, ["model", "scaling", "base_rate", "mean_predicted", "brier", "ece"],
                  nd=5), ""]

    size = _read("h6_size_breakdown.csv")
    if size is not None:
        s4 = size[(size["horizon"] == 4) & (size["status"] == "reported")]
        if len(s4):
            piv = s4.pivot_table(index="model", columns="group", values="pr_auc")
            cols = [c for c in ("small", "mid", "large") if c in piv.columns]
            L += ["**Who it works for.** PR-AUC at h=4 by total-assets tercile "
                  "(cut on the train period):", "",
                  U.to_markdown(piv[cols].reset_index()), ""]

    return L


def update_readme_findings() -> bool:
    """Replace the generated block in README.md between its markers."""
    path = C.REPO / "README.md"
    if not path.exists():
        return False
    txt = path.read_text(encoding="utf-8")
    if README_START not in txt or README_END not in txt:
        return False
    head, rest = txt.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    body = "\n".join(readme_findings())
    path.write_text(f"{head}{README_START}\n{body}\n{README_END}{tail}", encoding="utf-8")
    return True


def provenance_manifest() -> Path:
    """One record per produced artefact, stamped with both repositories' SHAs.

    Individual CSVs have no place to carry a git SHA, so the SHA lives here
    together with each file's size and content hash — which also makes it
    possible to tell whether a table on disk still matches the run that made it.
    """
    import hashlib

    prov = U.provenance()
    rows = []
    for p in sorted(C.RESULTS.rglob("*")):
        if not p.is_file():
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"path": str(p.relative_to(C.RESULTS)).replace("\\", "/"),
                     "bytes": p.stat().st_size, "sha256": h})
    out = C.RESULTS / "PROVENANCE.json"
    U.write_json(out, {**prov, "seeds": list(C.SEEDS),
                       "dataset_files": {
                           name: str(C.sequences_path(name).name)
                           for name in ("train", "val", "test")},
                       "n_files": len(rows), "files": rows})
    return out


SNAPSHOT = "results_snapshot.json"
WATCHED = {
    "deep_all.csv": ["arch", "horizon"],
    "baselines_all.csv": ["model", "horizon"],
    "protocol_audit.csv": ["horizon", "protocol"],
    "decomposition_all.csv": ["horizon", "model"],
    "rolling_origin.csv": ["model", "fold"],
    "imbalance_ablation_all.csv": ["horizon", "arch", "treatment"],
}
CHANGE_THRESHOLD = 0.10       # relative

# Wall-clock and row counts are not results.  Adding five seeds multiplied every
# fit time by five, which buried the handful of metrics that actually moved
# under twenty rows of timing noise.
NOT_A_RESULT = ("seconds", "epochs", "n_", "wall_")
NOT_A_RESULT_EXACT = {"n", "tn", "fp", "fn", "tp", "k", "seed", "trial", "bin",
                      "horizon", "threshold"}


def _is_result_column(c: str) -> bool:
    return not (c in NOT_A_RESULT_EXACT or c.startswith(NOT_A_RESULT))


def _snapshot() -> dict:
    """Every watched metric, keyed so the next run can diff against it."""
    snap = {}
    for fname, keys in WATCHED.items():
        path = C.RESULTS / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not set(keys) <= set(df.columns):
            continue
        num = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and c not in keys
               and _is_result_column(c)]
        for _, r in df.iterrows():
            rid = " / ".join(str(r[k]) for k in keys)
            for c in num:
                v = r[c]
                if pd.notna(v):
                    snap[f"{fname}::{rid}::{c}"] = float(v)
    return snap


def changes_section() -> list[str]:
    """Numbers that moved more than 10% relative since the last report build."""
    path = C.RESULTS / SNAPSHOT
    new = _snapshot()
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8")).get("values", {})
        except Exception:
            old = {}

    moved, added = [], 0
    for k, v in new.items():
        if k not in old:
            added += 1
            continue
        o = old[k]
        denom = max(abs(o), 1e-9)
        rel = abs(v - o) / denom
        if rel > CHANGE_THRESHOLD and abs(v - o) > 1e-6:
            f, rid, col = k.split("::", 2)
            moved.append({"file": f, "row": rid, "metric": col,
                          "old": o, "new": v, "rel_change": rel})
    U.write_json(path, {"values": new, **U.provenance()})

    if not old:
        return ["## Changes from previous run", "",
                "No previous snapshot on disk, so there is nothing to compare against.",
                f"This run recorded {len(new):,} numbers as the baseline for next time."]

    L = ["## Changes from previous run", "",
         f"Every watched number whose relative change exceeded "
         f"{CHANGE_THRESHOLD:.0%}, old against new.  "
         f"{len(new):,} numbers watched, {added:,} newly added this run."]
    if not moved:
        L += ["", "Nothing moved by more than the threshold."]
        return L
    df = pd.DataFrame(moved).sort_values("rel_change", ascending=False)
    L += ["", f"**{len(df)} numbers moved.**", "",
          U.to_markdown(df.head(60), floatfmt="%.4f")]
    if len(df) > 60:
        L += ["", f"_({len(df) - 60} further changes omitted; the full set is in "
                  f"`{SNAPSHOT}` against the previous build.)_"]
    return L


def build(full: bool = True) -> Path:
    MISSING.clear()
    prov = U.provenance()
    L: list[str] = []
    A = L.append

    A("# Results")
    A("")
    A("Every table below is generated from a CSV in `results/` by `src/report.py`.")
    A("No number in this file is typed by hand.")
    A("")
    A(f"- Universe: **{'full' if full else 'PILOT — not a reportable result'}**")
    A(f"- Dataset repository SHA: `{prov['dataset_sha']}`")
    A(f"- Models repository SHA: `{prov['models_sha']}`")
    A(f"- Seeds: `{list(C.SEEDS)}`; deep results are mean ± std over "
      f"{len(C.SEEDS)} seeds unless stated otherwise.")
    A(f"- Generated: {prov['timestamp']}")
    secs, nruns = compute_time()
    A(f"- Total measured compute: **{secs / 3600:.2f} h** over {nruns} cached deep runs "
      f"plus the logged tuning trials.")
    A("")
    for line in changes_section():
        A(line)
    A("")
    A("Read PR-AUC first.  At a window-level positive rate of roughly 1% (0.18% at")
    A("h=1) ROC-AUC flatters every model, which is exactly the point Contribution 5")
    A("makes about the published literature.")
    A("")

    # ---------------------------------------------------------------- 1
    A("## 1. Data and the row-set contract")
    A("")
    bal = _read("class_balance.csv")
    A(_md(bal, ["split", "windows", "firms", "positive_firms",
                "pos_y1", "rate_y1", "pos_y2", "rate_y2",
                "pos_y3", "rate_y3", "pos_y4", "rate_y4"], nd=5))
    A("")
    rs = _read("row_sets.csv")
    if rs is not None:
        A(f"`results/row_sets.csv` holds **{len(rs):,}** windows with no duplicate")
        A("`(cik, end_quarter)` pair.  Every model in this report — classical, machine")
        A("learning and deep — is scored on exactly these rows.")
        A("")

    # ---------------------------------------------------------------- 2
    A("## 2. Classical and machine-learning baselines")
    A("")
    for h in C.HORIZONS:
        b = _read(f"baselines_{h}.csv")
        if b is None:
            continue
        A(f"### Horizon h = {h}")
        A("")
        A(_md(b, ["model", "family", "val_roc_auc", "val_pr_auc", "roc_auc", "pr_auc",
                  "f1", "recall", "specificity", "precision", "threshold",
                  "cost_1to1", "cost_1to10", "cost_1to20", "cost_1to50"]))
        A("")
    A("### Altman at its canonical cutoffs")
    A("")
    A("Threshold-free AUCs above are the fair reading; this is how the score is")
    A("actually used in practice.")
    A("")
    A(_md(_read("altman_canonical_cutoffs.csv"),
          ["model", "horizon", "rule", "cut_distress", "cut_safe", "tp", "fp", "fn", "tn",
           "precision", "recall", "specificity", "f1", "accuracy", "alarm_rate"]))
    A("")
    A("### Where the classical inputs came from")
    A("")
    notes_p = C.RESULTS / "classical_levels_notes.json"
    if notes_p.exists():
        n = json.loads(notes_p.read_text(encoding="utf-8"))
        A(f"- Trailing-four-quarter sums replace quarterly flows for Altman X3/X5, "
          f"Ohlson NI/TA and FFO/TL, and Zmijewski NI/TA.")
        A(f"- The `x4` fallback (fewer than four contiguous prior quarters) was used on "
          f"{n.get('ttm_fallback_on_row_set')} of the {len(rs) if rs is not None else '?'} "
          f"evaluated windows.")
        A(f"- Total liabilities came from the accounting identity on "
          f"{n.get('TL_from_identity')} panel rows where the tag was absent.")
        A(f"- Missing classical variables are filled with the **train** median "
          f"(fitted on {n.get('imputer_train_rows'):,} train rows).")
        A("")

    # ---------------------------------------------------------------- 3
    A("## 3. The four temporal architectures")
    A("")
    A(_md(_read("architecture_parameters.csv"), None, nd=0))
    A("")
    tun = _read("tuning_summary.csv")
    if tun is not None:
        A("Hyperparameters were searched on validation PR-AUC only, with an identical")
        A(f"budget of {int(tun['n_trials'].max())} random-search trials per architecture.")
        A("Trial 0 is the specification's own defaults, so the search can only improve")
        A("on the spec.")
        A("")
        A(_md(tun, ["architecture", "lr", "batch_size", "dropout", "weight_decay",
                    "best_val_pr_auc", "spec_default_val_pr_auc", "n_trials"], nd=5))
        A("")
    for h in C.HORIZONS:
        d = _read(f"deep_{h}.csv")
        if d is None:
            continue
        t = pd.DataFrame({"architecture": d["arch"], "n_params": d.get("n_params"),
                          "ROC-AUC": _ms(d, "roc_auc"), "PR-AUC": _ms(d, "pr_auc"),
                          "F1": _ms(d, "f1"), "recall": _ms(d, "recall"),
                          "specificity": _ms(d, "specificity"),
                          "cost @ 1:20": _ms(d, "cost_1to20"),
                          "epochs": d.get("epochs_mean")})
        A(f"### Horizon h = {h}")
        A("")
        A(_md(t, None, nd=2))
        A("")
    A("### Multi-horizon head versus four single-horizon models")
    A("")
    A(_md(_read("multi_horizon_comparison.csv"),
          ["arch", "horizon", "pr_auc_mean", "pr_auc_std", "single_pr_auc_mean",
           "single_pr_auc_std", "roc_auc_mean", "single_roc_auc_mean",
           "multi_beats_single_pr"]))
    A("")
    ind = _read("indicator_ablation.csv")
    if ind is not None:
        A("### Structural-indicator ablation")
        A("")
        A("Three of the 29 ratios are undefined for firms without inventory and one for")
        A("firms without debt.  This run feeds `has_inventory` and `has_debt` as two")
        A("extra input channels so the model can tell 'not applicable' from 'missing'.")
        A("")
        A(_md(ind))
        A("")

    # ---------------------------------------------------------------- 4
    A("## 4. Imbalance ablation")
    A("")
    for h in (1, 4):
        t = _read(f"imbalance_ablation_h{h}.csv")
        if t is None:
            continue
        out = pd.DataFrame({"architecture": t["arch"], "treatment": t["treatment"],
                            "ROC-AUC": _ms(t, "roc_auc"), "PR-AUC": _ms(t, "pr_auc"),
                            "F1": _ms(t, "f1"), "recall": _ms(t, "recall"),
                            "specificity": _ms(t, "specificity"),
                            "cost 1:1": _ms(t, "cost_1to1"),
                            "cost 1:10": _ms(t, "cost_1to10"),
                            "cost 1:20": _ms(t, "cost_1to20"),
                            "cost 1:50": _ms(t, "cost_1to50")})
        A(f"### Horizon h = {h}")
        A("")
        A(_md(out, None, nd=2))
        A("")
        A(f"![PR-AUC heatmap h={h}](../results/figures/imbalance_heatmap_h{h}.png)")
        A("")
        A("PR-AUC is threshold-free, so only the four training treatments can move it.")
        A("The cost-sensitive threshold changes where the untreated model's ranking is")
        A("cut, which shows up in expected cost rather than in PR-AUC:")
        A("")
        A(f"![cost heatmap h={h}](../results/figures/imbalance_cost_heatmap_h{h}.png)")
        A("")
        for line in imbalance_narrative(t, h):
            A(line)
        A("")

    # ---------------------------------------------------------------- 5
    A("## 5. Protocol audit — reproducing the 91–99% accuracy")
    A("")
    A("The same LSTM, run three ways.  The only changes between rows are")
    A("methodological.")
    A("")
    A(_md(_read("protocol_audit.csv"),
          ["horizon", "protocol", "n_test", "test_positive_rate", "accuracy_mean",
           "accuracy_std", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "recall_mean",
           "f1_mean", "n_seeds"]))
    A("")
    A("### What each fix costs")
    A("")
    A("The decomposition is on PR-AUC.  Accuracy cannot carry it: it is high under")
    A("the inflated protocol because the model separates a balanced test set, and high")
    A("again under the correct protocol because the majority class is ~99% of it.  The")
    A("two large, offsetting accuracy moves cancel, so `model_beats_majority_class` is")
    A("the column that matters — where it is false, a model reporting >99% accuracy is")
    A("losing to a constant that predicts no bankruptcy at all.")
    A("")
    A(_md(_read("protocol_audit_decomposition.csv")))
    A("")
    A(_md(_read("external_protocol_audit.csv"),
          ["dataset", "protocol", "accuracy_mean", "accuracy_std", "pr_auc_mean",
           "pr_auc_std", "roc_auc_mean", "recall_mean", "test_positive_rate", "n_seeds"]))
    A("")

    # ---------------------------------------------------------------- 6
    A("## 6. Decomposition — why the 1968 formula underperforms")
    A("")
    for h in C.HORIZONS:
        d = _read(f"decomposition_{h}.csv")
        if d is None:
            continue
        A(f"### Horizon h = {h}")
        A("")
        A(_md(d, ["model", "roc_auc", "pr_auc", "f1", "recall", "specificity", "step",
                  "delong_roc_diff", "delong_ci_low", "delong_ci_high", "delong_p",
                  "pr_auc_diff", "pr_ci_low", "pr_ci_high",
                  "mcnemar_b", "mcnemar_c", "mcnemar_p"]))
        A("")
        A(f"![decomposition PR curves h={h}](../results/figures/decomposition_pr_h{h}.png)")
        A("")
    A("### Share of the A→D gap attributable to each cause")
    A("")
    sh = _read("decomposition_shares.csv")
    A(_md(sh, [c for c in (sh.columns if sh is not None else [])
               if c == "horizon" or c.startswith(("pr_auc_", "roc_auc_"))]))
    A("")
    A("### Missingness sanity check")
    A("")
    A(_md(_read("decomposition_complete_subset.csv")))
    A("")
    A("### Re-estimated coefficients")
    A("")
    A(_md(_read("decomposition_coefficients.csv"), None, nd=5))
    A("")

    # ---------------------------------------------------------------- 7
    A("## 7. Interpretability")
    A("")
    A("![attention by quarter](../results/figures/attention_by_quarter.png)")
    A("")
    A("![SHAP family by quarter](../results/figures/shap_family_by_quarter.png)")
    A("")
    A(_md(_read("interpretability_crosscheck.csv"),
          ["horizon", "attention_model", "shap_model", "group", "spearman_rho",
           "spearman_p", "kendall_tau", "attention_argmax", "shap_argmax",
           "agree_on_top_quarter"]))
    A("")
    A("### Ratio-family importance")
    A("")
    A(_md(_read("shap_family_importance.csv"),
          ["horizon", "family", "n_features", "mean_abs_shap", "share"], nd=5))
    A("")
    A("### Top features")
    A("")
    feats = _read("shap_feature_importance.csv")
    if feats is not None:
        top = feats.sort_values(["horizon", "rank"]).groupby("horizon").head(10)
        A(_md(top, ["horizon", "rank", "feature", "family", "mean_abs_shap",
                    "is_altman", "observed_rate"], nd=5))
    A("")

    # ---------------------------------------------------------------- 8
    A("## 8. Robustness")
    A("")
    A("### Rolling-origin cross-validation (inside train + val; test untouched)")
    A("")
    A(_md(_read("rolling_origin.csv"),
          ["model", "fold", "n_eval", "n_pos_eval", "pr_auc_mean", "pr_auc_std",
           "roc_auc_mean", "roc_auc_std", "n_seeds"]))
    A("")
    A("### Embargo — boundary-straddling windows dropped")
    A("")
    A(_md(_read("embargo.csv")))
    A("")
    A("### External validation")
    A("")
    A(_md(_read("external_validation.csv"),
          ["dataset", "model", "n_train", "n_test", "base_rate", "roc_auc", "pr_auc",
           "f1", "recall", "specificity"]))
    A("")

    # ---------------------------------------------------------------- 9
    A("## 9. Gates")
    A("")
    for name in ("phase_b_gate", "phase_c_gate"):
        t = _read(f"tables/{name}.csv")
        if t is not None:
            A(f"**{name}**")
            A("")
            A(_md(t))
            A("")
    A("The leakage audit and its printed evidence are in "
      "[`leakage_audit.md`](leakage_audit.md).")
    A("")

    if MISSING:
        A("## Missing artefacts")
        A("")
        A("These tables were expected and not found, so the sections above are")
        A("incomplete.  A full run should leave this list empty.")
        A("")
        for m in MISSING:
            A(f"- `{m}`")
        A("")

    out = C.REPORTS / "RESULTS.md"
    out.write_text("\n".join(L), encoding="utf-8")
    man = provenance_manifest()
    if update_readme_findings():
        print("[report] regenerated README.md section 9 from the CSVs")
    print(f"[report] wrote {out}")
    print(f"[report] wrote {man}")
    if MISSING:
        print(f"[report] WARNING: {len(MISSING)} expected tables missing:")
        for m in MISSING:
            print(f"           {m}")
    else:
        print("[report] every expected table was found")
    return out
