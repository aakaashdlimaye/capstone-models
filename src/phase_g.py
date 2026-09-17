"""Phase G — robustness.

Three checks, each answering a question a reviewer will ask:

  1. rolling-origin CV inside train+val — is the single chronological split a
     lucky draw, and is the 2020 (COVID) fold an outlier?
  2. embargo — does dropping windows that straddle a split boundary change the
     answer?
  3. external validation on the UCI Taiwanese and Polish sets — does the
     protocol audit reproduce on the very datasets the inflated-accuracy
     literature uses?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import classical as K
from . import config as C
from . import data as D
from . import experiment as E
from . import imbalance as IMB
from . import metrics as M
from . import ml_baselines as ML
from . import train as T
from . import tuning as Tune
from . import utils as U

CV_HORIZON = 4          # h=4 has enough positives per fold for PR-AUC to be estimable
EMBARGO_HORIZONS = (4,)


# --------------------------------------------------------------------------
# 1. Rolling-origin cross-validation
# --------------------------------------------------------------------------
def _pooled_trainval(splits) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """train + val stacked; test is not touched anywhere in this function."""
    X = np.concatenate([splits["train"].X, splits["val"].X], axis=0)
    Y = np.concatenate([splits["train"].y, splits["val"].y], axis=0)
    q = np.concatenate([splits["train"].end_quarter, splits["val"].end_quarter])
    cik = np.concatenate([splits["train"].cik, splits["val"].cik])
    return X, Y, q, cik


def rolling_origin(splits, j: pd.DataFrame, arch: str, horizon: int = CV_HORIZON,
                   seeds=C.SEEDS, force: bool = False) -> pd.DataFrame:
    """Expanding-window folds by window end quarter, inside train + val only.

    Each fold trains on everything up to its cut-off, holds the final four
    quarters of that training range back as an inner validation set for early
    stopping and threshold choice, and scores the fold's out-of-sample period.
    """
    X, Y, q, cik = _pooled_trainval(splits)
    y = Y[:, horizon - 1].astype(np.float32)
    hp = Tune.best_hparams(arch)

    # Altman Z'' on the same rows, for the side-by-side comparison.
    key = pd.MultiIndex.from_arrays([j["cik"].astype(str), j["end_quarter"]])
    zdp = pd.Series(-K.altman_z_double_prime(j).ravel(), index=key)

    rows = []
    for cut, (lo, hi) in C.CV_FOLDS:
        in_train_all = q <= cut
        in_fold = (q >= lo) & (q <= hi)
        if in_fold.sum() == 0 or y[in_fold].sum() == 0:
            continue
        inner_cut = _minus_quarters(cut, 4)
        inner_tr = in_train_all & (q <= inner_cut)
        inner_va = in_train_all & (q > inner_cut)
        if y[inner_va].sum() == 0:            # fall back to a random inner split
            rng = np.random.default_rng(0)
            idx = np.flatnonzero(in_train_all)
            rng.shuffle(idx)
            inner_va = np.zeros(len(y), bool); inner_va[idx[: len(idx) // 5]] = True
            inner_tr = in_train_all & ~inner_va

        for seed in seeds:
            res = T.train_one(arch, X[inner_tr], y[inner_tr], X[inner_va], y[inner_va],
                              X[in_fold], y[in_fold], treatment="class_weight",
                              seed=seed, hp=hp, horizon=horizon)
            thr, _ = M.best_f1_threshold(y[inner_va], res.val_pred)
            m = M.evaluate(y[in_fold], res.test_pred, thr=thr)
            rows.append({"fold": f"<= {cut} -> {lo}..{hi}", "cut": cut,
                         "eval_from": lo, "eval_to": hi, "model": arch, "seed": seed,
                         "n_train": int(inner_tr.sum()), "n_eval": int(in_fold.sum()),
                         "n_pos_eval": int(y[in_fold].sum()),
                         "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                         "f1": m["f1"], "recall": m["recall"]})
            print(f"[phase G] CV {arch} fold {cut} seed {seed}: "
                  f"PR-AUC {m['pr_auc']:.4f}", flush=True)

        z = zdp.reindex(pd.MultiIndex.from_arrays([cik[in_fold], q[in_fold]])).to_numpy()
        rows.append({"fold": f"<= {cut} -> {lo}..{hi}", "cut": cut,
                     "eval_from": lo, "eval_to": hi, "model": "altman_zdp", "seed": -1,
                     "n_train": 0, "n_eval": int(in_fold.sum()),
                     "n_pos_eval": int(y[in_fold].sum()),
                     "roc_auc": M.roc_auc(y[in_fold], z), "pr_auc": M.pr_auc(y[in_fold], z),
                     "f1": np.nan, "recall": np.nan})
    return pd.DataFrame(rows)


def _minus_quarters(q: str, n: int) -> str:
    yr, qt = int(q[:4]), int(q[5])
    total = yr * 4 + (qt - 1) - n
    return f"{total // 4}Q{total % 4 + 1}"


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    return (cv.groupby(["model", "fold", "eval_from", "eval_to", "n_eval", "n_pos_eval"])
            .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                 roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                 n_seeds=("seed", "count")).reset_index().sort_values(["model", "eval_from"]))


# --------------------------------------------------------------------------
# 2. Embargo
# --------------------------------------------------------------------------
def embargo(splits, arch: str, horizons=EMBARGO_HORIZONS, seeds=C.SEEDS,
            full: bool = True, force: bool = False) -> pd.DataFrame:
    """Re-run the best model on the stricter, boundary-free split.

    The dataset repo's `--embargo` flag drops val windows whose input starts on
    or before the train cut-off and test windows whose input starts before the
    test period.  Train windows are untouched, so the train-fitted scaler is
    identical and the existing tensors can be masked rather than rebuilt.
    """
    mask = {s: D.embargo_mask(splits[s], full=full) for s in ("train", "val", "test")}
    rows = []
    for h in horizons:
        kept = {s: int(mask[s].sum()) for s in mask}
        pos = {s: int(splits[s].y[mask[s], h - 1].sum()) for s in mask}
        if pos["val"] == 0:
            rows.append({"horizon": h, "arch": arch, "status": "not evaluable",
                         "reason": "the embargoed validation split has no positives at "
                                   "this horizon, so there is nothing to early-stop or "
                                   "threshold on",
                         **{f"n_{s}": kept[s] for s in kept},
                         **{f"pos_{s}": pos[s] for s in pos}})
            continue

        b = E.bundle(splits, horizon=h, row_mask=mask)
        hp = Tune.best_hparams(arch)
        ps, ys = [], None
        for seed in seeds:
            spec = E.RunSpec(arch=arch, horizon=h, treatment="class_weight", seed=seed,
                             tag="embargo", hp=hp, note="boundary-straddling windows dropped")
            E.run_deep(spec, b, force=force)
            te = E.load_preds(spec.key, "test")
            ps.append(te["p_hat"].to_numpy()); ys = te["y_true"].to_numpy().astype(int)
        p = np.mean(ps, axis=0)

        base_key = lambda s: E.RunSpec(arch=arch, horizon=h, treatment="class_weight", seed=s).key  # noqa: E731
        bte = [E.load_preds(base_key(s), "test") for s in seeds]
        bidx = pd.MultiIndex.from_arrays([bte[0]["cik"], bte[0]["end_quarter"]])
        keep = pd.Series(mask["test"], index=bidx)
        bp = np.mean([f["p_hat"].to_numpy() for f in bte], axis=0)[keep.to_numpy()]
        by = bte[0]["y_true"].to_numpy().astype(int)[keep.to_numpy()]

        rows.append({
            "horizon": h, "arch": arch, "status": "evaluated",
            **{f"n_{s}": kept[s] for s in kept}, **{f"pos_{s}": pos[s] for s in pos},
            "embargo_roc_auc": M.roc_auc(ys, p), "embargo_pr_auc": M.pr_auc(ys, p),
            "standard_model_on_embargo_rows_roc_auc": M.roc_auc(by, bp),
            "standard_model_on_embargo_rows_pr_auc": M.pr_auc(by, bp),
            "pr_auc_difference": M.pr_auc(ys, p) - M.pr_auc(by, bp),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. External validation
# --------------------------------------------------------------------------
def load_taiwanese() -> tuple[np.ndarray, np.ndarray, str]:
    df = pd.read_csv(C.EXTERNAL / "taiwanese" / "data.csv")
    y = df.iloc[:, 0].to_numpy().astype(int)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    return X, y, "UCI Taiwanese (6,819 firms, single snapshot)"


def load_polish(year: int = 1) -> tuple[np.ndarray, np.ndarray, str]:
    from scipy.io import arff

    d, _ = arff.loadarff(C.EXTERNAL / "polish" / f"{year}year.arff")
    df = pd.DataFrame(d)
    y = df["class"].apply(lambda v: int(v.decode() if isinstance(v, bytes) else v)).to_numpy()
    X = df.drop(columns=["class"]).to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X, y, f"UCI Polish year {year} (single snapshot)"


def external_protocol(X, y, name: str, seeds=C.SEEDS, n_trials: int = C.TUNING_TRIALS,
                      dataset_key: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same protocol as the main study, plus the protocol audit on the same data.

    These sets carry no time axis, so a chronological split is impossible; the
    honest analogue is a stratified random split with the threshold chosen on
    validation and applied once to test.  That is stated rather than presented
    as equivalent to the chronological protocol.
    """
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=0)
    pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))

    rows = []
    for m in ("logreg", "svm_exact", "random_forest", "xgboost"):
        if m == "svm_exact":
            # These sets are small enough for the exact RBF SVC the spec names.
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            est = Pipeline([("scale", StandardScaler()),
                            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                                        random_state=0))])
            est.fit(Xtr, ytr)
            params = {"kernel": "rbf", "exact": True}
        else:
            res = ML.tune(m, Xtr, ytr, Xva, yva, n_trials=n_trials,
                          path=C.RESULTS / "tuning" / f"ext_{dataset_key}_{m}.json")
            est = ML.make_estimator(m, res["best"], pos_weight, seed=0)
            est.fit(Xtr, ytr)
            params = res["best"]
        pv, pt = ML.predict_proba(est, Xva), ML.predict_proba(est, Xte)
        thr, _ = M.best_f1_threshold(yva, pv)
        r = {"dataset": name, "model": m, "n_train": len(ytr), "n_test": len(yte),
             "base_rate": float(y.mean()), "params": str(params)}
        r.update(M.evaluate(yte, pt, thr=thr))
        rows.append(r)

    # --- the protocol audit, on the datasets the inflated literature uses ----
    audit = []
    for seed in seeds:
        Xs, ys, parent, _ = IMB.smote_with_parents(X[:, None, :], y, seed=seed)
        Xs = Xs[:, 0, :]
        # Inflated: SMOTE first, then a random split, accuracy at 0.5.
        a_tr, a_te, b_tr, b_te = train_test_split(Xs, ys, test_size=0.2,
                                                  random_state=seed, stratify=ys)
        est = ML.make_estimator("random_forest", ML.DEFAULTS["random_forest"], 1.0, seed=seed)
        est.fit(a_tr, b_tr)
        p = ML.predict_proba(est, a_te)
        audit.append({"dataset": name, "protocol": "inflated", "seed": seed,
                      "n_test": len(b_te), "test_positive_rate": float(b_te.mean()),
                      **{k: M.threshold_metrics(b_te, p, 0.5)[k]
                         for k in ("accuracy", "f1", "recall", "specificity")},
                      "roc_auc": M.roc_auc(b_te, p), "pr_auc": M.pr_auc(b_te, p)})
        # Correct: split first, resample inside train only, threshold on val.
        Xr, yr, _ = IMB.smote_sequences(Xtr[:, None, :], ytr, seed=seed)
        est = ML.make_estimator("random_forest", ML.DEFAULTS["random_forest"], 1.0, seed=seed)
        est.fit(Xr[:, 0, :], yr)
        pv, pt = ML.predict_proba(est, Xva), ML.predict_proba(est, Xte)
        thr, _ = M.best_f1_threshold(yva, pv)
        audit.append({"dataset": name, "protocol": "correct", "seed": seed,
                      "n_test": len(yte), "test_positive_rate": float(yte.mean()),
                      **{k: M.threshold_metrics(yte, pt, thr)[k]
                         for k in ("accuracy", "f1", "recall", "specificity")},
                      "roc_auc": M.roc_auc(yte, pt), "pr_auc": M.pr_auc(yte, pt)})
    return pd.DataFrame(rows), pd.DataFrame(audit)


# --------------------------------------------------------------------------
def main(full: bool = True, seeds=C.SEEDS, n_trials: int = C.TUNING_TRIALS,
         best_arch: str | None = None, force: bool = False) -> dict:
    print("=" * 70)
    print("PHASE G — ROBUSTNESS")
    print("=" * 70)
    from . import phase_b as PB

    splits = D.load_all(full=full)
    j, _ = PB.build_levels_table(full=full)

    if best_arch is None:
        deep = pd.read_csv(C.RESULTS / "deep_all.csv")
        best_arch = (deep[deep["horizon"] == 4]
                     .sort_values("pr_auc_mean", ascending=False)["arch"].iloc[0])
    print(f"[phase G] best architecture: {best_arch}")

    cv = rolling_origin(splits, j, best_arch, seeds=seeds, force=force)
    cv.to_csv(C.RESULTS / "rolling_origin_runs.csv", index=False)
    cvs = cv_summary(cv)
    U.write_table(cvs, C.RESULTS / "rolling_origin")

    emb = embargo(splits, best_arch, seeds=seeds, full=full, force=force)
    U.write_table(emb, C.RESULTS / "embargo")

    ext_rows, ext_audit = [], []
    for loader, keyname in ((load_taiwanese, "taiwanese"), (lambda: load_polish(1), "polish1")):
        X, y, name = loader()
        r, a = external_protocol(X, y, name, seeds=seeds, n_trials=n_trials,
                                 dataset_key=keyname)
        ext_rows.append(r); ext_audit.append(a)
    ext = pd.concat(ext_rows, ignore_index=True)
    aud = pd.concat(ext_audit, ignore_index=True)
    U.write_table(ext.drop(columns=["params"]), C.RESULTS / "external_validation")
    ext.to_csv(C.RESULTS / "external_validation_full.csv", index=False)
    auds = (aud.groupby(["dataset", "protocol"])
            .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                 pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                 roc_auc_mean=("roc_auc", "mean"), recall_mean=("recall", "mean"),
                 test_positive_rate=("test_positive_rate", "mean"),
                 n_seeds=("seed", "count")).reset_index())
    U.write_table(auds, C.RESULTS / "external_protocol_audit")
    print(auds.to_string(index=False))
    return {"best_arch": best_arch, "cv_folds": int(cv["fold"].nunique())}
