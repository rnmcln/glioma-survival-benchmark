"""Post hoc analysis R4 (PROTOCOL.md, section 12): evaluation on the full TCGA
cohort with censoring-adjusted metrics, including a neural network trained on
the Cox partial likelihood.

All 664 TCGA patients are split by repeated stratified 5-fold cross-validation
(5 repeats; strata: died by 24 months, alive at 24 months, censored before 24
months). In each fold:
  * binary models (age-and-grade LR, penalised LR, compact MLP with and without
    class weighting) are fitted on the labelled training patients, as in the
    main analysis, and predict 24-month risk for every test patient, including
    those censored before 24 months;
  * survival models (elastic-net penalised Cox, penaliser 0.1, L1 ratio 0.5;
    Cox-loss MLPs with the compact and regularised architectures) are fitted on
    all training patients and predict 1 - S(t | x) at 12, 24 and 60 months.
Metrics on the out-of-fold predictions averaged across repeats: cumulative/
dynamic AUROC at 24 months with inverse-probability-of-censoring weights
(Uno), IPCW Brier score at 24 months, Harrell's C, and mean predicted 24-month
risk against the Kaplan-Meier estimate. External: models fitted on all TCGA
patients and applied to every Gravendeel and REMBRANDT patient.
Paired differences use 2000 bootstrap resamples of patients.
"""
import _path  # noqa: F401
import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from src import config
from src.cv import release_memory
from src.endpoints import make_binary_endpoint
from src.harmonise import load_cohort
from src.models import KerasClassifier, build_models
from src.nn_extra import CoxMLP
from src.preprocessing import make_preprocessor, resolve_features
from src.utils import configure_tensorflow, get_logger, set_seed

log = get_logger()
OUT = config.PREDICTIONS / "revision"
OUT.mkdir(exist_ok=True)
H = 24.0
TIMES = (12.0, 24.0, 60.0)
BINARY = ["logistic_age_grade", "penalised_logistic", "tf_compact", "tf_compact_unweighted"]
SURV = ["cox_penalised", "coxmlp_compact", "coxmlp_regularised"]
COXMLP = {"coxmlp_compact": dict(hidden=(32, 16), dropout=0.2, l2=0.0),
          "coxmlp_regularised": dict(hidden=(64, 32), dropout=0.4, l2=1e-3)}


class PenCox:
    def __init__(self, feats, pen=0.1, l1=0.5):
        self.feats, self.pen, self.l1 = feats, pen, l1

    def fit(self, df):
        self.pre_ = make_preprocessor(df, self.feats).fit(df[self.feats])
        X = self.pre_.transform(df[self.feats])
        self.keep_ = X.std(axis=0) > 1e-8
        d = pd.DataFrame(X[:, self.keep_], columns=[f"x{i}" for i in range(self.keep_.sum())])
        d["T"], d["E"] = df.os_months.values, df.os_event.values
        self.m_ = CoxPHFitter(penalizer=self.pen, l1_ratio=self.l1).fit(d, "T", "E")
        return self

    def risk(self, df, times):
        X = self.pre_.transform(df[self.feats])[:, self.keep_]
        d = pd.DataFrame(X, columns=[f"x{i}" for i in range(self.keep_.sum())])
        S = self.m_.predict_survival_function(d, times=list(times)).values.T
        return 1 - S


def fit_predict(train, test, feats, seed):
    """Return dict model -> array (n_test, len(TIMES)) of predicted risk of
    death by each time (binary models: 24-month column only, others NaN)."""
    out = {}
    lab, _ = make_binary_endpoint(train, H)
    yl = lab.y.values
    ref = [c for c in ("age", "grade") if lab[c].notna().any() and test[c].notna().any()]
    mods = build_models(lab, feats, seed=seed, include_nn=False, reference_vars=ref)
    est = {"logistic_age_grade": mods["logistic_age_grade"], "penalised_logistic": mods["penalised_logistic"],
           "tf_compact": KerasClassifier(feats, arch="compact", class_weight=True, seed=seed),
           "tf_compact_unweighted": KerasClassifier(feats, arch="compact", class_weight=False, seed=seed)}
    for m, e in est.items():
        e.fit(lab, yl)
        r = np.full((len(test), len(TIMES)), np.nan)
        r[:, TIMES.index(H)] = e.predict_proba(test)[:, 1]
        out[m] = r
        release_memory()
    out["cox_penalised"] = PenCox(feats).fit(train).risk(test, TIMES)
    for m, kw in COXMLP.items():
        cm = CoxMLP(feats, seed=seed, **kw).fit(train, train.os_months.values, train.os_event.values)
        out[m] = 1 - cm.predict_survival(test, TIMES)
        release_memory()
    return out


def metrics(df, risk, t=H, train=None):
    """Uno cumulative/dynamic AUROC, IPCW Brier and Harrell's C at time t."""
    ys = Surv.from_arrays(df.os_event.values.astype(bool), df.os_months.values.astype(float))
    ytr = ys if train is None else train
    auc, _ = cumulative_dynamic_auc(ytr, ys, risk, [t])
    _, bs = brier_score(ytr, ys, 1 - risk, [t])
    c = concordance_index(df.os_months.values, -risk, df.os_event.values)
    km = KaplanMeierFitter().fit(df.os_months, df.os_event)
    return {"td_auroc": float(auc[0]), "brier_ipcw": float(bs[0]), "harrell_c": float(c),
            "mean_predicted_risk": float(np.mean(risk)),
            "km_risk": float(1 - km.survival_function_at_times(t).iloc[0])}


def boot_diff(df, ra, rb, n_boot, seed, t=H):
    ys = Surv.from_arrays(df.os_event.values.astype(bool), df.os_months.values.astype(float))
    rng = np.random.default_rng(seed)
    da, db = [], []
    tmax = df.os_months.values
    while len(da) < n_boot:
        i = rng.integers(0, len(df), len(df))
        yb = ys[i]
        ev = yb["event"] & (yb["time"] <= t)
        if ev.sum() < 2 or (yb["time"] > t).sum() < 2 or yb["time"].max() <= t:
            continue
        try:
            aa = cumulative_dynamic_auc(ys, yb, ra[i], [t])[0][0] - cumulative_dynamic_auc(ys, yb, rb[i], [t])[0][0]
            bb = brier_score(ys, yb, 1 - ra[i], [t])[1][0] - brier_score(ys, yb, 1 - rb[i], [t])[1][0]
        except ValueError:
            continue
        da.append(aa)
        db.append(bb)
    return np.percentile(da, [2.5, 97.5]), np.percentile(db, [2.5, 97.5])


def main():
    configure_tensorflow()
    set_seed()
    tcga = load_cohort("tcga", with_expr=False).reset_index(drop=True)
    feats, _ = resolve_features("B", tcga)
    t, e = tcga.os_months.values, tcga.os_event.values
    strat = np.where((e == 1) & (t <= H), 1, np.where(t >= H, 0, 2))
    R = config.CV_REPEATS_PRIMARY
    names = BINARY + SURV
    ck = OUT / "full_cohort.ckpt.npz"
    oof = {m: np.full((R, len(tcga), len(TIMES)), np.nan) for m in names}
    done = []
    if ck.exists():
        z = np.load(ck)
        done = z["done"].tolist()
        oof = {m: z[f"oof__{m}"] for m in names}
    t0 = time.time()
    for r in range(R):
        if r in done:
            continue
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for k, (tr, te) in enumerate(skf.split(tcga, strat)):
            pr = fit_predict(tcga.iloc[tr].reset_index(drop=True), tcga.iloc[te].reset_index(drop=True),
                             feats, config.SEED + 100 * r + k)
            for m in names:
                oof[m][r, te] = pr[m]
        done.append(r)
        np.savez(ck, done=np.array(done), **{f"oof__{m}": v for m, v in oof.items()})
        log.info("full cohort repeat %d/%d (%.0fs)", r + 1, R, time.time() - t0)
    np.savez(OUT / "full_cohort_tcga.npz", sample=tcga["sample"].values.astype(str),
             **{f"oof__{m}": v for m, v in oof.items()})
    ck.unlink(missing_ok=True)

    rows, diffs = [], []
    j = TIMES.index(H)
    mean = {m: np.nanmean(oof[m][:, :, j], axis=0) for m in names}
    for m in names:
        reps = [metrics(tcga, oof[m][r, :, j])["td_auroc"] for r in range(R)]
        rows.append({"cohort": "tcga", "model": m, "n": len(tcga), "deaths_total": int(e.sum()),
                     **metrics(tcga, mean[m]), "td_auroc_repeat_sd": float(np.std(reps, ddof=1))})
    for m in SURV:
        for tt in (12.0, 60.0):
            mm = metrics(tcga, np.nanmean(oof[m][:, :, TIMES.index(tt)], axis=0), t=tt)
            rows.append({"cohort": "tcga", "model": f"{m}@{int(tt)}", "n": len(tcga),
                         "deaths_total": int(e.sum()), **mm})
    for a, b in [("coxmlp_compact", "cox_penalised"), ("coxmlp_regularised", "cox_penalised"),
                 ("tf_compact", "penalised_logistic"), ("tf_compact_unweighted", "penalised_logistic"),
                 ("cox_penalised", "penalised_logistic")]:
        ca, cb = boot_diff(tcga, mean[a], mean[b], config.N_BOOTSTRAP, config.SEED)
        ma, mb = metrics(tcga, mean[a]), metrics(tcga, mean[b])
        diffs.append({"cohort": "tcga", "comparison": f"{a} - {b}",
                      "delta_td_auroc": ma["td_auroc"] - mb["td_auroc"], "lo": ca[0], "hi": ca[1],
                      "delta_brier": ma["brier_ipcw"] - mb["brier_ipcw"], "brier_lo": cb[0], "brier_hi": cb[1]})
    log.info("internal metrics done")

    for tgt in ("gravendeel", "rembrandt"):
        tdf = load_cohort(tgt, with_expr=False).reset_index(drop=True)
        f2, _ = resolve_features("B", tcga, tdf)
        pr = fit_predict(tcga, tdf, f2, config.SEED)
        np.savez(OUT / f"full_cohort_{tgt}.npz", sample=tdf["sample"].values.astype(str),
                 **{f"pred__{m}": v for m, v in pr.items()})
        ext = {m: pr[m][:, j] for m in names}
        for m in names:
            rows.append({"cohort": tgt, "model": m, "n": len(tdf), "deaths_total": int(tdf.os_event.sum()),
                         **metrics(tdf, ext[m])})
        comps = [("coxmlp_compact", "cox_penalised"), ("coxmlp_regularised", "cox_penalised"),
                 ("tf_compact", "penalised_logistic"), ("tf_compact_unweighted", "penalised_logistic")]
        for a, b in comps:
            ca, cb = boot_diff(tdf, ext[a], ext[b], config.N_BOOTSTRAP, config.SEED)
            ma, mb = metrics(tdf, ext[a]), metrics(tdf, ext[b])
            diffs.append({"cohort": tgt, "comparison": f"{a} - {b}",
                          "delta_td_auroc": ma["td_auroc"] - mb["td_auroc"], "lo": ca[0], "hi": ca[1],
                          "delta_brier": ma["brier_ipcw"] - mb["brier_ipcw"], "brier_lo": cb[0],
                          "brier_hi": cb[1]})
        log.info("external %s done (features: %s)", tgt, ";".join(f2))
    pd.DataFrame(rows).to_csv(config.TABLES / "full_cohort_metrics.csv", index=False)
    pd.DataFrame(diffs).to_csv(config.TABLES / "full_cohort_differences.csv", index=False)


if __name__ == "__main__":
    main()
