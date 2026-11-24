"""Evaluation of the revised primary survival analysis (PROTOCOL.md, section
13), from the predictions of scripts 13 and 17; no model is refitted except the
reference Cox models for diagnostic groups, which are fitted on the same folds.

Estimand. Performance of a single model developed by each procedure on 80% of
TCGA and evaluated on the remaining 20%, averaged over the 25 outer folds
(metrics are computed per repeat on that repeat's out-of-fold predictions and
then averaged over the five repeats). The repeat-averaged ("ensemble")
prediction is reported separately.

Uncertainty. For each repeat, 300 bootstrap resamples of patients are drawn
(seed 20260628 + r) and the paired difference is computed on that repeat's
predictions; the 1500 differences from the five repeats are pooled, so the
percentile interval reflects both patient sampling and variation between
partitions and model fits. External intervals use 1000 bootstrap resamples of
the external cohort for fixed predictions (calibration: the first 300).

Set SURV_CRITERION=pll to evaluate the sensitivity analysis with selection by
held-out partial log-likelihood (outputs carry the suffix _pll).

Outputs (results/tables): survival_horizons.csv, survival_primary_metrics.csv,
survival_primary_differences.csv, survival_primary_calibration.csv,
survival_ph_test.csv, survival_groups.csv, survival_group_pairs.csv,
survival_external_metrics.csv, survival_external_differences.csv,
survival_external_idh_pairs.csv, survival_choices.csv.
"""
import _path  # noqa: F401
import importlib.util

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from src import config
from src.harmonise import load_cohort
from src.preprocessing import resolve_features
from src.survmetrics import brier, cal_slope, km_risk, oe_ratio, pair_concordance, surv, td_auc
from src.utils import get_logger

log = get_logger()
TIMES = (12.0, 24.0, 60.0)
CRIT = __import__("os").environ.get("SURV_CRITERION", "cindex")
SD = config.PREDICTIONS / ("survival" if CRIT == "cindex" else f"survival_{CRIT}")
SUF = "" if CRIT == "cindex" else f"_{CRIT}"
RV = config.PREDICTIONS / "revision"
T = config.TABLES
B_PER_REPEAT = 300
EXT_B = 1000
TUNED = ["cox_linear", "cox_flexible", "cox_mlp"]
FIXED = ["cox_penalised", "coxmlp_compact", "coxmlp_regularised"]
BINARY = ["penalised_logistic", "tf_compact", "tf_compact_unweighted"]
COMPARE = [("cox_mlp", "cox_flexible"), ("cox_mlp", "cox_linear"), ("cox_flexible", "cox_linear"),
           ("coxmlp_compact", "cox_penalised"), ("tf_compact", "penalised_logistic")]

_spec = importlib.util.spec_from_file_location("s17", str(config.ROOT / "scripts" / "17_survival_primary.py"))
s17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s17)


def molecular_group(df):
    idh, cod = df.idh_status.values, df.codel_1p19q.values
    g = np.full(len(df), None, dtype=object)
    g[idh == "WT"] = "IDH-wildtype"
    g[(idh == "Mutant") & (cod == "Codeleted")] = "IDH-mutant, 1p/19q-codeleted"
    g[(idh == "Mutant") & (cod == "Non-codeleted")] = "IDH-mutant, non-codeleted"
    return g


def load_internal(n):
    """risk[model] -> array (R, n, 3); score[model] -> (R, n)."""
    R = config.CV_REPEATS_PRIMARY
    risk = {m: np.full((R, n, 3), np.nan) for m in TUNED + FIXED + BINARY}
    score = {m: np.full((R, n), np.nan) for m in TUNED + FIXED + BINARY}
    for r in range(R):
        z = np.load(SD / f"internal_r{r}.npz")
        for m in TUNED:
            risk[m][r] = z[f"risk__{m}"]
            score[m][r] = z[f"score__{m}"]
    z = np.load(RV / "full_cohort_tcga.npz")
    for m in FIXED + BINARY:
        risk[m] = z[f"oof__{m}"]
        score[m] = z[f"oof__{m}"][:, :, 1]
    return risk, score


def horizons():
    rows = []
    for c in ("tcga", "gravendeel", "rembrandt"):
        d = load_cohort(c, with_expr=False)
        kmc = KaplanMeierFitter().fit(d.os_months, 1 - d.os_event)
        for t in TIMES:
            rows.append({"cohort": c, "horizon": t, "n": len(d),
                         "deaths_by_t": int(((d.os_event == 1) & (d.os_months <= t)).sum()),
                         "censored_before_t": int(((d.os_event == 0) & (d.os_months < t)).sum()),
                         "at_risk_at_t": int((d.os_months >= t).sum()),
                         "km_risk": km_risk(d, t),
                         "censoring_survival_G": float(kmc.survival_function_at_times(t).iloc[0])})
    pd.DataFrame(rows).to_csv(T / f"survival_horizons{SUF}.csv", index=False)


def metrics_row(df, risk3, score, ref=None):
    out = {"harrell_c": concordance_index(df.os_months.values, -score, df.os_event.values)}
    for j, t in enumerate(TIMES):
        r = risk3[:, j]
        if np.isnan(r).all():
            continue
        out[f"td_auc_{int(t)}"] = td_auc(df, r, t, ref)
        out[f"brier_{int(t)}"] = brier(df, r, t, ref)
        out[f"oe_{int(t)}"] = oe_ratio(df, r, t)
        out[f"slope_{int(t)}"] = cal_slope(df, r, t)
    return out


def diff_metrics(df, ra, rb, sa, sb, ref, horizons_ok):
    out = {"harrell_c": concordance_index(df.os_months.values, -sa, df.os_event.values)
           - concordance_index(df.os_months.values, -sb, df.os_event.values)}
    for j, t in enumerate(TIMES):
        if not horizons_ok[j]:
            continue
        out[f"td_auc_{int(t)}"] = td_auc(df, ra[:, j], t, ref) - td_auc(df, rb[:, j], t, ref)
        out[f"brier_{int(t)}"] = brier(df, ra[:, j], t, ref) - brier(df, rb[:, j], t, ref)
    return out


def valid_boot(d, rng, n):
    while True:
        i = rng.integers(0, n, n)
        tt, ee = d.os_months.values[i], d.os_event.values[i]
        if all(((ee == 1) & (tt <= t)).sum() >= 2 and (tt > t).sum() >= 2 for t in TIMES) and tt.max() > max(TIMES):
            return i


def internal_eval(tcga, risk, score):
    R = config.CV_REPEATS_PRIMARY
    ref = surv(tcga)
    rows = []
    for m in TUNED + FIXED + BINARY:
        per = [metrics_row(tcga, risk[m][r], score[m][r], ref) for r in range(R)]
        dfp = pd.DataFrame(per)
        ens = metrics_row(tcga, np.nanmean(risk[m], axis=0), np.nanmean(score[m], axis=0), ref)
        for k in dfp.columns:
            rows.append({"model": m, "metric": k, "mean_over_repeats": dfp[k].mean(), "sd_over_repeats": dfp[k].std(ddof=1),
                         "min": dfp[k].min(), "max": dfp[k].max(), "ensemble": ens.get(k, np.nan)})
    pd.DataFrame(rows).to_csv(T / f"survival_primary_metrics{SUF}.csv", index=False)
    log.info("internal metrics done")
    drows = []
    for a, b in COMPARE:
        ok = [not np.isnan(risk[a][0][:, j]).all() and not np.isnan(risk[b][0][:, j]).all() for j in range(3)]
        point = pd.DataFrame([diff_metrics(tcga, risk[a][r], risk[b][r], score[a][r], score[b][r], ref, ok)
                              for r in range(R)])
        boots = []
        for r in range(R):
            rng = np.random.default_rng(config.SEED + r)
            for _ in range(B_PER_REPEAT):
                i = valid_boot(tcga, rng, len(tcga))
                d = tcga.iloc[i].reset_index(drop=True)
                boots.append(diff_metrics(d, risk[a][r][i], risk[b][r][i], score[a][r][i], score[b][r][i], ref, ok))
        boots = pd.DataFrame(boots)
        ens = diff_metrics(tcga, np.nanmean(risk[a], 0), np.nanmean(risk[b], 0), np.nanmean(score[a], 0),
                           np.nanmean(score[b], 0), ref, ok)
        for k in point.columns:
            drows.append({"comparison": f"{a} - {b}", "metric": k, "delta": point[k].mean(),
                          "sd_over_repeats": point[k].std(ddof=1), "lo": np.percentile(boots[k], 2.5),
                          "hi": np.percentile(boots[k], 97.5), "lo90": np.percentile(boots[k], 5),
                          "hi90": np.percentile(boots[k], 95), "ensemble_delta": ens[k]})
        log.info("differences %s - %s done", a, b)
    pd.DataFrame(drows).to_csv(T / f"survival_primary_differences{SUF}.csv", index=False)
    crows = []
    for m in TUNED:
        bo = []
        for r in range(R):
            rng = np.random.default_rng(config.SEED + 50 + r)
            for _ in range(100):
                i = valid_boot(tcga, rng, len(tcga))
                d = tcga.iloc[i].reset_index(drop=True)
                bo.append({"oe": oe_ratio(d, risk[m][r][i, 1], 24.0), "slope": cal_slope(d, risk[m][r][i, 1], 24.0)})
        bo = pd.DataFrame(bo)
        pt = pd.DataFrame([{"oe": oe_ratio(tcga, risk[m][r][:, 1], 24.0), "slope": cal_slope(tcga, risk[m][r][:, 1], 24.0)}
                           for r in range(R)]).mean()
        crows.append({"model": m, "horizon": 24, "oe": pt.oe, "oe_lo": np.percentile(bo.oe, 2.5), "oe_hi": np.percentile(bo.oe, 97.5),
                      "slope": pt.slope, "slope_lo": np.percentile(bo.slope, 2.5), "slope_hi": np.percentile(bo.slope, 97.5)})
    pd.DataFrame(crows).to_csv(T / f"survival_primary_calibration{SUF}.csv", index=False)
    log.info("calibration intervals done")


def ph_test(tcga):
    feats, _ = resolve_features("B", tcga)
    des = s17.LinearDesign(feats).fit(tcga)
    X = des.transform(tcga)
    names = [n for n, k in zip(des.pre_.get_feature_names_out(), des.keep_) if k]
    d = pd.DataFrame(X, columns=names)
    d["T"], d["E"] = tcga.os_months.values, tcga.os_event.values
    cph = CoxPHFitter(penalizer=0.01).fit(d, "T", "E")
    res = proportional_hazard_test(cph, d, time_transform="rank")
    out = res.summary.reset_index().rename(columns={"index": "term"})
    out["hazard_ratio_per_sd_or_level"] = out["term"].map(np.exp(cph.params_))
    out.to_csv(T / f"survival_ph_test{SUF}.csv", index=False)


def groups_eval(tcga, risk, score):
    """Discrimination across and within molecular groups."""
    R = config.CV_REPEATS_PRIMARY
    g = molecular_group(tcga)
    rows, prow = [], []
    # reference Cox models on the same outer folds: molecular group only; age and group
    ref_scores = {"group_only": np.full((R, len(tcga)), np.nan), "age_and_group": np.full((R, len(tcga)), np.nan)}
    gd = pd.get_dummies(pd.Series(g).fillna("unclassified"), drop_first=False).astype(float)
    for r in range(R):
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for tr, te in skf.split(tcga, s17.strata(tcga)):
            for name, cols in (("group_only", []), ("age_and_group", ["age"])):
                X = gd.copy()
                for c in cols:
                    X[c] = tcga[c].fillna(tcga[c].iloc[tr].median()).values
                X = X.drop(columns=[X.columns[0]])
                d = X.iloc[tr].copy()
                d["T"], d["E"] = tcga.os_months.values[tr], tcga.os_event.values[tr]
                cph = CoxPHFitter(penalizer=0.01).fit(d, "T", "E")
                ref_scores[name][r, te] = np.log(cph.predict_partial_hazard(X.iloc[te]).values.ravel())
    allscores = {**{m: score[m] for m in TUNED + ["coxmlp_compact", "cox_penalised"]}, **ref_scores}
    for m, S in allscores.items():
        per = pd.DataFrame([pair_concordance(tcga.os_months.values, tcga.os_event.values, S[r], g) for r in range(R)])
        prow.append({"model": m, **per.mean().to_dict()})
    pd.DataFrame(prow).to_csv(T / f"survival_group_pairs{SUF}.csv", index=False)
    for grp in ["IDH-wildtype", "IDH-mutant, non-codeleted", "IDH-mutant, 1p/19q-codeleted"]:
        mask = g == grp
        d = tcga[mask].reset_index(drop=True)
        for t in TIMES:
            ev = int(((d.os_event == 1) & (d.os_months <= t)).sum())
            alive = int((d.os_months > t).sum())
            row = {"group": grp, "n": int(mask.sum()), "deaths_total": int(d.os_event.sum()), "horizon": t,
                   "deaths_by_t": ev, "followed_beyond_t": alive}
            if ev >= 10 and alive >= 10:
                for m in TUNED + ["coxmlp_compact", "cox_penalised"]:
                    j = TIMES.index(t)
                    vals = [td_auc(d, risk[m][r][mask, j], t) for r in range(R)]
                    row[f"td_auc_{m}"] = float(np.mean(vals))
            rows.append(row)
    pd.DataFrame(rows).to_csv(T / f"survival_groups{SUF}.csv", index=False)
    log.info("group analysis done")


def external_eval():
    tcga = load_cohort("tcga", with_expr=False)
    rows, drows, prow = [], [], []
    for tgt in ("gravendeel", "rembrandt"):
        tdf = load_cohort(tgt, with_expr=False).reset_index(drop=True)
        z = np.load(SD / f"external_{tgt}.npz")
        assert (z["sample"] == tdf["sample"].values.astype(str)).all()
        variants = ["mode"] + (["model_imputed"] if tgt == "gravendeel" else [])
        subsets = {"all": np.ones(len(tdf), bool)}
        if tgt == "gravendeel":
            subsets["idh_recorded"] = tdf.idh_status.notna().values
        for v in variants:
            for sname, mask in subsets.items():
                if v == "model_imputed" and sname != "all":
                    continue
                d = tdf[mask].reset_index(drop=True)
                rk = {m: z[f"risk__{m}__{v}"][mask] for m in TUNED}
                sc = {m: z[f"score__{m}__{v}"][mask] for m in TUNED}
                for m in TUNED:
                    rows.append({"target": tgt, "idh_handling": v, "subset": sname, "model": m, "n": len(d),
                                 **metrics_row(d, rk[m], sc[m])})
                rng = np.random.default_rng(config.SEED)
                idx = [valid_boot(d, rng, len(d)) for _ in range(EXT_B)]
                for a, b in COMPARE[:3]:
                    pt = diff_metrics(d, rk[a], rk[b], sc[a], sc[b], None, [True] * 3)
                    bo = pd.DataFrame([diff_metrics(d.iloc[i].reset_index(drop=True), rk[a][i], rk[b][i], sc[a][i],
                                                    sc[b][i], surv(d), [True] * 3) for i in idx])
                    for k in pt:
                        drows.append({"target": tgt, "idh_handling": v, "subset": sname, "comparison": f"{a} - {b}",
                                      "metric": k, "delta": pt[k], "lo": np.percentile(bo[k], 2.5),
                                      "hi": np.percentile(bo[k], 97.5)})
                if sname == "all":
                    for m in TUNED:
                        bo = [{"oe": oe_ratio(d.iloc[i].reset_index(drop=True), rk[m][i, 1], 24.0),
                               "slope": cal_slope(d.iloc[i].reset_index(drop=True), rk[m][i, 1], 24.0)} for i in idx[:300]]
                        bo = pd.DataFrame(bo)
                        rows[-len(TUNED) + TUNED.index(m)].update({
                            "oe_24_lo": np.percentile(bo.oe, 2.5), "oe_24_hi": np.percentile(bo.oe, 97.5),
                            "slope_24_lo": np.percentile(bo.slope, 2.5), "slope_24_hi": np.percentile(bo.slope, 97.5)})
                log.info("external %s %s %s done", tgt, v, sname)
        if tgt == "gravendeel":
            grp = np.where(tdf.idh_status.notna(), "recorded", "missing").astype(object)
            for m in TUNED:
                prow.append({"model": m, **pair_concordance(tdf.os_months.values, tdf.os_event.values,
                                                            z[f"score__{m}__mode"], grp)})
    pd.DataFrame(rows).to_csv(T / f"survival_external_metrics{SUF}.csv", index=False)
    pd.DataFrame(drows).to_csv(T / f"survival_external_differences{SUF}.csv", index=False)
    pd.DataFrame(prow).to_csv(T / f"survival_external_idh_pairs{SUF}.csv", index=False)


def choices():
    frames = [pd.read_csv(SD / f"internal_r{r}_choices.csv") for r in range(config.CV_REPEATS_PRIMARY)]
    for tgt in ("gravendeel", "rembrandt"):
        e = pd.read_csv(SD / f"external_{tgt}_choices.csv")
        frames.append(e)
    pd.concat(frames, ignore_index=True).to_csv(T / f"survival_choices{SUF}.csv", index=False)


def main():
    tcga = load_cohort("tcga", with_expr=False).reset_index(drop=True)
    z = np.load(RV / "full_cohort_tcga.npz")
    assert (z["sample"] == tcga["sample"].values.astype(str)).all()
    risk, score = load_internal(len(tcga))
    horizons()
    choices()
    ph_test(tcga)
    groups_eval(tcga, risk, score)
    internal_eval(tcga, risk, score)
    external_eval()


if __name__ == "__main__":
    main()
