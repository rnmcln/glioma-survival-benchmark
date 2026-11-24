"""Metrics, bootstrap intervals, the planned decision rule and decision
curves for every setting; writes results/tables/*.csv and a JSON summary.

Selection within each setting: the best of the three multilayer perceptrons and
the best transparent model (age + grade LR or penalised LR) are each chosen by
point AUROC; their difference is estimated over 2000 paired bootstrap
resamples of the evaluation patients.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from src import config
from src.cv import load
from src.evaluation import (recalibrate_intercept, bootstrap_ci, bootstrap_indices, decision_rule, net_benefit,
                            net_benefit_difference, net_benefit_treat_all, paired_difference,
                            point_metrics)
from src.models import LABELS, MLPS, TRANSPARENT
from src.settings import EXTERNAL_LABELS, internal_settings
from src.utils import get_logger, load_json, save_json

log = get_logger()


def settings():
    s = [(t, lab, "internal") for t, lab, *_ in internal_settings()]
    s += [(t, lab, "external") for t, lab in EXTERNAL_LABELS.items()]
    return s


def performance(tag, label, kind, idx, y, mean, per):
    rows = []
    for m, p in mean.items():
        pm = point_metrics(y, p)
        alo, ahi = bootstrap_ci(y, p, "auroc", idx)
        blo, bhi = bootstrap_ci(y, p, "brier", idx)
        rep = [point_metrics(y, per[m][r])["auroc"] for r in range(per[m].shape[0])] if kind == "internal" else []
        rows.append({"setting": tag, "label": label, "model": m, "model_label": LABELS[m],
                     "n": pm["n"], "events": pm["events"], "event_rate": pm["event_rate"],
                     "auroc": pm["auroc"], "auroc_lo": alo, "auroc_hi": ahi,
                     "auroc_repeat_sd": float(np.std(rep, ddof=1)) if len(rep) > 1 else np.nan,
                     "auprc": pm["auprc"], "brier": pm["brier"], "brier_lo": blo, "brier_hi": bhi,
                     "calibration_intercept": pm["calibration_intercept"],
                     "calibration_slope": pm["calibration_slope"],
                     "brier_after_intercept_update": (point_metrics(y, recalibrate_intercept(y, p))["brier"]
                                                      if kind == "external" else np.nan)})
    return rows


def decide(tag, label, y, mean_ref, mean_nn, idx):
    au = {m: point_metrics(y, mean_ref[m])["auroc"] for m in TRANSPARENT}
    ref = max(au, key=au.get)
    aun = {m: point_metrics(y, mean_nn[m])["auroc"] for m in MLPS}
    nn = max(aun, key=aun.get)
    pr, pn = point_metrics(y, mean_ref[ref]), point_metrics(y, mean_nn[nn])
    d = paired_difference(y, mean_nn[nn], mean_ref[ref], "auroc", idx)
    db = paired_difference(y, mean_nn[nn], mean_ref[ref], "brier", idx)
    rule = decision_rule(d["delta"], pn["brier"], pr["brier"], pn["calibration_slope"])
    return {"setting": tag, "label": label, "n": pn["n"], "events": pn["events"],
            "best_nn": nn, "best_transparent": ref,
            "auroc_nn": pn["auroc"], "auroc_ref": pr["auroc"],
            "delta_auroc": d["delta"], "delta_lo": d["lo"], "delta_hi": d["hi"],
            "brier_nn": pn["brier"], "brier_ref": pr["brier"],
            "delta_brier": db["delta"], "delta_brier_lo": db["lo"], "delta_brier_hi": db["hi"],
            "slope_nn": pn["calibration_slope"], "slope_ref": pr["calibration_slope"],
            "intercept_nn": pn["calibration_intercept"], "intercept_ref": pr["calibration_intercept"],
            **rule}


def dca(tag, y, mean, nn, idx):
    th = config.DCA_REPORT_THRESHOLDS
    rows = []
    all_ = net_benefit_treat_all(y, th)
    for m in ("penalised_logistic", "hist_gboost", nn):
        nb = net_benefit(y, mean[m], th)
        for t, v, a in zip(th, nb, all_):
            rows.append({"setting": tag, "model": m, "threshold": t, "net_benefit": v, "treat_all": a})
    d, lo, hi = net_benefit_difference(y, mean[nn], mean["penalised_logistic"], th, idx)
    diff = pd.DataFrame({"setting": tag, "comparison": f"{nn} - penalised_logistic", "threshold": th,
                         "delta_nb": d, "lo": lo, "hi": hi})
    return rows, diff


def main():
    perf, dec, dec_u, dca_rows, dca_diff, ftt = [], [], [], [], [], []
    udir = config.PREDICTIONS / "unweighted"
    for tag, label, kind in settings():
        y, mean, per, _ = load(tag)
        idx = bootstrap_indices(len(y), y)
        perf += performance(tag, label, kind, idx, y, mean, per)
        d = decide(tag, label, y, mean, mean, idx)
        dec.append(d)
        if (udir / f"{tag}.npz").exists():
            yu, mu, peru, _ = load(tag, udir)
            assert (yu == y).all()
            perf_u = performance(tag, label, kind, idx, yu, mu, peru)
            for r in perf_u:
                r["model"] += "_unweighted"
            perf += perf_u
            dec_u.append(decide(tag, label, y, mean, mu, idx))
        if tag in ("int_B_2y", "ext_gravendeel_B_2y", "ext_rembrandt_B_2y"):
            r, df = dca(tag, y, mean, d["best_nn"], idx)
            dca_rows += r
            dca_diff.append(df)
        if "ft_transformer" in mean:
            for comp in ("penalised_logistic", d["best_nn"]):
                pdiff = paired_difference(y, mean["ft_transformer"], mean[comp], "auroc", idx)
                ftt.append({"setting": tag, "comparator": comp, "auroc_ftt": point_metrics(y, mean["ft_transformer"])["auroc"],
                            "auroc_comparator": point_metrics(y, mean[comp])["auroc"], **pdiff})
        log.info("tabulated %s", tag)
    T = config.TABLES
    pd.DataFrame(perf).to_csv(T / "performance_all.csv", index=False)
    dec = pd.DataFrame(dec)
    dec.to_csv(T / "decision_rule.csv", index=False)
    if dec_u:
        pd.DataFrame(dec_u).to_csv(T / "decision_rule_unweighted.csv", index=False)
    pd.DataFrame(dca_rows).to_csv(T / "decision_curve.csv", index=False)
    pd.concat(dca_diff).to_csv(T / "decision_curve_difference.csv", index=False)
    pd.DataFrame(ftt).to_csv(T / "ft_transformer.csv", index=False)
    summary = {"settings": len(dec), "criterion_met": int(dec.criterion_met.sum()),
               "nn_auroc_higher": int((dec.delta_auroc > 0).sum()),
               "auroc_margin_met": dec.loc[dec.auroc_margin_met, "setting"].tolist()}
    if dec_u:
        du = pd.DataFrame(dec_u)
        summary.update({"unweighted_criterion_met": int(du.criterion_met.sum()),
                        "unweighted_settings": len(du)})
    save_json(summary, config.METRICS / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
