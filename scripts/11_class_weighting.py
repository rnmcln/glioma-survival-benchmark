"""Post hoc analysis R3 (PROTOCOL.md, section 12): class weighting, outcome
prevalence and calibration, computed from stored predictions.

With inverse-frequency weighting (weight n0/n1 on events), the minimiser of the
weighted cross-entropy is p_w with logit(p_w) = logit(p) + log(n0/n1), where p
is the unweighted conditional risk. The expected calibration intercept of a
weighted model is therefore about -log(n0/n1) = logit(training prevalence).
This script (a) tabulates, for every internal and external setting and every
network, the event rate, the calibration intercept and slope, Brier score and
AUROC with and without weighting; (b) applies the analytic prior-shift
correction logit(p) = logit(p_w) - log(n0/n1), using the event frequency of
each training fold (training data only); (c) reports net benefit at the
plausible thresholds; and (d) recomputes decision-curve differences in the
external cohorts after intercept updating of both models (model updating).
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import config
from src.cv import load
from src.evaluation import (_logit, bootstrap_indices, net_benefit, net_benefit_difference,
                            paired_difference, point_metrics, recalibrate_intercept)
from src.models import MLPS
from src.settings import EXTERNAL_LABELS, internal_settings
from src.utils import get_logger, load_json

log = get_logger()


def expit(x):
    return 1 / (1 + np.exp(-x))


def corrected_internal(y, per_rep):
    """Prior-shift correction with training-fold prevalence (folds regenerated
    exactly as in src/cv.py: StratifiedKFold(5, shuffle, seed + r) on y)."""
    out = np.full_like(per_rep, np.nan)
    for r in range(per_rep.shape[0]):
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for tr, te in skf.split(np.zeros(len(y)), y):
            pi = y[tr].mean()
            out[r, te] = expit(_logit(per_rep[r, te]) + np.log(pi / (1 - pi)))
    return np.nanmean(out, axis=0)


def main():
    rows, nb_rows, ddca = [], [], []
    udir = config.PREDICTIONS / "unweighted"
    tcga_pi = load_json(config.PREDICTIONS / "ext_gravendeel_B_2y.json")["train_endpoint"]["event_rate"]
    tags = [t for t, *_ in internal_settings()] + list(EXTERNAL_LABELS)
    th = config.DCA_REPORT_THRESHOLDS
    for tag in tags:
        y, mean, per, _ = load(tag)
        _, mu, _, _ = load(tag, udir)
        idx = bootstrap_indices(len(y), y)
        internal = tag.startswith("int_")
        train_pi = y.mean() if internal else tcga_pi
        for m in MLPS:
            pc = corrected_internal(y, per[m]) if internal else \
                expit(_logit(mean[m]) + np.log(tcga_pi / (1 - tcga_pi)))
            versions = {"weighted": mean[m], "unweighted": mu[m], "weighted_corrected": pc}
            for v, p in versions.items():
                pm = point_metrics(y, p)
                rows.append({"setting": tag, "network": m, "version": v, "n": pm["n"], "events": pm["events"],
                             "event_rate": pm["event_rate"], "logit_train_prevalence": float(np.log(train_pi / (1 - train_pi))),
                             "auroc": pm["auroc"], "brier": pm["brier"],
                             "calibration_intercept": pm["calibration_intercept"],
                             "calibration_slope": pm["calibration_slope"], "mean_predicted": float(np.mean(p))})
            db = paired_difference(y, mu[m], mean[m], "brier", idx)
            da = paired_difference(y, mu[m], mean[m], "auroc", idx)
            rows.append({"setting": tag, "network": m, "version": "unweighted_minus_weighted",
                         "brier": db["delta"], "brier_lo": db["lo"], "brier_hi": db["hi"],
                         "auroc": da["delta"], "auroc_lo": da["lo"], "auroc_hi": da["hi"]})
        if tag in ("int_B_1y", "int_B_2y", "int_B_5y", "ext_gravendeel_B_2y"):
            pc = corrected_internal(y, per["tf_compact"]) if internal else \
                expit(_logit(mean["tf_compact"]) + np.log(tcga_pi / (1 - tcga_pi)))
            for v, p in {"penalised_logistic": mean["penalised_logistic"], "tf_compact": mean["tf_compact"],
                         "tf_compact_unweighted": mu["tf_compact"], "tf_compact_corrected": pc}.items():
                for t, nb, frac in zip(th, net_benefit(y, p, th), [np.mean(p >= t) for t in th]):
                    nb_rows.append({"setting": tag, "model": v, "threshold": t, "net_benefit": nb,
                                    "fraction_high_risk": frac})
        if tag in ("ext_gravendeel_B_2y", "ext_rembrandt_B_2y"):
            d0 = pd.read_csv(config.TABLES / "decision_rule.csv").set_index("setting").loc[tag]
            nn = d0.best_nn
            pa, pb = recalibrate_intercept(y, mean[nn]), recalibrate_intercept(y, mean["penalised_logistic"])
            d, lo, hi = net_benefit_difference(y, pa, pb, th, idx)
            ddca.append(pd.DataFrame({"setting": tag, "comparison": f"{nn} - penalised_logistic (both intercept-updated)",
                                      "threshold": th, "delta_nb": d, "lo": lo, "hi": hi}))
        log.info("class weighting %s", tag)
    T = config.TABLES
    pd.DataFrame(rows).to_csv(T / "class_weighting.csv", index=False)
    pd.DataFrame(nb_rows).to_csv(T / "class_weighting_net_benefit.csv", index=False)
    pd.concat(ddca).to_csv(T / "decision_curve_difference_updated.csv", index=False)


if __name__ == "__main__":
    main()
