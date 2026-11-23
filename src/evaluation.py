"""Performance metrics, calibration, decision curves, bootstrap inference and
the prespecified decision rule."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config

EPS = 1e-7


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p):
    """Logistic recalibration: slope from y ~ a + b*logit(p); calibration-in-the-
    large intercept from y ~ a + offset(logit(p))."""
    import statsmodels.api as sm
    y = np.asarray(y, float)
    lp = _logit(p)
    if len(np.unique(y)) < 2 or np.std(lp) < 1e-10:
        return np.nan, np.nan
    try:
        slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
    except Exception:
        slope = np.nan
    try:
        intercept = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                                 offset=lp).fit().params[0])
    except Exception:
        intercept = np.nan
    return intercept, slope


def point_metrics(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    two = len(np.unique(y)) == 2
    ci, cs = calibration_intercept_slope(y, p)
    return {"n": int(len(y)), "events": int(y.sum()), "event_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)) if two else np.nan,
            "auprc": float(average_precision_score(y, p)) if two else np.nan,
            "brier": float(brier_score_loss(y, p)),
            "calibration_intercept": ci, "calibration_slope": cs}


def fast_auroc(y, p):
    """Mann-Whitney AUROC with mid-ranks for ties (equal to roc_auc_score)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _metric(y, p, m):
    if m == "auroc":
        return fast_auroc(y, p)
    if m == "auprc":
        return average_precision_score(y, p)
    if m == "brier":
        return brier_score_loss(y, np.clip(p, EPS, 1 - EPS))
    raise ValueError(m)


def bootstrap_indices(n, y, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """Fixed bootstrap resamples (skipping single-class resamples), shared by all
    models in a setting so that paired comparisons use identical patients."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            out.append(idx)
    return out


def bootstrap_ci(y, p, metric, idx_list):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    vals = np.array([_metric(y[i], p[i], metric) for i in idx_list])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_difference(y, pa, pb, metric, idx_list):
    """Percentile CI of metric(A) - metric(B) over identical resamples."""
    y = np.asarray(y).astype(int)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    d = np.array([_metric(y[i], pa[i], metric) - _metric(y[i], pb[i], metric) for i in idx_list])
    point = _metric(y, pa, metric) - _metric(y, pb, metric)
    return {"delta": float(point), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5))}


def net_benefit(y, p, thresholds):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    n = len(y)
    out = []
    for t in thresholds:
        pos = p >= t
        tp = np.sum(pos & (y == 1)) / n
        fp = np.sum(pos & (y == 0)) / n
        out.append(tp - fp * t / (1 - t))
    return np.array(out)


def net_benefit_treat_all(y, thresholds):
    prev = float(np.mean(y))
    return np.array([prev - (1 - prev) * t / (1 - t) for t in thresholds])


def net_benefit_difference(y, pa, pb, thresholds, idx_list):
    y = np.asarray(y).astype(int)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    pt = net_benefit(y, pa, thresholds) - net_benefit(y, pb, thresholds)
    boot = np.array([net_benefit(y[i], pa[i], thresholds) - net_benefit(y[i], pb[i], thresholds)
                     for i in idx_list])
    return pt, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def calibration_curve(y, p, n_bins=10):
    """Quantile-binned mean predicted risk and observed proportion."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    return (np.array([p[b].mean() for b in bins]), np.array([y[b].mean() for b in bins]),
            np.array([len(b) for b in bins]))


def decision_rule(delta_auroc, brier_nn, brier_ref, slope_nn) -> dict:
    lo, hi = config.CALIBRATION_SLOPE_RANGE
    a = delta_auroc >= config.MEANINGFUL_AUROC_DELTA - 1e-9
    b = brier_nn <= brier_ref + 1e-9
    c = bool(np.isfinite(slope_nn) and lo <= slope_nn <= hi)
    return {"auroc_margin_met": bool(a), "brier_not_worse": bool(b),
            "calibration_slope_ok": c, "criterion_met": bool(a and b and c)}


def recalibrate_intercept(y, p):
    """Calibration-in-the-large updating: shift logit(p) by the fitted
    intercept of y ~ offset(logit(p)). Uses the evaluation outcomes, so it is a
    model-updating analysis, not a validation estimate."""
    a, _ = calibration_intercept_slope(y, p)
    lp = _logit(p) + (a if np.isfinite(a) else 0.0)
    return 1 / (1 + np.exp(-lp))
