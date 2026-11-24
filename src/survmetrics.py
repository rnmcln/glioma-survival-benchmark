"""Survival performance measures used in the revised analyses.

td_auc      cumulative/dynamic AUROC at time t with inverse-probability-of-
            censoring weights (Uno et al.), scikit-survival implementation;
            the censoring distribution is the Kaplan-Meier estimate in the
            evaluation cohort.
brier       IPCW Brier score at time t (Graf/Gerds), same censoring estimate.
oe_ratio    observed/expected: Kaplan-Meier risk of death by t divided by the
            mean predicted risk by t (calibration-in-the-large).
cal_slope   coefficient of cloglog(predicted risk by t) in a Cox model with
            follow-up administratively censored at t (calibration slope).
pair_concordance
            Harrell's C split into comparable pairs within and between groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.util import Surv


def surv(df):
    return Surv.from_arrays(df.os_event.values.astype(bool), df.os_months.values.astype(float))


def td_auc(df, risk, t, ref=None):
    y = surv(df)
    return float(cumulative_dynamic_auc(y if ref is None else ref, y, risk, [t])[0][0])


def brier(df, risk, t, ref=None):
    y = surv(df)
    return float(brier_score(y if ref is None else ref, y, 1 - np.asarray(risk), [t])[1][0])


def km_risk(df, t):
    km = KaplanMeierFitter().fit(df.os_months, df.os_event)
    return float(1 - km.survival_function_at_times(t).iloc[0])


def oe_ratio(df, risk, t):
    return km_risk(df, t) / float(np.mean(risk))


def cal_slope(df, risk, t):
    r = np.clip(np.asarray(risk, float), 1e-6, 1 - 1e-6)
    d = pd.DataFrame({"lp": np.log(-np.log(1 - r)),
                      "T": np.minimum(df.os_months.values, t),
                      "E": ((df.os_event.values == 1) & (df.os_months.values <= t)).astype(int)})
    return float(CoxPHFitter().fit(d, "T", "E").params_["lp"])


def pair_concordance(time, event, score, group):
    """Harrell's C over comparable pairs (i has the event and t_i < t_j),
    overall, within groups and between groups. Ties in score count 0.5.
    Pairs involving a missing group label are counted only overall."""
    t, e, s = np.asarray(time, float), np.asarray(event, int), np.asarray(score, float)
    g = np.asarray(group, object)
    res = {k: [0.0, 0] for k in ("overall", "within", "between")}
    for i in np.where(e == 1)[0]:
        j = t > t[i]
        if not j.any():
            continue
        conc = (s[i] > s[j]).astype(float) + 0.5 * (s[i] == s[j])
        res["overall"][0] += conc.sum()
        res["overall"][1] += j.sum()
        if g[i] is None or (isinstance(g[i], float) and np.isnan(g[i])):
            continue
        gj = g[j]
        valid = np.array([x is not None and not (isinstance(x, float) and np.isnan(x)) for x in gj])
        same = valid & (gj == g[i])
        diff = valid & (gj != g[i])
        res["within"][0] += conc[same].sum()
        res["within"][1] += same.sum()
        res["between"][0] += conc[diff].sum()
        res["between"][1] += diff.sum()
    out = {f"c_{k}": (v[0] / v[1] if v[1] else np.nan) for k, v in res.items()}
    out.update({f"pairs_{k}": v[1] for k, v in res.items()})
    return out


def auroc_pairs(y, p, group):
    """AUROC split into event/non-event pairs within and between groups."""
    y, p, g = np.asarray(y, int), np.asarray(p, float), np.asarray(group, object)
    res = {k: [0.0, 0] for k in ("overall", "within", "between")}
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for i in pos:
        conc = (p[i] > p[neg]).astype(float) + 0.5 * (p[i] == p[neg])
        same = g[neg] == g[i]
        res["overall"][0] += conc.sum()
        res["overall"][1] += len(neg)
        res["within"][0] += conc[same].sum()
        res["within"][1] += same.sum()
        res["between"][0] += conc[~same].sum()
        res["between"][1] += (~same).sum()
    out = {f"auc_{k}": (v[0] / v[1] if v[1] else np.nan) for k, v in res.items()}
    out.update({f"pairs_{k}": v[1] for k, v in res.items()})
    return out
