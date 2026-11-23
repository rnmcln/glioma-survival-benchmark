"""Post hoc analysis R4a (PROTOCOL.md, section 12): covariate-dependent
inverse-probability-of-censoring weights for the primary binary analysis.

The prespecified IPCW analysis used the marginal Kaplan-Meier censoring
distribution. Here the censoring distribution is modelled by a Cox model for
the censoring hazard (event = censored) on the set B predictors in the full
TCGA cohort (median/mode imputation, one-hot encoding, ridge penalty 0.01).
Weights: 1/G(T_i- | x_i) for deaths by 24 months and 1/G(24 | x_i) for
patients alive at 24 months. Weighted AUROC and Brier scores are reported for
every model, and the paired weighted AUROC difference between the compact
network and penalised logistic regression is bootstrapped (2000 resamples).
A sensitivity analysis truncates weights at their 99th percentile.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import brier_score_loss, roc_auc_score
from src import config
from src.cv import load
from src.evaluation import bootstrap_indices
from src.harmonise import load_cohort
from src.preprocessing import make_preprocessor, resolve_features
from src.utils import get_logger, save_json

log = get_logger()
H = 24.0


def weights(lab, tcga, feats):
    pre = make_preprocessor(tcga, feats).fit(tcga[feats])
    X = pre.transform(tcga[feats])
    keep = X.std(axis=0) > 1e-8
    cols = [f"x{i}" for i in range(keep.sum())]
    d = pd.DataFrame(X[:, keep], columns=cols)
    d["T"], d["C"] = tcga.os_months.values, 1 - tcga.os_event.values
    cox = CoxPHFitter(penalizer=0.01).fit(d, "T", "C")
    Xl = pd.DataFrame(pre.transform(lab[feats])[:, keep], columns=cols)
    dead = (lab.os_event.values == 1) & (lab.os_months.values <= H)
    t_eval = np.where(dead, np.nextafter(lab.os_months.values, -np.inf), H)
    sf = cox.predict_survival_function(Xl, times=np.unique(t_eval))
    G = np.array([sf.loc[t, i] for i, t in enumerate(t_eval)])
    km = KaplanMeierFitter().fit(tcga.os_months, event_observed=1 - tcga.os_event)
    Gm = np.array([float(km.survival_function_at_times(t).iloc[0]) for t in t_eval])
    return 1 / G, 1 / Gm, cox


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


def main():
    y, mean, _, sample = load("int_B_2y")
    tcga = load_cohort("tcga", with_expr=False)
    feats, _ = resolve_features("B", tcga)
    lab = pd.DataFrame({"sample": sample}).merge(tcga, on="sample", how="left")
    assert len(lab) == len(y) and lab.os_months.notna().all()
    w_cond, w_marg, cox = weights(lab, tcga, feats)
    w_trunc = np.minimum(w_cond, np.percentile(w_cond, 99))
    rows = []
    for m, p in mean.items():
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        rows.append({"model": m, "auroc_unweighted": roc_auc_score(y, p),
                     "auroc_ipcw_marginal": wauc(y, p, w_marg), "auroc_ipcw_conditional": wauc(y, p, w_cond),
                     "auroc_ipcw_conditional_truncated": wauc(y, p, w_trunc),
                     "brier_unweighted": brier_score_loss(y, pc),
                     "brier_ipcw_marginal": brier_score_loss(y, pc, sample_weight=w_marg),
                     "brier_ipcw_conditional": brier_score_loss(y, pc, sample_weight=w_cond)})
    idx = bootstrap_indices(len(y), y)
    diffs = []
    for a, b in (("tf_compact", "penalised_logistic"), ("tf_regularised", "penalised_logistic")):
        for lab_w, w in (("marginal", w_marg), ("conditional", w_cond)):
            pt = wauc(y, mean[a], w) - wauc(y, mean[b], w)
            bt = [wauc(y[i], mean[a][i], w[i]) - wauc(y[i], mean[b][i], w[i]) for i in idx]
            diffs.append({"comparison": f"{a} - {b}", "weights": lab_w, "delta": pt,
                          "lo": np.percentile(bt, 2.5), "hi": np.percentile(bt, 97.5)})
    pd.DataFrame(rows).to_csv(config.TABLES / "ipcw_conditional.csv", index=False)
    pd.DataFrame(diffs).to_csv(config.TABLES / "ipcw_conditional_differences.csv", index=False)
    hr = cox.summary[["exp(coef)", "p"]]
    save_json({"weight_conditional_min": float(w_cond.min()), "weight_conditional_max": float(w_cond.max()),
               "weight_conditional_p99": float(np.percentile(w_cond, 99)),
               "weight_marginal_min": float(w_marg.min()), "weight_marginal_max": float(w_marg.max()),
               "censoring_model_concordance": float(cox.concordance_index_),
               "n_labelled": int(len(y))}, config.METRICS / "ipcw_conditional.json")
    log.info("conditional IPCW done; censoring-model C %.3f", cox.concordance_index_)
    del hr


if __name__ == "__main__":
    main()
