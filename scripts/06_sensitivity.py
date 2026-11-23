"""Sensitivity analyses.

(a) Inverse-probability-of-censoring weighting (IPCW) for the primary setting:
    patients censored before 24 months are represented by up-weighting
    comparable labelled patients, using the Kaplan-Meier estimate of the
    censoring distribution in the full TCGA cohort (marginal weights).
(b) Nested cross-validation of penalised logistic regression (inner 3-fold grid
    search over C and the L1 ratio) on the primary-setting folds.
(c) Expression scaling for external transfer: within-cohort standardisation
    (default) versus applying the TCGA mean and standard deviation to the target.
(d) Final networks fitted on all labelled TCGA patients (set B, 2 years):
    training histories and number of epochs.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from src import config
from src.cv import load
from src.endpoints import make_binary_endpoint
from src.evaluation import fast_auroc, point_metrics
from src.harmonise import load_cohort
from src.models import KerasClassifier, build_models
from src.preprocessing import make_preprocessor, resolve_features
from src.utils import configure_tensorflow, get_logger, save_json, set_seed

log = get_logger()


def ipcw():
    y, mean, _, sample = load("int_B_2y")
    t = load_cohort("tcga", with_expr=False)
    km = KaplanMeierFitter().fit(t.os_months, event_observed=1 - t.os_event)
    lab = pd.DataFrame({"sample": sample}).merge(t[["sample", "os_months", "os_event"]], on="sample", how="left")
    assert lab.os_months.notna().all()
    def G(x, left=True):
        # G(t-) for events, G(t) for administrative horizon
        return float(km.survival_function_at_times(np.nextafter(x, -np.inf) if left else x).iloc[0])
    w = np.where((lab.os_event == 1) & (lab.os_months <= 24),
                 [1 / G(v) for v in lab.os_months], 1 / G(24.0, left=False))
    rows = []
    from sklearn.metrics import brier_score_loss, roc_auc_score
    for m, p in mean.items():
        rows.append({"model": m, "auroc_unweighted": point_metrics(y, p)["auroc"],
                     "auroc_ipcw": roc_auc_score(y, p, sample_weight=w),
                     "brier_unweighted": point_metrics(y, p)["brier"],
                     "brier_ipcw": brier_score_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=w)})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(config.TABLES / "ipcw_primary.csv", index=False)
    save_json({"G_24": G(24.0, left=False), "weight_min": float(w.min()), "weight_max": float(w.max())},
              config.METRICS / "ipcw_weights.json")
    log.info("IPCW done (G(24)=%.3f)", G(24.0, left=False))


def nested():
    tcga = load_cohort("tcga", with_expr=False)
    feats, _ = resolve_features("B", tcga)
    lab, _ = make_binary_endpoint(tcga, 24)
    lab = lab.reset_index(drop=True)
    y = lab.y.values
    _, _, per, _ = load("int_B_2y")
    rows = []
    for r in range(config.CV_REPEATS_PRIMARY):
        oof = np.full(len(y), np.nan)
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for tr, te in skf.split(lab, y):
            pipe = Pipeline([("pre", make_preprocessor(lab, feats)),
                             ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000,
                                                        random_state=config.SEED))])
            gs = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.2, 0.5, 0.8]},
                              scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=config.SEED))
            gs.fit(lab.iloc[tr], y[tr])
            oof[te] = gs.predict_proba(lab.iloc[te])[:, 1]
        rows.append({"repeat": r, "nested_auroc": fast_auroc(y, oof),
                     "fixed_auroc": fast_auroc(y, per["penalised_logistic"][r])})
    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES / "nested_cv.csv", index=False)
    log.info("nested CV: %.3f vs fixed %.3f", out.nested_auroc.mean(), out.fixed_auroc.mean())


def expr_scaling():
    raw = {c: pd.read_csv(config.PROCESSED / f"expr_panel_{c}.csv", dtype={"Sample": str})
           for c in ("tcga", "gravendeel", "rembrandt")}
    clin = {c: pd.read_csv(config.PROCESSED / f"clinical_{c}.csv", dtype={"sample": str}) for c in raw}
    genes = [g for g in raw["tcga"].columns if g != "Sample"]
    mu, sd = raw["tcga"][genes].mean(), raw["tcga"][genes].std(ddof=0)
    def merged(c, scheme):
        e = raw[c].copy()
        if scheme == "within_cohort" or c == "tcga":
            e[genes] = (e[genes] - e[genes].mean()) / e[genes].std(ddof=0)
        else:
            e[genes] = (e[genes] - mu) / sd
        d = clin[c].merge(e.rename(columns={"Sample": "sample"}), on="sample")
        return make_binary_endpoint(d, 24)[0].reset_index(drop=True)
    rows = []
    for tgt in ("gravendeel", "rembrandt"):
        for scheme in ("within_cohort", "tcga_reference"):
            tr, te = merged("tcga", scheme), merged(tgt, scheme)
            models = build_models(tr, genes, include_nn=True)
            for m in ("penalised_logistic", "tf_regularised"):
                models[m].fit(tr, tr.y.values)
                pm = point_metrics(te.y.values, models[m].predict_proba(te)[:, 1])
                rows.append({"target": tgt, "scheme": scheme, "model": m, "auroc": pm["auroc"],
                             "brier": pm["brier"], "calibration_slope": pm["calibration_slope"]})
    pd.DataFrame(rows).round(3).to_csv(config.TABLES / "expression_scaling.csv", index=False)
    log.info("expression scaling done")


def final_networks():
    tcga = load_cohort("tcga", with_expr=False)
    feats, _ = resolve_features("B", tcga)
    lab, _ = make_binary_endpoint(tcga, 24)
    hist, rows = {}, []
    for arch in ("compact", "medium", "regularised"):
        clf = KerasClassifier(feats, arch=arch).fit(lab, lab.y.values)
        hist[arch] = clf.history_
        best = int(np.argmin(clf.history_["val_loss"])) + 1
        rows.append({"network": f"tf_{arch}", "encoded_inputs": clf.n_input_,
                     "parameters": int(clf.model_.count_params()),
                     "epochs_run": len(clf.history_["loss"]), "best_epoch": best})
    save_json(hist, config.METRICS / "final_network_histories.json")
    pd.DataFrame(rows).to_csv(config.TABLES / "final_networks.csv", index=False)
    log.info("final networks done")


if __name__ == "__main__":
    configure_tensorflow()
    set_seed()
    ipcw()
    nested()
    expr_scaling()
    final_networks()
