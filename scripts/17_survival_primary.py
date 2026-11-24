"""Revised primary analysis S1 (PROTOCOL.md, section 13): survival models on
all TCGA patients with model selection confined to the training data.

Design. All 664 TCGA patients, predictor set B. Outer loop: 5 repeats of
stratified 5-fold cross-validation (strata: died by 24 months, alive at 24
months, censored before 24 months; seeds 20260628 + r), identical to the folds
of post hoc analysis R4. Inner loop: within each outer training fold, every
modelling procedure selects its hyperparameters by stratified 3-fold
cross-validation (stratified by the event indicator, seed 20260628), using
Harrell's C of the risk score on the inner validation folds; the selected
configuration is refitted on the whole outer training fold.

Procedures (all fitted to identical training data):
  cox_linear    elastic-net Cox (L1 ratio 0.5) on the original design
                (age and grade linear); penaliser in {0.01, 0.03, 0.1, 0.3, 1, 3}
  cox_flexible  ridge Cox on the flexible design (src/flexible.py: age spline,
                categorical grade, age x IDH and grade x IDH interactions);
                penaliser in {0.03, 0.1, 0.3, 1, 3, 10}
  cox_mlp       multilayer perceptron trained on the Cox partial likelihood
                (src/nn_extra.CoxMLP); four configurations:
                32-16 dropout 0.2; 32-16 dropout 0.2 L2 1e-3;
                64-32 dropout 0.4 L2 1e-3; 128-64 dropout 0.3
Survival probabilities use the Breslow baseline cumulative hazard estimated in
the outer training fold (lifelines for the Cox models, nn_extra for the
network). Predicted risks of death by 12, 24 and 60 months and the risk score
are stored per repeat.

External: each procedure is tuned by 3-fold cross-validation on all TCGA
patients and refitted on all TCGA patients with the predictors shared with the
target cohort (common-predictor transport), then applied unchanged to every
Gravendeel and REMBRANDT patient. For Gravendeel, predictions are also made
with missing IDH status imputed from a logistic model of IDH status on age,
grade and histology fitted in TCGA (training data only).

Selection criterion: Harrell's C (primary, defined before the results were
seen) or, as a sensitivity analysis added after the primary results showed
calibration slopes above 1 for the penalised Cox models, the held-out Cox
partial log-likelihood (--criterion pll; outputs in predictions/survival_pll).

Usage: 17_survival_primary.py --repeats 0 1 2   (internal, selected repeats)
       17_survival_primary.py --external [--criterion pll]
"""
import _path  # noqa: F401
import argparse
import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src import config
from src.cv import release_memory
from src.flexible import FlexibleDesign
from src.harmonise import load_cohort
from src.nn_extra import CoxMLP
from src.preprocessing import make_preprocessor, resolve_features
from src.utils import configure_tensorflow, get_logger, save_json, set_seed

log = get_logger()
def outdir(criterion):
    d = config.PREDICTIONS / ("survival" if criterion == "cindex" else f"survival_{criterion}")
    d.mkdir(parents=True, exist_ok=True)
    return d
TIMES = (12.0, 24.0, 60.0)
PEN_GRID_LINEAR = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
PEN_GRID_FLEX = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
MLP_GRID = [dict(hidden=(32, 16), dropout=0.2, l2=0.0), dict(hidden=(32, 16), dropout=0.2, l2=1e-3),
            dict(hidden=(64, 32), dropout=0.4, l2=1e-3), dict(hidden=(128, 64), dropout=0.3, l2=0.0)]
PROCS = ["cox_linear", "cox_flexible", "cox_mlp"]


class LinearDesign:
    def __init__(self, features):
        self.features = features

    def fit(self, df):
        self.pre_ = make_preprocessor(df, self.features).fit(df[self.features])
        X = self.pre_.transform(df[self.features])
        self.keep_ = X.std(axis=0) > 1e-8
        return self

    def transform(self, df):
        return self.pre_.transform(df[self.features])[:, self.keep_]


class CoxModel:
    """lifelines Cox model on a design matrix built by `design`."""

    def __init__(self, design, penalizer, l1_ratio):
        self.design, self.penalizer, self.l1_ratio = design, penalizer, l1_ratio

    def fit(self, df):
        X = self.design.fit(df).transform(df)
        d = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        d["T"], d["E"] = df.os_months.values, df.os_event.values
        self.m_ = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio).fit(d, "T", "E")
        return self

    def _X(self, df):
        X = self.design.transform(df)
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    def score(self, df):
        return np.log(self.m_.predict_partial_hazard(self._X(df)).values.ravel())

    def risk(self, df, times=TIMES):
        return 1 - self.m_.predict_survival_function(self._X(df), times=list(times)).values.T


class MLPModel:
    def __init__(self, features, seed, **kw):
        self.m = CoxMLP(features, seed=seed, **kw)

    def fit(self, df):
        self.m.fit(df, df.os_months.values, df.os_event.values)
        return self

    def score(self, df):
        return self.m.predict_risk(df)

    def risk(self, df, times=TIMES):
        return 1 - self.m.predict_survival(df, times)


def candidates(proc, feats, seed):
    if proc == "cox_linear":
        return [({"penalizer": p}, lambda p=p: CoxModel(LinearDesign(feats), p, 0.5)) for p in PEN_GRID_LINEAR]
    if proc == "cox_flexible":
        return [({"penalizer": p}, lambda p=p: CoxModel(FlexibleDesign(feats), p, 0.0)) for p in PEN_GRID_FLEX]
    return [({"config": i}, lambda g=g: MLPModel(feats, seed, **g)) for i, g in enumerate(MLP_GRID)]


def cindex(df, s):
    return concordance_index(df.os_months.values, -s, df.os_event.values)


def heldout_pll(df, s):
    """Mean Cox partial log-likelihood (Breslow ties) of risk score s on the
    held-out data, per event (higher is better)."""
    t, e, s = df.os_months.values, df.os_event.values, np.asarray(s, float)
    tot = 0.0
    for i in np.where(e == 1)[0]:
        rs = s[t >= t[i]]
        m = rs.max()
        tot += s[i] - (m + np.log(np.exp(rs - m).sum()))
    return tot / max(e.sum(), 1)


CRITERION = {"cindex": cindex, "pll": heldout_pll}


def tune_fit(proc, train, feats, seed, criterion="cindex"):
    """Inner 3-fold selection by Harrell's C (primary) or by held-out partial
    log-likelihood (sensitivity analysis); refit on all of `train`."""
    crit = CRITERION[criterion]
    inner = StratifiedKFold(3, shuffle=True, random_state=config.SEED)
    cands = candidates(proc, feats, seed)
    scores = []
    for _, make in cands:
        s = []
        for a, b in inner.split(train, train.os_event.astype(int)):
            ta, tb = train.iloc[a].reset_index(drop=True), train.iloc[b].reset_index(drop=True)
            try:
                s.append(crit(tb, make().fit(ta).score(tb)))
            except Exception as e:  # convergence failure counts as the worst score
                log.info("fit failed (%s): %s", proc, e)
                s.append(-np.inf)
            release_memory()
        scores.append(float(np.mean(s)))
    k = int(np.argmax(scores))
    model = cands[k][1]().fit(train)
    return model, {**cands[k][0], f"inner_{criterion}": scores[k]}


def strata(df, H=24.0):
    t, e = df.os_months.values, df.os_event.values
    return np.where((e == 1) & (t <= H), 1, np.where(t >= H, 0, 2))


def internal(repeats, criterion="cindex"):
    OUT = outdir(criterion)
    tcga = load_cohort("tcga", with_expr=False).reset_index(drop=True)
    feats, _ = resolve_features("B", tcga)
    for r in repeats:
        out = OUT / f"internal_r{r}.npz"
        if out.exists():
            log.info("exists %s", out.name)
            continue
        t0 = time.time()
        risk = {p: np.full((len(tcga), len(TIMES)), np.nan) for p in PROCS}
        score = {p: np.full(len(tcga), np.nan) for p in PROCS}
        chosen = []
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for k, (tr, te) in enumerate(skf.split(tcga, strata(tcga))):
            seed = config.SEED + 100 * r + k
            train, test = tcga.iloc[tr].reset_index(drop=True), tcga.iloc[te].reset_index(drop=True)
            for p in PROCS:
                m, info = tune_fit(p, train, feats, seed, criterion)
                risk[p][te] = m.risk(test)
                score[p][te] = m.score(test)
                chosen.append({"repeat": r, "fold": k, "procedure": p, **info})
                release_memory()
            log.info("survival primary r%d fold %d (%.0fs)", r, k, time.time() - t0)
        np.savez(out, sample=tcga["sample"].values.astype(str),
                 **{f"risk__{p}": v for p, v in risk.items()}, **{f"score__{p}": v for p, v in score.items()})
        pd.DataFrame(chosen).to_csv(OUT / f"internal_r{r}_choices.csv", index=False)


def impute_idh_model(tcga):
    """Logistic model for IDH-wildtype status from age, grade and histology,
    fitted on TCGA patients with recorded IDH status."""
    f = ["age", "grade", "histology"]
    d = tcga[tcga.idh_status.notna()]
    pre = make_preprocessor(d, f).fit(d[f])
    lr = LogisticRegression(max_iter=2000).fit(pre.transform(d[f]), (d.idh_status == "WT").astype(int))
    return lambda df: np.where(lr.predict_proba(pre.transform(df[f]))[:, 1] >= 0.5, "WT", "Mutant")


def external(criterion="cindex"):
    OUT = outdir(criterion)
    tcga = load_cohort("tcga", with_expr=False).reset_index(drop=True)
    imp = impute_idh_model(tcga)
    rows = []
    for tgt in ("gravendeel", "rembrandt"):
        tdf = load_cohort(tgt, with_expr=False).reset_index(drop=True)
        feats, _ = resolve_features("B", tcga, tdf)
        variants = {"mode": tdf}
        if tgt == "gravendeel":
            alt = tdf.copy()
            miss = alt.idh_status.isna()
            alt.loc[miss, "idh_status"] = imp(alt.loc[miss])
            variants["model_imputed"] = alt
            rows.append({"target": tgt, "idh_missing": int(miss.sum()),
                         "imputed_wildtype": int((alt.loc[miss, "idh_status"] == "WT").sum())})
        res, chosen = {}, []
        for p in PROCS:
            m, info = tune_fit(p, tcga, feats, config.SEED, criterion)
            chosen.append({"target": tgt, "procedure": p, "features": ";".join(feats), **info})
            for v, d in variants.items():
                res[f"risk__{p}__{v}"] = m.risk(d)
                res[f"score__{p}__{v}"] = m.score(d)
            release_memory()
        np.savez(OUT / f"external_{tgt}.npz", sample=tdf["sample"].values.astype(str), **res)
        pd.DataFrame(chosen).to_csv(OUT / f"external_{tgt}_choices.csv", index=False)
        log.info("external %s done", tgt)
    save_json(rows, OUT / "external_idh_imputation.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, nargs="*")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--criterion", default="cindex", choices=["cindex", "pll"])
    a = ap.parse_args()
    configure_tensorflow()
    set_seed()
    if a.repeats is not None:
        internal(a.repeats, a.criterion)
    if a.external:
        external(a.criterion)
