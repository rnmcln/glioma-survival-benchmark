"""Time-to-event benchmark on full follow-up (no horizon exclusion).

Cox proportional hazards (unpenalised) and elastic-net penalised Cox
(penaliser 0.1, L1 ratio 0.5), fitted with identical settings for internal and
external evaluation. Internal: 5 x 5 repeated cross-validation stratified on
the event indicator; external: fit on full TCGA. Discrimination is Harrell's C;
external C has a 2000-replicate percentile bootstrap interval.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from src import config
from src.harmonise import load_cohort
from src.preprocessing import make_preprocessor, resolve_features
from src.utils import get_logger

log = get_logger()
MODELS = {"cox": (0.0, 0.0), "cox_penalised": (0.1, 0.5)}


def fit_predict(train, test, feats, pen, l1):
    pre = make_preprocessor(train, feats).fit(train[feats])
    Xtr, Xte = pre.transform(train[feats]), pre.transform(test[feats])
    var = Xtr.std(axis=0) > 1e-8
    Xtr, Xte = Xtr[:, var], Xte[:, var]
    cols = [f"x{i}" for i in range(Xtr.shape[1])]
    d = pd.DataFrame(Xtr, columns=cols)
    d["T"], d["E"] = train.os_months.values, train.os_event.values
    for p in ([pen] if pen > 0 else [0.0, 0.01]):
        try:
            cph = CoxPHFitter(penalizer=p, l1_ratio=l1).fit(d, "T", "E")
            break
        except Exception:
            continue
    return cph.predict_partial_hazard(pd.DataFrame(Xte, columns=cols)).values.ravel()


def cindex(df, risk):
    return concordance_index(df.os_months.values, -risk, df.os_event.values)


def main():
    tcga = load_cohort("tcga", with_expr=False)
    rows = []
    for fs in ("A", "B"):
        feats, _ = resolve_features(fs, tcga)
        for m, (pen, l1) in MODELS.items():
            cs = []
            for r in range(config.CV_REPEATS_PRIMARY):
                skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
                for tr, te in skf.split(tcga, tcga.os_event.astype(int)):
                    a, b = tcga.iloc[tr], tcga.iloc[te]
                    cs.append(cindex(b, fit_predict(a, b, feats, pen, l1)))
            rows.append({"analysis": "internal", "cohort": "tcga", "feature_set": fs, "model": m,
                         "n": len(tcga), "events": int(tcga.os_event.sum()),
                         "cindex": np.mean(cs), "sd": np.std(cs, ddof=1), "lo": np.nan, "hi": np.nan,
                         "features": ";".join(feats)})
        for tgt in ("gravendeel", "rembrandt"):
            tdf = load_cohort(tgt, with_expr=False)
            f2, _ = resolve_features(fs, tcga, tdf)
            rng = np.random.default_rng(config.SEED)
            for m, (pen, l1) in MODELS.items():
                risk = fit_predict(tcga, tdf, f2, pen, l1)
                c = cindex(tdf, risk)
                boot = []
                for _ in range(config.N_BOOTSTRAP):
                    i = rng.integers(0, len(tdf), len(tdf))
                    boot.append(cindex(tdf.iloc[i], risk[i]))
                rows.append({"analysis": "external", "cohort": tgt, "feature_set": fs, "model": m,
                             "n": len(tdf), "events": int(tdf.os_event.sum()), "cindex": c, "sd": np.nan,
                             "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
                             "features": ";".join(f2)})
        log.info("survival set %s done", fs)
    out = pd.DataFrame(rows).round(3)
    out.to_csv(config.TABLES / "survival_cindex.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
