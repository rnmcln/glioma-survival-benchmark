"""Post hoc analysis R6 (PROTOCOL.md, section 12): variability due to network
initialisation alone.

Primary setting (TCGA, set B, 24 months), folds of the first cross-validation
repeat held fixed; the three planned networks (class-weighted) are
refitted with five different seeds per fold. The spread of out-of-fold AUROC
and of its difference from penalised logistic regression (same folds) isolates
training stochasticity from resampling variability.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import config
from src.cv import load, release_memory
from src.endpoints import make_binary_endpoint
from src.evaluation import fast_auroc, point_metrics
from src.harmonise import load_cohort
from src.models import MLPS, KerasClassifier
from src.preprocessing import resolve_features
from src.utils import configure_tensorflow, get_logger, set_seed

log = get_logger()
SEEDS = [11, 22, 33, 44, 55]


def main():
    configure_tensorflow()
    set_seed()
    tcga = load_cohort("tcga", with_expr=False)
    feats, _ = resolve_features("B", tcga)
    lab, _ = make_binary_endpoint(tcga, 24)
    lab = lab.reset_index(drop=True)
    y = lab.y.values.astype(int)
    y0, _, per, _ = load("int_B_2y")
    assert (y0 == y).all()
    plr0 = per["penalised_logistic"][0]
    a_plr = fast_auroc(y, plr0)
    skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    folds = list(skf.split(lab, y))
    rows = []
    for s in SEEDS:
        oof = {m: np.full(len(y), np.nan) for m in MLPS}
        for k, (tr, te) in enumerate(folds):
            for m in MLPS:
                est = KerasClassifier(feats, arch=m[3:], class_weight=True, seed=config.SEED + 1000 * s + k)
                est.fit(lab.iloc[tr], y[tr])
                oof[m][te] = est.predict_proba(lab.iloc[te])[:, 1]
                release_memory()
        for m in MLPS:
            pm = point_metrics(y, oof[m])
            rows.append({"seed": s, "network": m, "auroc": pm["auroc"], "auroc_plr": a_plr,
                         "delta": pm["auroc"] - a_plr, "brier": pm["brier"],
                         "calibration_intercept": pm["calibration_intercept"],
                         "calibration_slope": pm["calibration_slope"]})
        log.info("seed %d done", s)
    pd.DataFrame(rows).to_csv(config.TABLES / "seed_variability.csv", index=False)


if __name__ == "__main__":
    main()
