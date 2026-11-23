"""Post hoc analysis R5 (PROTOCOL.md, section 12): nested tuning of both model
classes in the primary setting (TCGA, set B, 24 months).

Outer loop: the first three repeats of the primary 5-fold cross-validation
(identical folds to the main analysis). Inner loop: 3-fold stratified
cross-validation within each outer training set, selecting by mean inner AUROC.

Penalised logistic regression grid: C in {0.01, 0.1, 1, 10} x L1 ratio in
{0.2, 0.5, 0.8}. Multilayer perceptron grid (no class weighting): hidden layers
(32, 16) or (128, 64) x dropout 0 or 0.3 x L2 penalty 0 or 1e-3. External or
outer-test data are never used for selection.

Output: results/predictions/revision/nested_B_2y.npz (out-of-fold
probabilities per repeat), results/tables/nested_tuning_choices.csv.
"""
import _path  # noqa: F401
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src import config
from src.cv import release_memory
from src.endpoints import make_binary_endpoint
from src.evaluation import fast_auroc
from src.harmonise import load_cohort
from src.nn_extra import ConfigurableMLP
from src.preprocessing import make_preprocessor, resolve_features
from src.utils import configure_tensorflow, get_logger, set_seed

log = get_logger()
N_REPEATS = int(__import__("os").environ.get("GSB_NESTED_REPEATS", 3))
PLR_GRID = [{"C": c, "l1_ratio": l} for c in (0.01, 0.1, 1.0, 10.0) for l in (0.2, 0.5, 0.8)]
MLP_GRID = [{"hidden": h, "dropout": d, "l2": l2}
            for h, d, l2 in itertools.product([(32, 16), (128, 64)], [0.0, 0.3], [0.0, 1e-3])]
OUT = config.PREDICTIONS / "revision"
OUT.mkdir(exist_ok=True)


def plr(df, feats, C, l1_ratio, seed):
    return Pipeline([("pre", make_preprocessor(df, feats)),
                     ("clf", LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                                                max_iter=10000, random_state=seed))])


def select(make, grid, d, y, seed):
    inner = StratifiedKFold(3, shuffle=True, random_state=config.SEED)
    scores = []
    for g in grid:
        s = []
        for a, b in inner.split(d, y):
            est = make(g, seed).fit(d.iloc[a], y[a])
            s.append(fast_auroc(y[b], est.predict_proba(d.iloc[b])[:, 1]))
            release_memory()
        scores.append(np.mean(s))
    k = int(np.argmax(scores))
    return grid[k], scores[k]


def main():
    configure_tensorflow()
    set_seed()
    tcga = load_cohort("tcga", with_expr=False)
    feats, _ = resolve_features("B", tcga)
    lab, _ = make_binary_endpoint(tcga, 24)
    lab = lab.reset_index(drop=True)
    y = lab.y.values.astype(int)
    ck = OUT / "nested_B_2y.ckpt.npz"
    oof = {m: np.full((N_REPEATS, len(y)), np.nan) for m in ("plr_tuned", "mlp_tuned")}
    choices, done = [], []
    if ck.exists():
        z = np.load(ck, allow_pickle=True)
        done = z["done"].tolist()
        oof = {m: z[f"oof__{m}"] for m in oof}
        choices = pd.read_csv(OUT / "nested_choices.ckpt.csv").to_dict("records")
    t0 = time.time()
    for r in range(N_REPEATS):
        if r in done:
            continue
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for k, (tr, te) in enumerate(skf.split(lab, y)):
            seed = config.SEED + 100 * r + k
            d, yt = lab.iloc[tr].reset_index(drop=True), y[tr]
            g_p, s_p = select(lambda g, s: plr(d, feats, g["C"], g["l1_ratio"], s), PLR_GRID, d, yt, seed)
            g_m, s_m = select(lambda g, s: ConfigurableMLP(feats, seed=s, **g), MLP_GRID, d, yt, seed)
            oof["plr_tuned"][r, te] = plr(d, feats, g_p["C"], g_p["l1_ratio"], seed).fit(d, yt) \
                .predict_proba(lab.iloc[te])[:, 1]
            oof["mlp_tuned"][r, te] = ConfigurableMLP(feats, seed=seed, **g_m).fit(d, yt) \
                .predict_proba(lab.iloc[te])[:, 1]
            release_memory()
            choices.append({"repeat": r, "fold": k, "plr_C": g_p["C"], "plr_l1_ratio": g_p["l1_ratio"],
                            "plr_inner_auroc": s_p, "mlp_hidden": "-".join(map(str, g_m["hidden"])),
                            "mlp_dropout": g_m["dropout"], "mlp_l2": g_m["l2"], "mlp_inner_auroc": s_m})
            log.info("nested r%d k%d: PLR %s, MLP %s (%.0fs)", r, k, g_p, g_m, time.time() - t0)
        done.append(r)
        np.savez(ck, done=np.array(done), **{f"oof__{m}": v for m, v in oof.items()})
        pd.DataFrame(choices).to_csv(OUT / "nested_choices.ckpt.csv", index=False)
    np.savez(OUT / "nested_B_2y.npz", y=y, sample=lab["sample"].values.astype(str),
             **{f"oof__{m}": v for m, v in oof.items()})
    pd.DataFrame(choices).to_csv(config.TABLES / "nested_tuning_choices.csv", index=False)
    ck.unlink(missing_ok=True)
    (OUT / "nested_choices.ckpt.csv").unlink(missing_ok=True)


def summarise():
    """Paired comparisons on the first N_REPEATS repeats (out-of-fold
    probabilities averaged over those repeats), 2000 shared bootstrap resamples."""
    from src.cv import load
    from src.evaluation import bootstrap_indices, paired_difference, point_metrics
    z = np.load(OUT / "nested_B_2y.npz")
    y = z["y"]
    y0, _, per, _ = load("int_B_2y")
    _, _, peru, _ = load("int_B_2y", config.PREDICTIONS / "unweighted")
    assert (y0 == y).all()
    k = z["oof__plr_tuned"].shape[0]
    pr = {"plr_tuned": z["oof__plr_tuned"].mean(0), "mlp_tuned": z["oof__mlp_tuned"].mean(0),
          "plr_fixed": per["penalised_logistic"][:k].mean(0), "compact_fixed": per["tf_compact"][:k].mean(0),
          "compact_fixed_unweighted": peru["tf_compact"][:k].mean(0)}
    idx = bootstrap_indices(len(y), y)
    rows = []
    for m, p in pr.items():
        pm = point_metrics(y, p)
        rows.append({"row": "model", "model": m, "auroc": pm["auroc"], "brier": pm["brier"],
                     "calibration_intercept": pm["calibration_intercept"],
                     "calibration_slope": pm["calibration_slope"]})
    for a, b in [("mlp_tuned", "plr_tuned"), ("mlp_tuned", "plr_fixed"), ("plr_tuned", "plr_fixed"),
                 ("compact_fixed", "plr_fixed"), ("mlp_tuned", "compact_fixed_unweighted")]:
        da = paired_difference(y, pr[a], pr[b], "auroc", idx)
        db = paired_difference(y, pr[a], pr[b], "brier", idx)
        rows.append({"row": "difference", "model": f"{a} - {b}", "auroc": da["delta"], "auroc_lo": da["lo"],
                     "auroc_hi": da["hi"], "brier": db["delta"], "brier_lo": db["lo"], "brier_hi": db["hi"]})
    pd.DataFrame(rows).to_csv(config.TABLES / "nested_tuning_summary.csv", index=False)


if __name__ == "__main__":
    import sys
    if "--summary" not in sys.argv:
        main()
    summarise()
