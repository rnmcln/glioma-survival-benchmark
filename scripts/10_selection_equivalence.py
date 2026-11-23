"""Post hoc analyses R1 and R2 (PROTOCOL.md, section 12), computed from the
stored out-of-fold and external predictions; no model is refitted.

R1, selection-aware inference. The prespecified rule compares the best of
three networks with the best of two transparent models, each chosen by point
AUROC. Here the selection is repeated within each of the 2000 shared bootstrap
resamples, so that the interval reflects the variability of the selection.

R2, fixed comparisons and equivalence. Each network is compared with penalised
logistic regression, which uses the same predictors, without selection. The
90% bootstrap interval is compared with the margin of +/-0.03 (a post hoc
equivalence assessment; two one-sided tests at the 5% level).

Also: per-repeat AUROC differences between the compact network and penalised
logistic regression in the internal settings (resampling and training
variability across cross-validation repeats).
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from src import config
from src.cv import load
from src.evaluation import bootstrap_indices, fast_auroc
from src.models import MLPS, TRANSPARENT
from src.settings import EXTERNAL_LABELS, internal_settings
from src.utils import get_logger

log = get_logger()
M = config.MEANINGFUL_AUROC_DELTA


def tags():
    return [t for t, *_ in internal_settings()] + list(EXTERNAL_LABELS)


def boot_auroc(y, preds, idx):
    return {m: np.array([fast_auroc(y[i], p[i]) for i in idx]) for m, p in preds.items()}


def main():
    sel, fixed, reps = [], [], []
    udir = config.PREDICTIONS / "unweighted"
    for tag in tags():
        y, mean, per, _ = load(tag)
        _, mu, peru, _ = load(tag, udir)
        idx = bootstrap_indices(len(y), y)
        preds = {m: mean[m] for m in TRANSPARENT + MLPS}
        preds.update({f"{m}_unweighted": mu[m] for m in MLPS})
        pt = {m: fast_auroc(y, p) for m, p in preds.items()}
        B = boot_auroc(y, preds, idx)
        for wt, suffix in (("weighted", ""), ("unweighted", "_unweighted")):
            nets = [m + suffix for m in MLPS]
            best_nn = max(nets, key=pt.get)
            best_tr = max(TRANSPARENT, key=pt.get)
            naive = B[best_nn] - B[best_tr]
            resel = np.max([B[m] for m in nets], axis=0) - np.max([B[m] for m in TRANSPARENT], axis=0)
            sel.append({"setting": tag, "training": wt, "best_nn": best_nn, "best_transparent": best_tr,
                        "delta": pt[best_nn] - pt[best_tr],
                        "naive_lo": np.percentile(naive, 2.5), "naive_hi": np.percentile(naive, 97.5),
                        "reselect_mean": resel.mean(),
                        "reselect_lo": np.percentile(resel, 2.5), "reselect_hi": np.percentile(resel, 97.5),
                        "reselect_p_ge_margin": float(np.mean(resel >= M))})
            for m in nets:
                d = B[m] - B["penalised_logistic"]
                lo90, hi90 = np.percentile(d, [5, 95])
                fixed.append({"setting": tag, "training": wt, "network": m.replace("_unweighted", ""),
                              "delta": pt[m] - pt["penalised_logistic"],
                              "lo95": np.percentile(d, 2.5), "hi95": np.percentile(d, 97.5),
                              "lo90": lo90, "hi90": hi90,
                              "equivalent_within_margin": bool(lo90 > -M and hi90 < M),
                              "p_boot_ge_margin": float(np.mean(d >= M))})
        if tag.startswith("int_"):
            for wt, P in (("weighted", per), ("unweighted", peru)):
                for r in range(P["tf_compact"].shape[0]):
                    a = fast_auroc(y, P["tf_compact"][r])
                    b = fast_auroc(y, per["penalised_logistic"][r])
                    reps.append({"setting": tag, "training": wt, "repeat": r, "auroc_compact": a,
                                 "auroc_plr": b, "delta": a - b})
        log.info("selection/equivalence %s", tag)
    T = config.TABLES
    pd.DataFrame(sel).to_csv(T / "selection_aware.csv", index=False)
    pd.DataFrame(fixed).to_csv(T / "fixed_comparisons.csv", index=False)
    pd.DataFrame(reps).to_csv(T / "per_repeat_differences.csv", index=False)


if __name__ == "__main__":
    main()
