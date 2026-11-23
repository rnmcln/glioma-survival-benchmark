"""Repeated stratified cross-validation and external validation.

Internal validation: for each repeat r, StratifiedKFold(5, shuffle, seed+r);
every model is fitted on the same training folds and predicts the same test
folds. Out-of-fold predictions are stored per repeat; the primary estimate uses
out-of-fold probabilities averaged across repeats, and per-repeat metrics give
the resampling standard deviation. Runs are checkpointed per repeat.

External validation: models are fitted once on the full labelled TCGA cohort
and applied unchanged to the target cohort.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config
from .endpoints import make_binary_endpoint
from .models import build_models
from .utils import get_logger, save_json

log = get_logger()


def release_memory():
    """Free TensorFlow graph state between fits (Keras retains it otherwise)."""
    import gc
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    gc.collect()


def internal_cv(tag, df, features, horizon, n_repeats, include_nn=True, class_weight=True,
                include_ftt=False, only=None, outdir=None):
    outdir = outdir or config.PREDICTIONS
    out = outdir / f"{tag}.npz"
    if out.exists():
        log.info("exists %s", tag)
        return out
    lab, info = make_binary_endpoint(df, horizon)
    lab = lab.reset_index(drop=True)
    y = lab["y"].values.astype(int)
    n = len(lab)
    names = list(build_models(lab, features, include_nn=include_nn, class_weight=class_weight,
                              include_ftt=include_ftt).keys())
    if only:
        names = [m for m in names if m in only]
    ck = outdir / f"{tag}.ckpt.npz"
    oof = {m: np.full((n_repeats, n), np.nan) for m in names}
    done = []
    if ck.exists():
        d = np.load(ck)
        done = d["done"].tolist()
        for m in names:
            oof[m] = d[f"oof__{m}"]
    t0 = time.time()
    for r in range(n_repeats):
        if r in done:
            continue
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for k, (tr, te) in enumerate(skf.split(lab, y)):
            seed = config.SEED + 100 * r + k
            models = build_models(lab.iloc[tr], features, seed=seed, include_nn=include_nn,
                                  class_weight=class_weight, include_ftt=include_ftt,
                                  reference_vars=[c for c in ("age", "grade") if lab[c].notna().any()])
            for m in names:
                est = models[m]
                est.fit(lab.iloc[tr], y[tr])
                oof[m][r, te] = est.predict_proba(lab.iloc[te])[:, 1]
            del models
            release_memory()
        done.append(r)
        np.savez(ck, done=np.array(done), **{f"oof__{m}": oof[m] for m in names})
        log.info("%s repeat %d/%d (%.0fs)", tag, r + 1, n_repeats, time.time() - t0)
    np.savez(out, y=y, sample=lab["sample"].values.astype(str),
             **{f"oof__{m}": oof[m] for m in names})
    save_json({"setting": tag, "features": features, "endpoint": info,
               "cv": f"{n_repeats}x{config.CV_FOLDS}", "models": names,
               "class_weight": class_weight}, outdir / f"{tag}.json")
    ck.unlink(missing_ok=True)
    return out


def external_validation(tag, train_df, test_df, features, horizon, include_nn=True,
                        class_weight=True, include_ftt=False, only=None, outdir=None):
    outdir = outdir or config.PREDICTIONS
    out = outdir / f"{tag}.npz"
    if out.exists():
        log.info("exists %s", tag)
        return out
    tr, tr_info = make_binary_endpoint(train_df, horizon)
    te, te_info = make_binary_endpoint(test_df, horizon)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    ref = [c for c in ("age", "grade") if c in te.columns and tr[c].notna().any() and te[c].notna().any()]
    models = build_models(tr, features, include_nn=include_nn, class_weight=class_weight,
                          include_ftt=include_ftt, reference_vars=ref)
    names = [m for m in models if (not only or m in only)]
    preds = {}
    for m in names:
        models[m].fit(tr, tr["y"].values)
        preds[m] = models[m].predict_proba(te)[:, 1][None, :]
        release_memory()
    np.savez(out, y=te["y"].values.astype(int), sample=te["sample"].values.astype(str),
             **{f"oof__{m}": preds[m] for m in names})
    save_json({"setting": tag, "features": features, "train_endpoint": tr_info,
               "test_endpoint": te_info, "models": names, "class_weight": class_weight},
              outdir / f"{tag}.json")
    log.info("external %s done", tag)
    return out


def load(tag, outdir=None):
    d = np.load((outdir or config.PREDICTIONS) / f"{tag}.npz")
    y = d["y"]
    per_repeat = {k[5:]: d[k] for k in d.files if k.startswith("oof__")}
    mean = {m: np.nanmean(v, axis=0) for m, v in per_repeat.items()}
    return y, mean, per_repeat, d["sample"]
