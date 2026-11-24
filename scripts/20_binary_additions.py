"""Revised analyses for the original binary benchmark (PROTOCOL.md,
section 13).

B1  Flexible penalised logistic regression (ridge on the flexible design of
    src/flexible.py; C tuned in {0.01, 0.03, 0.1, 0.3, 1, 3} by inner 3-fold
    AUROC) in the primary setting, on the identical 5 x 5 outer folds; compared
    with the compact network and penalised logistic regression.
B2  Pooled repeat-bootstrap intervals (300 resamples per repeat, pooled) for the
    per-repeat AUROC and Brier differences, compact network minus penalised
    logistic regression, in every internal setting.
B3  Sensitivity of the decision rule to AUROC margins of 0.01 and 0.02.
B4  External binary settings: bootstrap intervals (first 500 of the 2000 shared
    resamples) for calibration
    intercepts and slopes of every model; cross-fitted intercept updating
    (random halves, update on one half and evaluate on the other, 200 splits).
B5  Gravendeel set B and IDH status: characteristics by IDH availability;
    AUROC decomposed into event/non-event pairs within and between the
    recorded and missing groups; predictions with IDH imputed by a TCGA-trained
    model (requires refitting the deterministic TCGA models, which are checked
    against the stored predictions).
B6  Class weighting at clinically interpretable risk levels: for patients with
    unweighted predicted risk within 0.025 of 0.10, 0.20 and 0.40, the
    weighted predictions of the same network (compact) at 12, 24 and 60 months.
B7  Expression panel with an inductive within-sample transformation: each
    patient's 40 panel values are replaced by their ranks within that patient
    (scaled to 0-1), which uses no information from other patients; models
    fitted on TCGA and applied to Gravendeel and REMBRANDT.
B8  Parameter counts of each network for each predictor set.
"""
import _path  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src import config
from src.cv import load, release_memory
from src.endpoints import make_binary_endpoint
from src.evaluation import (_logit, bootstrap_indices, calibration_intercept_slope, fast_auroc, paired_difference,
                            point_metrics)
from src.flexible import FlexibleDesign
from src.harmonise import load_cohort
from src.models import ARCHITECTURES, KerasClassifier, build_mlp, build_models
from src.preprocessing import make_preprocessor, resolve_features
from src.settings import EXTERNAL_LABELS, internal_settings
from src.survmetrics import auroc_pairs
from src.utils import configure_tensorflow, get_logger, set_seed

log = get_logger()
T = config.TABLES
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


class FlexLogistic:
    def __init__(self, feats, C):
        self.feats, self.C = feats, C

    def fit(self, df, y):
        self.d_ = FlexibleDesign(self.feats).fit(df)
        self.m_ = LogisticRegression(C=self.C, max_iter=5000).fit(self.d_.transform(df), y)
        return self

    def predict_proba(self, df):
        return self.m_.predict_proba(self.d_.transform(df))


def tuned_flex(df, y, feats):
    inner = StratifiedKFold(3, shuffle=True, random_state=config.SEED)
    sc = []
    for C in C_GRID:
        s = [fast_auroc(y[b], FlexLogistic(feats, C).fit(df.iloc[a], y[a]).predict_proba(df.iloc[b])[:, 1])
             for a, b in inner.split(df, y)]
        sc.append(np.mean(s))
    C = C_GRID[int(np.argmax(sc))]
    return FlexLogistic(feats, C).fit(df, y), C


def pooled(y, pa_rep, pb_rep, metric, seed_base=config.SEED, B=300):
    pts, boots = [], []
    for r in range(pa_rep.shape[0]):
        pts.append(paired_difference(y, pa_rep[r], pb_rep[r], metric, [np.arange(len(y))])["delta"])
        idx = bootstrap_indices(len(y), y, n_boot=B, seed=seed_base + r)
        boots += list(np.array([paired_difference(y, pa_rep[r], pb_rep[r], metric, [i])["delta"] for i in idx]))
    return float(np.mean(pts)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def b1():
    tcga = load_cohort("tcga", with_expr=False)
    feats, _ = resolve_features("B", tcga)
    lab, _ = make_binary_endpoint(tcga, 24)
    lab = lab.reset_index(drop=True)
    y = lab.y.values.astype(int)
    y0, mean, per, _ = load("int_B_2y")
    assert (y0 == y).all()
    R = per["penalised_logistic"].shape[0]
    flex = np.full((R, len(y)), np.nan)
    choice = []
    for r in range(R):
        skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
        for k, (tr, te) in enumerate(skf.split(lab, y)):
            m, C = tuned_flex(lab.iloc[tr].reset_index(drop=True), y[tr], feats)
            flex[r, te] = m.predict_proba(lab.iloc[te])[:, 1]
            choice.append({"repeat": r, "fold": k, "C": C})
    np.savez(config.PREDICTIONS / "revision" / "flex_logistic_B_2y.npz", oof=flex)
    rows = []
    for m, P in {"flexible_logistic": flex, "penalised_logistic": per["penalised_logistic"],
                 "tf_compact": per["tf_compact"]}.items():
        pm = pd.DataFrame([point_metrics(y, P[r]) for r in range(R)]).mean()
        rows.append({"row": "model", "model": m, "auroc": pm.auroc, "brier": pm.brier,
                     "calibration_intercept": pm.calibration_intercept, "calibration_slope": pm.calibration_slope})
    for a, b, A, Bm in (("tf_compact", "flexible_logistic", per["tf_compact"], flex),
                        ("flexible_logistic", "penalised_logistic", flex, per["penalised_logistic"])):
        da, dl, dh = pooled(y, A, Bm, "auroc")
        ba, bl, bh = pooled(y, A, Bm, "brier")
        rows.append({"row": "difference", "model": f"{a} - {b}", "auroc": da, "auroc_lo": dl, "auroc_hi": dh,
                     "brier": ba, "brier_lo": bl, "brier_hi": bh})
    pd.DataFrame(rows).to_csv(T / "binary_flexible.csv", index=False)
    pd.DataFrame(choice).to_csv(T / "binary_flexible_choices.csv", index=False)
    log.info("B1 done")


def b2_b3():
    rows = []
    udir = config.PREDICTIONS / "unweighted"
    for tag, *_ in internal_settings():
        y, _, per, _ = load(tag)
        _, _, peru, _ = load(tag, udir)
        for wt, P in (("weighted", per), ("unweighted", peru)):
            da, dl, dh = pooled(y, P["tf_compact"], per["penalised_logistic"], "auroc")
            ba, bl, bh = pooled(y, P["tf_compact"], per["penalised_logistic"], "brier")
            rows.append({"setting": tag, "training": wt, "delta_auroc": da, "lo": dl, "hi": dh,
                         "delta_brier": ba, "brier_lo": bl, "brier_hi": bh})
        log.info("B2 %s", tag)
    pd.DataFrame(rows).to_csv(T / "binary_pooled_intervals.csv", index=False)
    out = []
    for f, lab in (("decision_rule.csv", "weighted"), ("decision_rule_unweighted.csv", "unweighted")):
        d = pd.read_csv(T / f)
        for m in (0.01, 0.02, 0.03):
            a = d.delta_auroc >= m - 1e-9
            out.append({"training": lab, "margin": m, "n_settings": len(d), "auroc_margin_met": int(a.sum()),
                        "all_conditions_met": int((a & d.brier_not_worse & d.calibration_slope_ok).sum()),
                        "settings_meeting_margin": ";".join(d.setting[a])})
    pd.DataFrame(out).to_csv(T / "decision_rule_margins.csv", index=False)


def b4():
    rows, upd = [], []
    for tag in EXTERNAL_LABELS:
        y, mean, _, _ = load(tag)
        idx = bootstrap_indices(len(y), y)
        for m, p in mean.items():
            bo = np.array([calibration_intercept_slope(y[i], p[i]) for i in idx[:500]])
            ci, cs = calibration_intercept_slope(y, p)
            rows.append({"setting": tag, "model": m, "calibration_intercept": ci,
                         "intercept_lo": np.nanpercentile(bo[:, 0], 2.5), "intercept_hi": np.nanpercentile(bo[:, 0], 97.5),
                         "calibration_slope": cs, "slope_lo": np.nanpercentile(bo[:, 1], 2.5),
                         "slope_hi": np.nanpercentile(bo[:, 1], 97.5)})
        rng = np.random.default_rng(config.SEED)
        d = pd.read_csv(T / "decision_rule.csv").set_index("setting").loc[tag]
        for m in ("penalised_logistic", d.best_nn, "logistic_age_grade"):
            br = []
            for _ in range(200):
                perm = rng.permutation(len(y))
                h1, h2 = perm[: len(y) // 2], perm[len(y) // 2:]
                tot = 0.0
                for a, b in ((h1, h2), (h2, h1)):
                    ci, _ = calibration_intercept_slope(y[a], mean[m][a])
                    pb = 1 / (1 + np.exp(-(_logit(mean[m][b]) + ci)))
                    tot += np.sum((pb - y[b]) ** 2)
                br.append(tot / len(y))
            upd.append({"setting": tag, "model": m, "brier_validated": point_metrics(y, mean[m])["brier"],
                        "brier_crossfitted_update": float(np.mean(br)), "sd_over_splits": float(np.std(br))})
    pd.DataFrame(rows).to_csv(T / "external_calibration_intervals.csv", index=False)
    pd.DataFrame(upd).to_csv(T / "external_crossfitted_update.csv", index=False)
    log.info("B4 done")


def b5():
    tcga = load_cohort("tcga", with_expr=False)
    g = load_cohort("gravendeel", with_expr=False)
    lab_g, _ = make_binary_endpoint(g, 24)
    lab_g = lab_g.reset_index(drop=True)
    rows = []
    for name, mask in (("IDH recorded", g.idh_status.notna()), ("IDH missing", g.idh_status.isna())):
        d = g[mask]
        lg = lab_g[lab_g.idh_status.notna() == (name == "IDH recorded")]
        rows.append({"group": name, "n": len(d), "age_median": d.age.median(),
                     "glioblastoma_pct": 100 * (d.histology == "GBM").mean(), "grade_IV_pct": 100 * (d.grade == 4).mean(),
                     "deaths": int(d.os_event.sum()), "labelled_24m": len(lg), "deaths_24m": int(lg.y.sum()),
                     "event_rate_24m": float(lg.y.mean()), "idh_wildtype_pct_of_recorded": 100 * (d.idh_status == "WT").mean()
                     if name == "IDH recorded" else np.nan})
    pd.DataFrame(rows).to_csv(T / "gravendeel_idh_characteristics.csv", index=False)
    y, mean, _, sample = load("ext_gravendeel_B_2y")
    gi = lab_g.set_index("sample").loc[sample]
    grp = np.where(gi.idh_status.notna(), "recorded", "missing").astype(object)
    pr = []
    for m in ("penalised_logistic", "tf_compact", "tf_regularised", "logistic_age_grade"):
        pr.append({"model": m, **auroc_pairs(y, mean[m], grp)})
    pd.DataFrame(pr).to_csv(T / "gravendeel_idh_pairs.csv", index=False)
    # model-based imputation of IDH from age, grade and histology (TCGA only)
    feats, _ = resolve_features("B", tcga, g)
    lab_t, _ = make_binary_endpoint(tcga, 24)
    lab_t = lab_t.reset_index(drop=True)
    f = ["age", "grade", "histology"]
    dt = tcga[tcga.idh_status.notna()]
    pre = make_preprocessor(dt, f).fit(dt[f])
    lr = LogisticRegression(max_iter=2000).fit(pre.transform(dt[f]), (dt.idh_status == "WT").astype(int))
    alt = gi.reset_index().copy()
    miss = alt.idh_status.isna().values
    alt.loc[miss, "idh_status"] = np.where(lr.predict_proba(pre.transform(alt.loc[miss, f]))[:, 1] >= 0.5, "WT", "Mutant")
    ref = [c for c in ("age", "grade") if lab_t[c].notna().any() and alt[c].notna().any()]
    models = build_models(lab_t, feats, include_nn=True, reference_vars=ref)
    out = []
    preds = {}
    for m in ("penalised_logistic", "tf_compact"):
        models[m].fit(lab_t, lab_t.y.values)
        p_mode = models[m].predict_proba(gi.reset_index())[:, 1]
        assert np.allclose(p_mode, mean[m], atol=1e-6), m
        preds[m] = models[m].predict_proba(alt)[:, 1]
        release_memory()
    idx = bootstrap_indices(len(y), y)
    d = paired_difference(y, preds["tf_compact"], preds["penalised_logistic"], "auroc", idx)
    out.append({"idh_handling": "model-imputed", "auroc_tf_compact": fast_auroc(y, preds["tf_compact"]),
                "auroc_penalised_logistic": fast_auroc(y, preds["penalised_logistic"]), **d,
                "imputed_wildtype": int((alt.loc[miss, "idh_status"] == "WT").sum()), "imputed_n": int(miss.sum())})
    d = paired_difference(y, mean["tf_compact"], mean["penalised_logistic"], "auroc", idx)
    out.append({"idh_handling": "mode (original analysis)", "auroc_tf_compact": fast_auroc(y, mean["tf_compact"]),
                "auroc_penalised_logistic": fast_auroc(y, mean["penalised_logistic"]), **d})
    pd.DataFrame(out).to_csv(T / "gravendeel_idh_imputation.csv", index=False)
    log.info("B5 done")


def b6():
    rows = []
    for tag in ("int_B_1y", "int_B_2y", "int_B_5y"):
        _, mean, _, _ = load(tag)
        _, mu, _, _ = load(tag, config.PREDICTIONS / "unweighted")
        for level in (0.10, 0.20, 0.40):
            sel = np.abs(mu["tf_compact"] - level) <= 0.025
            w = mean["tf_compact"][sel]
            rows.append({"setting": tag, "unweighted_level": level, "n_patients": int(sel.sum()),
                         "weighted_median": float(np.median(w)) if sel.any() else np.nan,
                         "weighted_q1": float(np.percentile(w, 25)) if sel.any() else np.nan,
                         "weighted_q3": float(np.percentile(w, 75)) if sel.any() else np.nan})
    pd.DataFrame(rows).to_csv(T / "class_weighting_risk_levels.csv", index=False)


def rank_within(e, genes):
    v = e[genes].values.astype(float)
    r = np.argsort(np.argsort(v, axis=1), axis=1).astype(float)
    return pd.DataFrame(r / (len(genes) - 1), columns=genes, index=e.index)


def b7():
    raw = {c: pd.read_csv(config.PROCESSED / f"expr_panel_{c}.csv", dtype={"Sample": str})
           for c in ("tcga", "gravendeel", "rembrandt")}
    clin = {c: pd.read_csv(config.PROCESSED / f"clinical_{c}.csv", dtype={"sample": str}) for c in raw}
    genes = [x for x in raw["tcga"].columns if x != "Sample"]

    def merged(c):
        e = raw[c].copy()
        e[genes] = rank_within(e, genes)
        d = clin[c].merge(e.rename(columns={"Sample": "sample"}), on="sample")
        return make_binary_endpoint(d, 24)[0].reset_index(drop=True)
    tr = merged("tcga")
    rows = []
    for tgt in ("gravendeel", "rembrandt"):
        te = merged(tgt)
        models = build_models(tr, genes, include_nn=True)
        preds = {}
        for m in ("penalised_logistic", "tf_compact", "tf_regularised", "tf_medium"):
            models[m].fit(tr, tr.y.values)
            preds[m] = models[m].predict_proba(te)[:, 1]
            release_memory()
        y = te.y.values
        idx = bootstrap_indices(len(y), y)
        for m, p in preds.items():
            pm = point_metrics(y, p)
            row = {"target": tgt, "transform": "within-sample rank", "model": m, "n": len(y), "auroc": pm["auroc"],
                   "brier": pm["brier"], "calibration_intercept": pm["calibration_intercept"],
                   "calibration_slope": pm["calibration_slope"]}
            if m != "penalised_logistic":
                d = paired_difference(y, p, preds["penalised_logistic"], "auroc", idx)
                row.update({"delta_vs_plr": d["delta"], "lo": d["lo"], "hi": d["hi"]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(T / "expression_inductive.csv", index=False)
    log.info("B7 done")


def b8():
    rows = []
    tcga = load_cohort("tcga")
    for fs in ("A", "B", "C"):
        feats, _ = resolve_features(fs, tcga)
        lab, info = make_binary_endpoint(tcga, 24)
        n_in = make_preprocessor(lab, feats).fit(lab[feats]).transform(lab[feats].iloc[:2]).shape[1]
        for arch in ARCHITECTURES:
            m = build_mlp(n_in, arch, config.SEED)
            rows.append({"predictor_set": fs, "network": f"tf_{arch}", "encoded_inputs": n_in,
                         "parameters": int(m.count_params()),
                         "trainable_parameters": int(sum(np.prod(v.shape) for v in m.trainable_weights)),
                         "training_patients_per_fold": int(round(0.8 * info["n_labelled"])),
                         "training_deaths_per_fold": int(round(0.8 * info["n_deaths"]))})
            release_memory()
    pd.DataFrame(rows).to_csv(T / "network_parameters.csv", index=False)


if __name__ == "__main__":
    configure_tensorflow()
    set_seed()
    b8()
    b6()
    b2_b3()
    b4()
    b1()
    b5()
    b7()
