"""Revised analysis S2 (PROTOCOL.md, section 13): model development within
molecularly defined groups (exploratory).

Groups (TCGA, from recorded IDH and 1p/19q status): IDH-wildtype, and
IDH-mutant without 1p/19q codeletion. The IDH-mutant, 1p/19q-codeleted group
has too few deaths for model development and is not analysed. Within each
group, the three procedures of script 17 (tuned elastic-net Cox, tuned flexible
ridge Cox, tuned Cox-loss network) are developed and evaluated by 3 repeats of
stratified 5-fold cross-validation (strata: event indicator), with the same
inner tuning. Predictor set B without the variables that define the group
(IDH status; and, for the IDH-wildtype group, 1p/19q status, which is
non-codeleted in every IDH-wildtype patient). Evaluation: Harrell's C and
time-dependent AUROC at the horizon with adequate events (24 months for
IDH-wildtype, 60 months for IDH-mutant non-codeleted), per repeat; paired
differences with pooled repeat-bootstrap intervals (300 per repeat).
"""
import _path  # noqa: F401
import importlib.util

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from src import config
from src.cv import release_memory
from src.harmonise import load_cohort
from src.survmetrics import td_auc
from src.utils import configure_tensorflow, get_logger, set_seed

log = get_logger()
_spec = importlib.util.spec_from_file_location("s17", str(config.ROOT / "scripts" / "17_survival_primary.py"))
s17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s17)
R = 3
GROUPS = {"idhwt": ("WT", None, 24.0, ["age", "sex", "grade", "histology", "mgmt_status"]),
          "idhmut_noncodel": ("Mutant", "Non-codeleted", 60.0, ["age", "sex", "grade", "histology", "mgmt_status"])}
OUT = config.PREDICTIONS / "survival"


def run(name):
    idh, cod, H, feats = GROUPS[name]
    t = load_cohort("tcga", with_expr=False)
    m = (t.idh_status == idh) & ((t.codel_1p19q == cod) if cod else True)
    d = t[m].reset_index(drop=True)
    out = OUT / f"within_{name}.npz"
    if not out.exists():
        risk = {p: np.full((R, len(d), 3), np.nan) for p in s17.PROCS}
        score = {p: np.full((R, len(d)), np.nan) for p in s17.PROCS}
        for r in range(R):
            skf = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED + r)
            for k, (tr, te) in enumerate(skf.split(d, d.os_event.astype(int))):
                a, b = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
                for p in s17.PROCS:
                    mdl, _ = s17.tune_fit(p, a, feats, config.SEED + 100 * r + k)
                    risk[p][r, te] = mdl.risk(b)
                    score[p][r, te] = mdl.score(b)
                    release_memory()
            log.info("%s repeat %d done", name, r)
        np.savez(out, **{f"risk__{p}": v for p, v in risk.items()}, **{f"score__{p}": v for p, v in score.items()})
    z = np.load(out)
    j = s17.TIMES.index(H)
    rows = []

    def mets(dd, rr, ss):
        return {"harrell_c": concordance_index(dd.os_months.values, -ss, dd.os_event.values),
                "td_auc": td_auc(dd, rr, H)}
    for p in s17.PROCS:
        per = pd.DataFrame([mets(d, z[f"risk__{p}"][r][:, j], z[f"score__{p}"][r]) for r in range(R)])
        rows.append({"group": name, "row": "model", "model": p, "n": len(d), "deaths": int(d.os_event.sum()),
                     "deaths_by_h": int(((d.os_event == 1) & (d.os_months <= H)).sum()), "horizon": H,
                     **{k: per[k].mean() for k in per}})
    for a, b in (("cox_mlp", "cox_flexible"), ("cox_mlp", "cox_linear"), ("cox_flexible", "cox_linear")):
        pt, bo = [], []
        for r in range(R):
            ra, rb = z[f"risk__{a}"][r][:, j], z[f"risk__{b}"][r][:, j]
            sa, sb = z[f"score__{a}"][r], z[f"score__{b}"][r]
            A, Bm = mets(d, ra, sa), mets(d, rb, sb)
            pt.append({k: A[k] - Bm[k] for k in A})
            rng = np.random.default_rng(config.SEED + r)
            n = 0
            while n < 300:
                i = rng.integers(0, len(d), len(d))
                dd = d.iloc[i].reset_index(drop=True)
                if ((dd.os_event == 1) & (dd.os_months <= H)).sum() < 2 or (dd.os_months > H).sum() < 2:
                    continue
                try:
                    A, Bm = mets(dd, ra[i], sa[i]), mets(dd, rb[i], sb[i])
                except ValueError:
                    continue
                bo.append({k: A[k] - Bm[k] for k in A})
                n += 1
        pt, bo = pd.DataFrame(pt), pd.DataFrame(bo)
        for k in pt:
            rows.append({"group": name, "row": "difference", "model": f"{a} - {b}", "metric": k, "delta": pt[k].mean(),
                         "lo": np.percentile(bo[k], 2.5), "hi": np.percentile(bo[k], 97.5), "horizon": H})
    return rows


if __name__ == "__main__":
    configure_tensorflow()
    set_seed()
    rows = []
    for g in GROUPS:
        rows += run(g)
    pd.DataFrame(rows).to_csv(config.TABLES / "survival_within_group.csv", index=False)
