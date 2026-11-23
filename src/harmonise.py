"""Harmonise TCGA, REMBRANDT and Gravendeel into one clinical schema.

Source: GlioVis data objects (.Rds) at a pinned repository commit. Output
columns for every cohort:

    cohort, sample, age, sex, grade, grade_group, histology,
    idh_status, codel_1p19q, mgmt_status, os_months, os_event

Variables absent from a cohort are written as missing and are never imputed
across cohorts. Harmonisation decisions:

* Retained histologies: glioblastoma, astrocytoma, oligodendroglioma and mixed
  glioma/oligoastrocytoma. Non-tumour, pilocytic astrocytoma, unknown and
  unclassified samples are excluded.
* Glioblastoma is WHO grade IV by definition in the classifications used by
  these cohorts; grade is set to IV for glioblastoma where it is missing or
  recorded otherwise. The number of recoded records is reported.
* Patients younger than 18 years are excluded where age is recorded.
* Gravendeel 1p/19q status is derived from loss of heterozygosity at 1p and
  19q (codeleted when both arms show complete loss); partial loss is coded as
  non-codeleted.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

log = get_logger()

KEEP_HISTOLOGY = {
    "GBM": "GBM",
    "Astrocytoma": "Astrocytoma",
    "Oligodendroglioma": "Oligodendroglioma",
    "Oligoastrocytoma": "Mixed/OA",
    "Mixed glioma": "Mixed/OA",
}
GRADE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    return np.nan if s.lower() in ("", "nan", "na", "none") else s


def load_rds(name: str):
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rds(str(config.RAW / config.GLIOVIS_FILES[name]))
    return obj["pData"].copy(), obj["expr"].copy()


def _idh(v):
    s = _clean(v)
    if pd.isna(s):
        return np.nan
    s = s.lower()
    if s.startswith("mut"):
        return "Mutant"
    if s in ("wt", "wild_type", "wildtype", "wild type"):
        return "WT"
    return np.nan


def _codel(pdata: pd.DataFrame) -> pd.Series:
    if "Chr.1p_19q.codeletion" in pdata.columns:
        def f(v):
            s = _clean(v)
            if pd.isna(s):
                return np.nan
            s = s.lower()
            if s == "codel":
                return "Codeleted"
            if s == "non-codel":
                return "Non-codeleted"
            return np.nan
        return pdata["Chr.1p_19q.codeletion"].map(f)
    if {"LOH_1p", "LOH_19q"}.issubset(pdata.columns):
        def g(row):
            a, b = _clean(row["LOH_1p"]), _clean(row["LOH_19q"])
            if pd.isna(a) or pd.isna(b):
                return np.nan
            return "Codeleted" if (a.upper() == "YES" and b.upper() == "YES") else "Non-codeleted"
        return pdata.apply(g, axis=1)
    return pd.Series(np.nan, index=pdata.index)


def _mgmt(pdata: pd.DataFrame) -> pd.Series:
    if "MGMT.promoter.status" not in pdata.columns:
        return pd.Series(np.nan, index=pdata.index)
    def f(v):
        s = _clean(v)
        if pd.isna(s):
            return np.nan
        s = s.lower()
        if s.startswith("unmethyl"):
            return "Unmethylated"
        if s.startswith("methyl"):
            return "Methylated"
        return np.nan
    return pdata["MGMT.promoter.status"].map(f)


def harmonise_cohort(name: str):
    pdata, expr = load_rds(name)
    pdata = pdata.reset_index(drop=True)
    n_raw = len(pdata)
    out = pd.DataFrame({"cohort": name, "sample": pdata["Sample"].astype(str).values})
    out["age"] = pd.to_numeric(pdata["Age"], errors="coerce") if "Age" in pdata else np.nan
    out["sex"] = (pdata["Gender"].map(lambda v: _clean(v).capitalize() if not pd.isna(_clean(v)) else np.nan)
                  if "Gender" in pdata else np.nan)
    hist_raw = pdata["Histology"].map(_clean)
    out["histology"] = hist_raw.map(lambda v: KEEP_HISTOLOGY.get(v, np.nan))
    grade = pdata["Grade"].map(lambda v: GRADE_MAP.get(_clean(v), np.nan) if not pd.isna(_clean(v)) else np.nan)
    is_gbm = out["histology"] == "GBM"
    raw_grade = grade.copy()
    grade = grade.where(~is_gbm, 4)
    out["grade"] = grade.astype(float)
    out["grade_group"] = out["grade"].map(lambda g: "high" if g in (3, 4) else ("lower" if g in (1, 2) else np.nan))
    idh_col = next((c for c in ("IDH.status", "IDH1_status") if c in pdata.columns), None)
    out["idh_status"] = pdata[idh_col].map(_idh) if idh_col else np.nan
    out["codel_1p19q"] = _codel(pdata)
    out["mgmt_status"] = _mgmt(pdata)
    out["os_months"] = pd.to_numeric(pdata["survival"], errors="coerce")
    out["os_event"] = pd.to_numeric(pdata["status"], errors="coerce")

    keep_hist = out["histology"].notna()
    usable = out["os_months"].notna() & out["os_event"].notna() & (out["os_months"] >= 0)
    minor = out["age"].notna() & (out["age"] < config.MIN_AGE)
    mask = keep_hist & usable & ~minor
    clinical = out.loc[mask].reset_index(drop=True)

    meta = {"Sample", "Histology", "Grade", "Recurrence", "Subtype", "CIMP_status", "survival", "status"}
    genes = [g for g in config.GENE_PANEL if g in expr.columns and g not in meta]
    ex = expr[["Sample"] + genes].copy()
    ex["Sample"] = ex["Sample"].astype(str)
    ex = ex[ex["Sample"].isin(clinical["sample"])].reset_index(drop=True)

    info = {
        "cohort": name, "n_source": n_raw,
        "excluded_histology": int((~keep_hist).sum()),
        "excluded_no_outcome": int((keep_hist & ~usable).sum()),
        "excluded_age_under_18": int((keep_hist & usable & minor).sum()),
        "n_analysed": int(len(clinical)),
        "gbm_grade_recoded_to_IV": int((mask & is_gbm & (raw_grade != 4)).sum()),
        "panel_genes_available": len(genes),
    }
    log.info("harmonised %s: %d of %d retained", name, len(clinical), n_raw)
    return clinical, ex, info


def harmonise_all(save: bool = True):
    frames, infos = [], []
    for name in config.GLIOVIS_FILES:
        clin, ex, info = harmonise_cohort(name)
        frames.append(clin)
        infos.append(info)
        if save:
            clin.to_csv(config.PROCESSED / f"clinical_{name}.csv", index=False)
            ex.to_csv(config.PROCESSED / f"expr_panel_{name}.csv", index=False)
    summary = pd.DataFrame(infos)
    if save:
        summary.to_csv(config.PROCESSED / "harmonisation_summary.csv", index=False)
    return pd.concat(frames, ignore_index=True), summary


def load_cohort(name: str, with_expr: bool = True) -> pd.DataFrame:
    """Load a harmonised cohort; expression genes are z-scored within the cohort.

    Within-cohort standardisation uses no outcome data. For internal
    cross-validation it is immaterial, because every model re-standardises (or
    is invariant to affine rescaling) inside the training fold. For external
    validation it places RNA-seq and microarray measurements on a common scale.
    """
    clin = pd.read_csv(config.PROCESSED / f"clinical_{name}.csv", dtype={"sample": str})
    if with_expr:
        ex = pd.read_csv(config.PROCESSED / f"expr_panel_{name}.csv", dtype={"Sample": str})
        genes = [c for c in ex.columns if c != "Sample"]
        ex[genes] = (ex[genes] - ex[genes].mean()) / ex[genes].std(ddof=0)
        clin = clin.merge(ex.rename(columns={"Sample": "sample"}), on="sample", how="left")
        clin.attrs["genes"] = genes
    return clin
