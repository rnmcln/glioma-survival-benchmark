"""The 18 prespecified evaluation settings (13 internal, 5 external) plus the
REMBRANDT clinical set A, which reduces to the same predictors as set B and is
therefore not counted separately."""
from __future__ import annotations

from . import config
from .harmonise import load_cohort
from .preprocessing import resolve_features


def internal_settings():
    tcga = load_cohort("tcga")
    fa, _ = resolve_features("A", tcga)
    fb, _ = resolve_features("B", tcga)
    fc, _ = resolve_features("C", tcga)
    P, S = config.CV_REPEATS_PRIMARY, config.CV_REPEATS_SECONDARY
    sub = lambda m: tcga[m].reset_index(drop=True)
    return [
        # tag, label, df, features, horizon, repeats
        ("int_A_2y", "TCGA, clinical (A), 2-year", tcga, fa, 24, P),
        ("int_B_2y", "TCGA, clinical + molecular (B), 2-year", tcga, fb, 24, P),
        ("int_C_2y", "TCGA, expression panel (C), 2-year", tcga, fc, 24, P),
        ("int_B_1y", "TCGA, set B, 1-year", tcga, fb, 12, S),
        ("int_B_5y", "TCGA, set B, 5-year", tcga, fb, 60, S),
        ("int_B_2y_high", "TCGA, set B, grade III-IV, 2-year", sub(tcga.grade_group == "high"), fb, 24, S),
        ("int_B_2y_lower", "TCGA, set B, grade II, 2-year", sub(tcga.grade_group == "lower"), fb, 24, S),
        ("int_B_2y_cc", "TCGA, set B, complete cases, 2-year", tcga.dropna(subset=fb).reset_index(drop=True), fb, 24, S),
        ("int_B_6mland_2y", "TCGA, set B, 6-month landmark, 2-year", sub(tcga.os_months >= 6), fb, 24, S),
        ("int_idhwt_2y", "TCGA, IDH-wildtype, 2-year", sub(tcga.idh_status == "WT"), fb, 24, S),
        ("int_idhwt_1y", "TCGA, IDH-wildtype, 1-year", sub(tcga.idh_status == "WT"), fb, 12, S),
        ("int_idhmut_5y", "TCGA, IDH-mutant, 5-year", sub(tcga.idh_status == "Mutant"), fb, 60, S),
        ("int_gbm_1y", "TCGA, glioblastoma, 1-year", sub(tcga.histology == "GBM"), fb, 12, S),
    ]


def external_settings():
    tcga = load_cohort("tcga")
    out = []
    for tgt in ("gravendeel", "rembrandt"):
        tdf = load_cohort(tgt)
        for fs in ("A", "B", "C"):
            feats, report = resolve_features(fs, tcga, tdf)
            out.append((f"ext_{tgt}_{fs}_2y", tgt, fs, tdf, feats, report))
    return tcga, out


EXTERNAL_LABELS = {
    "ext_gravendeel_A_2y": "Gravendeel, clinical (A), 2-year",
    "ext_gravendeel_B_2y": "Gravendeel, clinical + molecular (B), 2-year",
    "ext_gravendeel_C_2y": "Gravendeel, expression panel (C), 2-year",
    "ext_rembrandt_B_2y": "REMBRANDT, grade + histology, 2-year",
    "ext_rembrandt_C_2y": "REMBRANDT, expression panel (C), 2-year",
}
COUNTED_EXTERNAL = list(EXTERNAL_LABELS)
