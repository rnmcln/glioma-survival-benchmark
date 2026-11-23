"""Paths, seeds and prespecified analysis settings.

Every analytical choice that is fixed before results are inspected is declared
here. PROTOCOL.md describes the same choices in prose.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DICTIONARY = DATA / "dictionary"
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
METRICS = RESULTS / "metrics"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"

for _d in (RAW, PROCESSED, DICTIONARY, PREDICTIONS, METRICS, TABLES, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# Reproducibility ------------------------------------------------------------
SEED = 20260628
N_BOOTSTRAP = int(os.environ.get("GSB_N_BOOTSTRAP", 2000))
CV_FOLDS = 5
CV_REPEATS_PRIMARY = int(os.environ.get("GSB_CV_REPEATS_PRIMARY", 5))
CV_REPEATS_SECONDARY = int(os.environ.get("GSB_CV_REPEATS_SECONDARY", 3))

# Population -------------------------------------------------------------------
MIN_AGE = 18  # adults only; applied where age is recorded

# Endpoints (months) -------------------------------------------------------------
ENDPOINTS = {"os_1y": 12, "os_2y": 24, "os_5y": 60}
PRIMARY_ENDPOINT = "os_2y"

# Predictor sets -----------------------------------------------------------------
CLINICAL = ["age", "sex", "grade", "histology"]
MOLECULAR = ["idh_status", "codel_1p19q", "mgmt_status"]
FEATURE_SETS = {"A": CLINICAL, "B": CLINICAL + MOLECULAR, "C": None}  # C = gene panel
PRIMARY_FEATURE_SET = "B"

# Variables with more than this fraction missing are not used, unless essential.
MISSINGNESS_DROP_THRESHOLD = 0.40
ESSENTIAL_VARS = {"age", "grade"}

# Decision rule for a clinically meaningful improvement by a neural network.
MEANINGFUL_AUROC_DELTA = 0.03
CALIBRATION_SLOPE_RANGE = (0.8, 1.25)

# Decision-curve thresholds (predicted probability of death by the horizon).
DCA_THRESHOLDS = [round(i / 100, 2) for i in range(1, 100)]
DCA_PLAUSIBLE = (0.10, 0.50)
DCA_REPORT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

# Fixed categorical levels (structural metadata; no outcome information).
CATEGORICAL_LEVELS = {
    "sex": ["Female", "Male"],
    "histology": ["GBM", "Astrocytoma", "Oligodendroglioma", "Mixed/OA"],
    "idh_status": ["Mutant", "WT"],
    "codel_1p19q": ["Codeleted", "Non-codeleted"],
    "mgmt_status": ["Methylated", "Unmethylated"],
}

# Feature set C: 40-gene panel fixed a priori from established glioma drivers,
# diagnostic markers, and microenvironment/proliferation markers. Outcome data
# were not used to choose genes.
GENE_PANEL = [
    "EGFR", "PDGFRA", "PTEN", "NF1", "TP53", "RB1", "CDKN2A", "CDKN2B",
    "MDM2", "MDM4", "CDK4", "CDK6", "MET", "PIK3CA", "PIK3R1", "PIK3CB",
    "IDH1", "IDH2", "ATRX", "CIC", "FUBP1", "NOTCH1", "TERT", "MGMT",
    "OLIG2", "GFAP", "SOX2", "NES", "CD44", "CHI3L1", "VEGFA", "MKI67",
    "BCAN", "DLL3", "POSTN", "SERPINE1", "TIMP1", "LGALS1", "ANXA1", "CASP8",
]

# Neural-network training settings (identical for every fit).
NN_TRAINING = {"learning_rate": 1e-3, "batch_size": 64, "max_epochs": 200,
               "patience": 20, "validation_fraction": 0.2}

# Data sources ---------------------------------------------------------------------
GLIOVIS_COMMIT = "eae2ce7852baf0536460fe1080b1ef395c853e59"
GLIOVIS_FILES = {"tcga": "TCGA_GBMLGG.Rds", "rembrandt": "Rembrandt.Rds",
                 "gravendeel": "Gravendeel.Rds"}
CBIOPORTAL_STUDY = "lgggbm_tcga_pub"
CBIOPORTAL_API = "https://www.cbioportal.org/api"
GEO_REMBRANDT = "GSE108476"
GEO_GRAVENDEEL = "GSE16011"
