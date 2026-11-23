#!/usr/bin/env bash
# Full reproduction. Outputs are written to results/.
set -euo pipefail
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=20260628
python scripts/01_download_data.py
python scripts/09_provenance.py
python scripts/02_build_cohort.py
python scripts/03_internal_cv.py
python scripts/03_internal_cv.py --unweighted
python scripts/04_external_validation.py
python scripts/04_external_validation.py --unweighted
python scripts/05_survival.py
python scripts/06_sensitivity.py
python scripts/07_tables.py
python scripts/08_figures.py
