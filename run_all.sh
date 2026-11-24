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
for s in 10_selection_equivalence 11_class_weighting 12_censoring_weights 13_full_cohort 14_nested_tuning 15_seed_variability 16_gravendeel_idh; do
  python scripts/$s.py
done
python scripts/17_survival_primary.py --repeats 0 1 2 3 4 --external
python scripts/17_survival_primary.py --criterion pll --repeats 0 1 2 3 4 --external
python scripts/18_survival_evaluation.py
SURV_CRITERION=pll python scripts/18_survival_evaluation.py
python scripts/19_within_group.py
python scripts/20_binary_additions.py
python scripts/08_figures.py
