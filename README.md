# glioma-survival-benchmark

Code for a benchmark of penalised regression and neural networks for
overall-survival prediction in adult diffuse glioma, using public structured
data. The primary analysis (version 2.0) compares linear and flexible penalised
Cox regression with Cox-loss neural networks in all TCGA patients (lower-grade
glioma and glioblastoma), with hyperparameter selection nested within the
training folds, and transports the models to the Gravendeel and REMBRANDT
cohorts. The original fixed-horizon binary benchmark (logistic regression,
tree ensembles and multilayer perceptrons) is retained as a secondary analysis. Results are summarised in [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md);
the analysis plan and its amendments are in [`PROTOCOL.md`](PROTOCOL.md).

## Contents

```
src/        harmonisation, endpoints, preprocessing, models, cross-validation, evaluation
scripts/    numbered pipeline steps (01-09 original analysis; 10-16 post hoc, version 1.2; 17-20 version 2.0)
data/       raw/ (downloaded), processed/ (harmonised cohorts), dictionary/ (checksums)
results/    created by the pipeline: predictions, metrics, tables, figures
```

## Requirements

Python 3.10 with the packages in `requirements.txt` (CPU only; no GPU needed):

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduction

```bash
bash run_all.sh
```

or step by step:

| Step | Script | Output |
|---|---|---|
| 1 | `01_download_data.py` | GlioVis objects at a pinned commit, GEO and cBioPortal tables |
| 2 | `02_build_cohort.py` | harmonised cohorts, cohort characteristics, missingness |
| 3 | `03_internal_cv.py` | repeated cross-validation in TCGA (13 settings) |
| 3b | `03_internal_cv.py --unweighted` | networks trained without class weighting |
| 4 | `04_external_validation.py [--unweighted]` | TCGA to Gravendeel and REMBRANDT |
| 5 | `05_survival.py` | Cox and penalised Cox, Harrell's C |
| 6 | `06_sensitivity.py` | IPCW, nested CV, expression scaling, final networks |
| 7 | `07_tables.py` | metrics, bootstrap intervals, decision rule, decision curves |
| 8 | `08_figures.py` | figures |
| 9 | `09_provenance.py` | SHA-256 manifest of raw inputs |
| 10 | `10_selection_equivalence.py` | selection-aware intervals, fixed comparisons, equivalence, per-repeat differences (post hoc R1, R2) |
| 11 | `11_class_weighting.py` | class weighting, event rate and calibration; prior-shift correction; decision curves after intercept updating (R3) |
| 12 | `12_censoring_weights.py` | covariate-dependent censoring weights (R4) |
| 13 | `13_full_cohort.py` | all patients, time-dependent AUROC and IPCW Brier; Cox-loss networks (R4) |
| 14 | `14_nested_tuning.py` | nested tuning of penalised regression and networks (R5) |
| 15 | `15_seed_variability.py` | network initialisation variability (R6) |
| 16 | `16_gravendeel_idh.py` | Gravendeel by availability of IDH status (R7) |
| 17 | `17_survival_primary.py [--criterion pll]` | primary survival analysis: tuned linear and flexible Cox and Cox-loss networks, internal and external (S1) |
| 18 | `18_survival_evaluation.py` | survival metrics, paired differences, proportional hazards, molecular-group decomposition (S1, S2); set `SURV_CRITERION=pll` for the sensitivity analysis |
| 19 | `19_within_group.py` | models developed within molecular groups (S2) |
| 20 | `20_binary_additions.py` | binary benchmark additions B1 to B8 |

Steps 10 to 16 are post hoc analyses (protocol version 1.2, Section 12) and
steps 17 to 20 the version 2.0 analyses (Section 13). Run them after step 7;
then re-run step 8 for the figures.

Seeds are fixed (`src/config.py`, seed 20260628) and TensorFlow runs
single-threaded with deterministic operations. A full run takes about eight to
ten hours on two CPU cores. Resampling depth can be reduced for a quick check
with `GSB_CV_REPEATS_PRIMARY`, `GSB_CV_REPEATS_SECONDARY` and `GSB_N_BOOTSTRAP`.

## Data

All inputs are public: TCGA LGG/GBM, Gravendeel (GEO GSE16011) and REMBRANDT
(GEO GSE108476), as distributed by the GlioVis portal
(Bowman et al., Neuro-Oncology 2017;19:139-141). The raw files are not
redistributed here; step 1 downloads them and step 9 checks them against
`data/dictionary/data_manifest.json`. Please cite the original data
publications when using derived data.

## Licence

MIT; see `LICENSE`.
