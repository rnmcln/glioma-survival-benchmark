# Analysis protocol

Version 1.0 was fixed on 28 June 2026, before model results were inspected.
Version 1.1 (23 November 2026) corrects data-harmonisation and implementation
errors identified during re-analysis and adds two sensitivity analyses. The
amendments are listed in Section 11; the design, endpoints, predictor sets,
model catalogue and decision rule are unchanged.

## 1. Objective

To estimate whether prespecified neural networks improve calibrated prediction
of fixed-horizon overall survival over transparent regression models in public
structured (clinical, molecular and limited expression) glioma data, under
internal and external validation.

## 2. Data

| Cohort | Role | Source |
|---|---|---|
| TCGA LGG and GBM | development | GlioVis `TCGA_GBMLGG.Rds` |
| Gravendeel (GSE16011) | external validation | GlioVis `Gravendeel.Rds` |
| REMBRANDT (GSE108476) | external validation with restricted predictors | GlioVis `Rembrandt.Rds` |

GlioVis objects are retrieved from the `msquatrito/shiny_GlioVis` repository at
a fixed commit; SHA-256 checksums are recorded. No imaging data are used.

## 3. Population

Adults (18 years or older where age is recorded) with glioblastoma,
astrocytoma, oligodendroglioma or mixed glioma/oligoastrocytoma and recorded
survival time and vital status. Non-tumour, pilocytic astrocytoma and
unclassified samples are excluded.

## 4. Endpoints

Primary: death by 24 months, as a binary label. Patients alive at 24 months are
non-events; patients censored before 24 months are excluded and compared with
included patients. Secondary horizons: 12 and 60 months. A time-to-event
analysis (Cox and penalised Cox; Harrell's C) uses full follow-up.

## 5. Predictor sets

* A: age, sex, WHO grade, histology.
* B: A plus IDH status, 1p/19q codeletion, MGMT promoter methylation.
* C: a 40-gene expression panel fixed a priori (see `src/config.py`).

A variable is used only if it is recorded in every cohort involved in an
analysis and is missing in at most 40% of patients in each, except age and
grade, which are retained wherever they are recorded. Grade is set to IV for
glioblastoma.

## 6. Preprocessing

Median imputation and standardisation of numeric predictors; mode imputation
and one-hot encoding (prespecified levels) of categorical predictors. All
preprocessing is fitted on training data only. Expression values are
standardised within each cohort.

## 7. Models

Transparent: base rate; unpenalised logistic regression on age and grade;
elastic-net logistic regression (C = 1, L1 ratio 0.5). Standard machine
learning: random forest (500 trees, minimum leaf size 5); histogram gradient
boosting (depth 3, learning rate 0.05, 400 iterations, L2 penalty 1).

Neural networks (TensorFlow): three multilayer perceptrons, compact
(32-16), medium (128-64-32 with batch normalisation, L2 1e-4) and regularised
(64-32, dropout 0.4, L2 1e-3); and a compact FT-Transformer (two blocks, token
dimension 32, four heads) for the primary internal and set-B external
settings. Training: Adam (learning rate 1e-3), binary cross-entropy, batch
size 64, up to 200 epochs, early stopping on a stratified 20% validation split
of the training data (patience 20, best weights restored), inverse-frequency
class weighting. A new network is initialised for every fit. No hyperparameter
is tuned on evaluation data.

## 8. Validation and inference

Internal: repeated stratified 5-fold cross-validation in TCGA (5 repeats for
feature sets A, B and C at 24 months; 3 repeats for other settings). All models
share folds. Out-of-fold probabilities are averaged across repeats for the
primary estimates; per-repeat metrics give the resampling standard deviation.
External: models fitted on all labelled TCGA patients and applied unchanged.

Metrics: AUROC, AUPRC, Brier score, calibration intercept and slope, decision-
curve net benefit (plausible thresholds 0.10 to 0.50). Uncertainty: 2000
percentile bootstrap resamples of evaluation patients, shared across models.

## 9. Decision rule

In each setting the best multilayer perceptron and the best transparent model
(age-and-grade or penalised logistic regression) are selected by AUROC. The
network is considered to improve prediction only if all three conditions hold:
AUROC difference at least 0.03; Brier score not higher; calibration slope
between 0.8 and 1.25.

## 10. Sensitivity analyses

1- and 5-year horizons; grade III-IV and grade II subgroups; complete cases;
6-month landmark; IDH-wildtype, IDH-mutant and glioblastoma strata; IPCW
reweighting of the primary analysis; time-to-event models; nested tuning of
penalised logistic regression; expression standardisation scheme;
FT-Transformer; networks trained without class weighting.

## 11. Amendments in version 1.1

1. Gravendeel 1p/19q status: source values carry trailing whitespace, which
   caused every recorded patient to be coded non-codeleted. Values are now
   trimmed before coding.
2. Glioblastoma with missing or inconsistent grade (REMBRANDT: 73 missing and
   2 coded grade I; TCGA: 3 missing) was previously imputed to the training
   median grade or kept as recorded; grade is now set to IV.
3. The adult-only criterion is now applied (2 TCGA and 2 Gravendeel patients
   younger than 18 years excluded).
4. The prespecified 40% missingness rule is now applied; it removes 1p/19q
   status (50% missing) from Gravendeel external validation.
5. Neural-network implementation: networks are now re-initialised with a new
   optimiser at every fit (previously weights were reset but optimiser state
   persisted across folds); training arrays are no longer padded by
   resampling; internal and external fits use identical training settings.
6. The age-and-grade model uses only the variables recorded in the
   evaluation cohort (grade alone in REMBRANDT) instead of imputing age.
7. Cox models use the same penalisation internally and externally.
8. IPCW weights use the Kaplan-Meier censoring distribution of the full cohort.
9. Added: networks without class weighting, because inverse-frequency
   weighting shifts predicted probabilities away from observed risk.
