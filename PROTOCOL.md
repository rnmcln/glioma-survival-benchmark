# Analysis protocol

Version history and status

* Version 1.0, dated 28 June 2026, is an internal analysis plan. It was first
  committed to a version-controlled repository (rnmcln/glioma-nn-penalty) on
  29 June 2026 together with the analysis code and initial results, so its
  precedence over the inspection of results cannot be verified independently.
  It specified the binary 24-month endpoint with 12- and 60-month horizons,
  predictor sets A to C, the model catalogue (including three class-weighted
  multilayer perceptrons), repeated cross-validation, external validation in
  Gravendeel and REMBRANDT on shared predictors, grade III-IV and grade II
  subgroups, complete-case analysis, Cox models as a secondary benchmark,
  decision curves and the decision rule (Section 9). The original code also
  contained analyses that the plan does not list: IDH-wildtype, IDH-mutant and
  glioblastoma strata, a 6-month landmark analysis, inverse-probability-of-
  censoring weighting, a feature-tokeniser transformer and nested tuning of
  penalised logistic regression. Sections 1 to 10 below describe version 1.0
  together with these analyses as implemented in the corrected code.
* Version 1.1 (23 November 2026) corrects data-harmonisation and
  implementation errors and adds networks trained without class weighting,
  expression-scaling and intercept-updating analyses (Section 11).
* Version 1.2 (23 November 2026) adds post hoc analyses R1 to R7 (Section 12).
* Version 2.0 (23 November 2026) was defined after the results of versions 1.0
  to 1.2 were known, in response to methodological review. It makes a
  time-to-event analysis of all patients the primary analysis, adds a flexible
  regression comparator and analyses of diagnostic heterogeneity, missing data
  and class weighting, and retains the binary analysis as a secondary
  benchmark (Section 13). No earlier analysis is removed or redefined.

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

## 10. Sensitivity and additional analyses

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
4. The planned 40% missingness rule is now applied; it removes 1p/19q
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

## 12. Post hoc analyses added in version 1.2

These analyses were defined after the results of Sections 8 to 10 were known,
in response to an internal methodological review. They are exploratory.

* R1, selection-aware inference (`scripts/10_selection_equivalence.py`): the
  selection of the best network and the best transparent model is repeated in
  each bootstrap resample.
* R2, fixed comparisons and equivalence (same script): each network against
  penalised logistic regression without selection; 90% bootstrap intervals
  compared with a margin of +/-0.03 AUROC; per-repeat differences.
* R3, class weighting (`scripts/11_class_weighting.py`): calibration intercept
  against the logit of the training event rate; analytic prior-shift
  correction using training-fold event rates; net benefit with and without
  weighting; decision-curve differences after intercept updating of both
  models in the external cohorts.
* R4a, covariate-dependent IPCW (`scripts/12_censoring_weights.py`): Cox model
  for the censoring hazard on predictor set B.
* R4b, full-cohort evaluation (`scripts/13_full_cohort.py`): all TCGA patients,
  folds stratified by 24-month status; binary models and survival models
  (penalised Cox; Cox-loss multilayer perceptrons with the compact and
  regularised architectures) evaluated by inverse-probability-of-censoring
  weighted time-dependent AUROC and Brier score at 24 months; external
  application to all Gravendeel and REMBRANDT patients.
* R5, nested tuning of both model classes (`scripts/14_nested_tuning.py`):
  first three repeats of the primary cross-validation, inner 3-fold selection
  by AUROC; penalised logistic regression grid C x L1 ratio (12 points);
  multilayer perceptron grid hidden layers x dropout x L2 (8 points), trained
  without class weighting.
* R6, initialisation variability (`scripts/15_seed_variability.py`): five
  seeds per network on fixed folds.
* R7, IDH imputation in Gravendeel (`scripts/16_gravendeel_idh.py`): stored
  external predictions re-evaluated separately in patients with and without
  recorded IDH status.

## 13. Version 2.0: revised primary analysis and revision analyses

Rationale. The binary endpoint excludes 260 of 664 TCGA patients censored
before 24 months, who differ systematically from labelled patients; the
natural task is survival prediction with right censoring. The original
comparator was additive and linear in age and grade; model selection in the
binary benchmark used evaluation data; and the original uncertainty
intervals conditioned on fitted predictions.

S1, primary analysis (`scripts/17_survival_primary.py`, evaluated by
`scripts/18_survival_evaluation.py`). All TCGA patients; predictor set B;
5 repeats of 5-fold cross-validation stratified by 24-month status (folds as
in R4). Procedures: elastic-net Cox (L1 ratio 0.5; penalty in {0.01, 0.03,
0.1, 0.3, 1, 3}); ridge Cox on the flexible design (`src/flexible.py`: age
restricted cubic spline with four knots, categorical grade, age x IDH-wildtype
and grade x IDH-wildtype interactions; penalty in {0.03, 0.1, 0.3, 1, 3, 10});
Cox-loss multilayer perceptron (four configurations). Selection by inner
3-fold cross-validation within the training folds using Harrell's C. Measures
at 12, 24 and 60 months: time-dependent AUROC (Uno), IPCW Brier score,
observed/expected ratio, calibration slope; Harrell's C. Estimand: single-model
performance averaged over repeats; uncertainty pooled over repeat-specific
bootstrap resamples (300 per repeat). External: common-predictor transport to
Gravendeel and REMBRANDT; IDH imputation by mode (original) and by a
TCGA-trained model; IDH-recorded subset. Sensitivity analysis, added after the
primary results showed calibration slopes above 1 for the penalised Cox
models: selection by the partial log-likelihood per event of the inner
validation folds, with risk sets formed within each fold (`--criterion pll`).

S2, diagnostic heterogeneity (`scripts/18_survival_evaluation.py`,
`scripts/19_within_group.py`). Molecular groups from IDH and 1p/19q status;
Harrell's C split into within-group and between-group pairs; reference Cox
models with molecular group alone and with age; within-group evaluation of
models developed in all patients; within-group development (IDH-wildtype and
IDH-mutant non-codeleted groups; 3 repeats of 5-fold cross-validation).

B1 to B8, binary benchmark (`scripts/20_binary_additions.py`): flexible
penalised logistic regression; pooled repeat-bootstrap intervals; decision-rule
margins of 0.01 and 0.02; external calibration intervals and cross-fitted
intercept updating; Gravendeel IDH characteristics, pair decomposition and
model-based imputation; class weighting at fixed risk levels; inductive
within-sample rank transformation of expression data; network parameter
counts.
