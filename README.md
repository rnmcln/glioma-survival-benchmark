# Neural networks and penalised regression for glioma survival prediction

**TCGA-CDR reproducibility release v2.1.0.** TCGA-CDR benchmark of neural networks and penalised regression, with external evaluation in Gravendeel and REMBRANDT. Includes analysis and figure code, manuscript figures, and aggregate supplementary tables. The TCGA development cohort comprises 663 adults.

This release contains the code and aggregate results for the manuscript
using overall survival from the [TCGA Pan-Cancer Clinical Data Resource
(TCGA-CDR)](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018).
The original upstream source files are obtained separately under their
respective terms of use.

- **supplementary_data/** contains the exact aggregate CSVs accompanying the manuscript.
- **figures/** contains the manuscript and supplementary figures in PDF format.
- **reproduction/** contains the endpoint preparer and staged runner.
- **scripts/** and **src/** contain the model and evaluation code.

For the endpoint workbook, obtain **Supplemental Table S1** from [Liu et al.,
Cell 2018](https://doi.org/10.1016/j.cell.2018.02.052). The preparer checks the
workbook MD5 (a4591b2dcee39591f59e5e25a6ce75fa). Use its absolute path wherever
/path/to/TCGA-CDR-SupplementalTableS1.xlsx appears below.

Generated intermediate outputs can include exploratory analyses and unfiltered
rows. Use the versioned supplementary_data/ folder for the exact reported
tables. Earlier annotations and analyses are documented at the historical
[source revision](https://github.com/rnmcln/glioma-survival-benchmark/tree/c05fa5d).

The MIT licence applies to the software. Cite this version and the associated
article. See CITATION.cff for the software citation.

## Reproduce the analysis

Use separate source-preparation and analysis folders. From a clean copy of
this release, copy the release into a sibling folder before downloading data.
For example, run the following inside an extracted release folder. Keep the
two working copies separate.

Python requirements are in requirements.txt. The versions from the
reported macOS arm64 run are recorded under reproduction/. On that platform
replace tensorflow-cpu==2.17.1 with tensorflow==2.17.1 before installing.

```sh
cp -R . ../benchmark-analysis
python3 -m venv ../benchmark-env
. ../benchmark-env/bin/activate
python -m pip install -r requirements.txt
python scripts/01_download_data.py
python scripts/02_build_cohort.py
python reproduction/prepare_benchmark.py /path/to/TCGA-CDR-SupplementalTableS1.xlsx data/processed ../benchmark-analysis ../benchmark-analysis/cdr_endpoint_counts.csv
python reproduction/run_staged.py benchmark ../benchmark-analysis
mkdir -p ../benchmark-analysis/reviewed-figures
python reproduction/figure1_benchmark.py ../benchmark-analysis/results/tables ../benchmark-analysis/reviewed-figures
python reproduction/redraw_b.py ../benchmark-analysis ../benchmark-analysis/reviewed-figures
```

Data are downloaded from the pinned GlioVis objects, cBioPortal and GEO.
The distributed tables omit the pooled internal Harrell C calculated from
raw scores across folds. Within-fold estimates remain in survival_cindex.csv.
