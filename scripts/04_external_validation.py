"""External validation: fit on full labelled TCGA, apply to Gravendeel and
REMBRANDT. Features are restricted by the missingness rule applied to both
cohorts; the predictors used are recorded.

  python scripts/04_external_validation.py [--unweighted]
"""
import sys
import _path  # noqa: F401
from src import config
from src.cv import external_validation
from src.models import MLPS
from src.settings import external_settings
from src.utils import configure_tensorflow, get_logger, save_json, set_seed

log = get_logger()


def main():
    configure_tensorflow()
    set_seed()
    unweighted = "--unweighted" in sys.argv
    outdir = config.PREDICTIONS / ("unweighted" if unweighted else "")
    outdir.mkdir(parents=True, exist_ok=True)
    tcga, jobs = external_settings()
    reports = {}
    for tag, tgt, fs, tdf, feats, report in jobs:
        reports[tag] = {"features": feats, "missingness": report}
        if unweighted:
            external_validation(tag, tcga, tdf, feats, 24, class_weight=False, only=MLPS, outdir=outdir)
        else:
            external_validation(tag, tcga, tdf, feats, 24, include_ftt=(fs == "B"), outdir=outdir)
    if not unweighted:
        save_json(reports, config.METRICS / "external_feature_resolution.json")


if __name__ == "__main__":
    main()
