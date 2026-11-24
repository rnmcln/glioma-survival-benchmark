"""Internal repeated cross-validation in TCGA for the 13 internal settings.

  python scripts/03_internal_cv.py               original analysis (class-weighted networks)
  python scripts/03_internal_cv.py --unweighted  sensitivity analysis: networks without class weights
"""
import sys
import _path  # noqa: F401
from src import config
from src.cv import internal_cv
from src.models import MLPS
from src.settings import internal_settings
from src.utils import configure_tensorflow, get_logger, set_seed

log = get_logger()


def main():
    configure_tensorflow()
    set_seed()
    unweighted = "--unweighted" in sys.argv
    outdir = config.PREDICTIONS / ("unweighted" if unweighted else "")
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, label, df, feats, horizon, reps in internal_settings():
        if unweighted:
            internal_cv(tag, df, feats, horizon, reps, class_weight=False, only=MLPS, outdir=outdir)
        else:
            internal_cv(tag, df, feats, horizon, reps, include_ftt=(tag == "int_B_2y"), outdir=outdir)
    log.info("internal CV complete (%s)", "unweighted" if unweighted else "class-weighted")


if __name__ == "__main__":
    main()
