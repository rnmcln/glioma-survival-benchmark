"""Download the public source data (GlioVis objects at a pinned commit; GEO
series matrices; cBioPortal TCGA clinical table)."""
import sys
import _path  # noqa: F401
from src.data_download import download_all

if __name__ == "__main__":
    download_all(force="--force" in sys.argv)
