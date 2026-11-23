"""Record SHA-256 checksums of the raw inputs and the pinned GlioVis commit."""
import hashlib
import _path  # noqa: F401
from src import config
from src.utils import save_json

if __name__ == "__main__":
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in sorted(config.RAW.glob("*")) if p.is_file()}
    save_json({"gliovis_repository": "msquatrito/shiny_GlioVis",
               "gliovis_commit": config.GLIOVIS_COMMIT, "files": files},
              config.DICTIONARY / "data_manifest.json")
    print(f"{len(files)} files recorded")
