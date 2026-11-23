"""Seeding, logging and JSON helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np

from . import config


def set_seed(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def configure_tensorflow(threads: int | None = None) -> None:
    """Single-threaded, deterministic TensorFlow execution."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    n = threads or int(os.environ.get("GSB_TF_THREADS", 1))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(n)
    except RuntimeError:
        pass
    tf.config.experimental.enable_op_determinism()


def get_logger(name: str = "gsb") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(config.LOGS / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
