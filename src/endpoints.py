"""Fixed-horizon binary overall-survival endpoint.

For horizon H (months): y = 1 if death at or before H; y = 0 if follow-up
reaches H; patients censored before H are excluded because their status at H
is unknown. Exclusion counts are returned for reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_binary_endpoint(df: pd.DataFrame, horizon: float):
    t = pd.to_numeric(df["os_months"], errors="coerce")
    e = pd.to_numeric(df["os_event"], errors="coerce")
    valid = t.notna() & e.notna()
    dead = valid & (e == 1) & (t <= horizon)
    alive = valid & (t >= horizon) & ~dead
    censored = valid & (e == 0) & (t < horizon)
    y = pd.Series(np.nan, index=df.index)
    y[dead] = 1.0
    y[alive] = 0.0
    keep = y.notna()
    out = df.loc[keep].copy()
    out["y"] = y[keep].astype(int)
    info = {
        "horizon_months": horizon,
        "n_input": int(len(df)),
        "n_censored_before_horizon": int(censored.sum()),
        "n_labelled": int(keep.sum()),
        "n_deaths": int(out["y"].sum()),
        "n_alive": int((out["y"] == 0).sum()),
        "event_rate": float(out["y"].mean()) if len(out) else float("nan"),
    }
    return out, info
