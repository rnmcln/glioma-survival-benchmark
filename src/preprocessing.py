"""Preprocessing fitted inside the training data only, and feature resolution."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def split_types(df: pd.DataFrame, features: list[str]):
    num = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    cat = [f for f in features if f not in num]
    return num, cat


def make_preprocessor(df: pd.DataFrame, features: list[str], scale: bool = True,
                      fixed_levels: bool = True) -> ColumnTransformer:
    """Median imputation and standardisation (numeric); mode imputation and
    one-hot encoding (categorical). Categorical levels are fixed a priori so the
    encoded design has the same columns in every fold and cohort."""
    num, cat = split_types(df, features)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    parts = []
    if num:
        parts.append(("num", Pipeline(num_steps), num))
    if cat:
        cats = ([config.CATEGORICAL_LEVELS[c] for c in cat]
                if fixed_levels and all(c in config.CATEGORICAL_LEVELS for c in cat) else "auto")
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(parts, remainder="drop")


def feature_list(feature_set: str, df: pd.DataFrame) -> list[str]:
    if feature_set == "C":
        return list(df.attrs.get("genes", []))
    return list(config.FEATURE_SETS[feature_set])


def missing_fraction(df: pd.DataFrame, f: str) -> float:
    return float(df[f].isna().mean()) if f in df.columns else 1.0


def resolve_features(feature_set: str, *cohorts: pd.DataFrame):
    """Apply the prespecified missingness rule across all cohorts involved.

    A variable is used only if it is recorded in every cohort and its missing
    fraction is at most the threshold in each, unless it is essential (age,
    grade), in which case it is kept wherever it is not entirely absent.
    Returns (features, report).
    """
    feats = feature_list(feature_set, cohorts[0])
    kept, report = [], []
    for f in feats:
        fr = [missing_fraction(c, f) for c in cohorts]
        absent = any(x >= 1.0 for x in fr)
        over = any(x > config.MISSINGNESS_DROP_THRESHOLD for x in fr)
        keep = (not absent) and (not over or f in config.ESSENTIAL_VARS)
        report.append({"variable": f, "missing_fractions": [round(x, 3) for x in fr], "used": keep})
        if keep:
            kept.append(f)
    return kept, report
