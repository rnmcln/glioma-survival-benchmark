"""Flexible regression design used in the revised analyses (PROTOCOL.md,
section 13).

Compared with the original design (age and grade as linear terms), the
flexible design contains:
  * age as a restricted cubic spline with four knots at the 5th, 35th, 65th
    and 95th percentiles of the training data (Harrell's parameterisation,
    three columns);
  * WHO grade as a categorical variable (indicators for grades III and IV,
    reference grade II);
  * sex, histology, IDH status, 1p/19q codeletion and MGMT methylation as
    indicators, as in the original design;
  * two interactions chosen on clinical grounds: age (linear) x IDH-wildtype and
    grade (ordinal) x IDH-wildtype, which allow the age and grade gradients to
    differ between IDH-wildtype and IDH-mutant tumours.
Only the variables in the supplied feature list are used; interactions are
included only when both components are available. Imputation values (median
for numeric, mode for categorical) and knots are estimated on the training
data passed to fit(). All columns are standardised with training means and
standard deviations so that a single penalty applies on a common scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def rcs_basis(x, knots):
    """Restricted cubic spline basis (linear term plus k-2 non-linear terms)."""
    k = np.asarray(knots, float)
    x = np.asarray(x, float)
    cols = [x]
    tk, tk1 = k[-1], k[-2]
    scale = (tk - k[0]) ** 2
    pos = lambda u: np.clip(u, 0, None) ** 3
    for j in range(len(k) - 2):
        cols.append((pos(x - k[j]) - pos(x - tk1) * (tk - k[j]) / (tk - tk1)
                     + pos(x - tk) * (tk1 - k[j]) / (tk - tk1)) / scale)
    return np.column_stack(cols)


class FlexibleDesign:
    def __init__(self, features):
        self.features = list(features)

    def fit(self, df: pd.DataFrame):
        f = self.features
        self.med_ = {c: float(df[c].median()) for c in ("age", "grade") if c in f}
        self.mode_ = {c: df[c].mode().iloc[0] for c in f if c not in ("age", "grade")}
        if "age" in f:
            a = df["age"].fillna(self.med_["age"]).values
            self.knots_ = np.percentile(a, [5, 35, 65, 95])
        Z = self._raw(df)
        self.mu_ = Z.mean(axis=0)
        self.sd_ = Z.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        return self

    def _raw(self, df):
        f = self.features
        cols, names = [], []
        if "age" in f:
            a = df["age"].fillna(self.med_["age"]).values.astype(float)
            B = rcs_basis(a, self.knots_)
            cols += [B[:, j] for j in range(B.shape[1])]
            names += ["age"] + [f"age_rcs{j}" for j in range(1, B.shape[1])]
        if "grade" in f:
            g = df["grade"].fillna(self.med_["grade"]).round().clip(2, 4).values
            cols += [(g == 3).astype(float), (g == 4).astype(float)]
            names += ["grade_III", "grade_IV"]
        for c in ("sex", "histology", "idh_status", "codel_1p19q", "mgmt_status"):
            if c not in f:
                continue
            v = df[c].fillna(self.mode_[c]).values
            for lev in config.CATEGORICAL_LEVELS[c][1:]:
                cols.append((v == lev).astype(float))
                names.append(f"{c}_{lev}")
        if "idh_status" in f:
            wt = (df["idh_status"].fillna(self.mode_["idh_status"]).values == "WT").astype(float)
            if "age" in f:
                cols.append(df["age"].fillna(self.med_["age"]).values.astype(float) * wt)
                names.append("age_x_IDHwt")
            if "grade" in f:
                cols.append(df["grade"].fillna(self.med_["grade"]).values.astype(float) * wt)
                names.append("grade_x_IDHwt")
        self.names_ = names
        return np.column_stack(cols).astype(float)

    def transform(self, df):
        return (self._raw(df) - self.mu_) / self.sd_

    def fit_transform(self, df):
        return self.fit(df).transform(df)
