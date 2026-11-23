"""Harmonise cohorts; write cohort description, missingness, feature
availability and the comparison of included versus early-censored patients."""
import _path  # noqa: F401
import pandas as pd
from src import config
from src.endpoints import make_binary_endpoint
from src.harmonise import harmonise_all, load_cohort
from src.utils import get_logger

log = get_logger()
VARS = ["age", "sex", "grade", "histology", "idh_status", "codel_1p19q", "mgmt_status"]


def describe(g):
    _, i2 = make_binary_endpoint(g, 24)
    has = lambda v: g[v].notna().any()
    pct = lambda v, lvl: round(100 * (g[v] == lvl).sum() / g[v].notna().sum(), 1) if has(v) else None
    return {
        "n": len(g),
        "age_median": round(g.age.median(), 1) if has("age") else None,
        "age_q1": round(g.age.quantile(.25), 1) if has("age") else None,
        "age_q3": round(g.age.quantile(.75), 1) if has("age") else None,
        "female_pct": pct("sex", "Female"),
        "gbm_n": int((g.histology == "GBM").sum()),
        "astro_n": int((g.histology == "Astrocytoma").sum()),
        "oligo_n": int((g.histology == "Oligodendroglioma").sum()),
        "mixed_n": int((g.histology == "Mixed/OA").sum()),
        "grade2_n": int((g.grade == 2).sum()), "grade3_n": int((g.grade == 3).sum()),
        "grade4_n": int((g.grade == 4).sum()), "grade_missing_n": int(g.grade.isna().sum()),
        "idh_mut_pct": pct("idh_status", "Mutant"),
        "codel_pct": pct("codel_1p19q", "Codeleted"),
        "mgmt_meth_pct": pct("mgmt_status", "Methylated"),
        "deaths_n": int(g.os_event.sum()),
        "followup_median_months": round(g.os_months.median(), 1),
        "labelled_2y": i2["n_labelled"], "deaths_2y": i2["n_deaths"],
        "event_rate_2y": round(i2["event_rate"], 3),
        "censored_before_2y": i2["n_censored_before_horizon"],
    }


def main():
    _, summary = harmonise_all(save=True)
    log.info("\n%s", summary.to_string(index=False))
    rows, miss = [], []
    for c in config.GLIOVIS_FILES:
        g = load_cohort(c, with_expr=False)
        rows.append({"cohort": c, **describe(g)})
        for v in VARS:
            miss.append({"cohort": c, "variable": v, "missing_fraction": round(g[v].isna().mean(), 3)})
    pd.DataFrame(rows).to_csv(config.TABLES / "cohort_characteristics.csv", index=False)
    pd.DataFrame(miss).pivot(index="variable", columns="cohort", values="missing_fraction") \
        .to_csv(config.TABLES / "missingness.csv")

    t = load_cohort("tcga", with_expr=False)
    cens = (t.os_event == 0) & (t.os_months < 24)
    comp = [{"group": "labelled at 2 years", **describe(t[~cens])},
            {"group": "censored before 2 years", **describe(t[cens])}]
    pd.DataFrame(comp).to_csv(config.TABLES / "included_vs_censored.csv", index=False)
    log.info("cohort tables written")


if __name__ == "__main__":
    main()
