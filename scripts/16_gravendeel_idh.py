"""Post hoc analysis R7 (PROTOCOL.md, section 12): external comparison in
Gravendeel (set B) restricted to patients with recorded IDH status.

In Gravendeel, IDH status is missing for some patients and is imputed with the
most frequent TCGA category. This script re-evaluates the stored external
predictions (no refitting) in patients with recorded IDH status and in those
without, to examine whether the network-regression difference depends on the
imputed values.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd
from src import config
from src.cv import load
from src.evaluation import bootstrap_indices, paired_difference, point_metrics
from src.harmonise import load_cohort

PAIRS_BIN = [("tf_compact", "penalised_logistic"), ("tf_regularised", "penalised_logistic")]



def main():
    g = load_cohort("gravendeel", with_expr=False)
    rec = dict(zip(g["sample"].astype(str), g.idh_status.notna()))
    rows = []
    y, mean, _, sample = load("ext_gravendeel_B_2y")
    mask_all = np.array([rec[s] for s in sample])
    for label, mask in (("IDH recorded", mask_all), ("IDH missing", ~mask_all)):
        yy = y[mask]
        idx = bootstrap_indices(len(yy), yy)
        for a, b in PAIRS_BIN:
            d = paired_difference(yy, mean[a][mask], mean[b][mask], "auroc", idx)
            rows.append({"subset": label, "endpoint": "binary 24 months", "comparison": f"{a} - {b}",
                         "n": int(mask.sum()), "events": int(yy.sum()),
                         "auroc_a": point_metrics(yy, mean[a][mask])["auroc"],
                         "auroc_b": point_metrics(yy, mean[b][mask])["auroc"], **d})
    import importlib.util
    spec = importlib.util.spec_from_file_location("fc", str(config.ROOT / "scripts" / "13_full_cohort.py"))
    fc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fc)
    z = np.load(config.PREDICTIONS / "revision" / "full_cohort_gravendeel.npz")
    gs = g.set_index(g["sample"].astype(str)).loc[z["sample"]].reset_index(drop=True)
    m_all = gs.idh_status.notna().values
    j = fc.TIMES.index(fc.H)
    for label, mask in (("IDH recorded", m_all), ("IDH missing", ~m_all)):
        sub = gs[mask].reset_index(drop=True)
        for a, b in (("coxmlp_compact", "cox_penalised"), ("tf_compact", "penalised_logistic")):
            ra, rb = z[f"pred__{a}"][mask, j], z[f"pred__{b}"][mask, j]
            ma, mb = fc.metrics(sub, ra), fc.metrics(sub, rb)
            ci, _ = fc.boot_diff(sub, ra, rb, config.N_BOOTSTRAP, config.SEED)
            rows.append({"subset": label, "endpoint": "all patients, IPCW 24 months", "comparison": f"{a} - {b}",
                         "n": int(mask.sum()), "events": int(sub.os_event.sum()), "auroc_a": ma["td_auroc"],
                         "auroc_b": mb["td_auroc"], "delta": ma["td_auroc"] - mb["td_auroc"], "lo": ci[0], "hi": ci[1]})
    pd.DataFrame(rows).to_csv(config.TABLES / "gravendeel_idh_subsets.csv", index=False)


if __name__ == "__main__":
    main()
