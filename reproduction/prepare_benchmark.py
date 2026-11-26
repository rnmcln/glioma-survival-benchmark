#!/usr/bin/env python3
"""Stage frozen benchmark cohorts and apply TCGA-CDR OS at the analysis input."""
import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("processed", type=Path)
    parser.add_argument("staged_repository", type=Path)
    parser.add_argument("audit_csv", type=Path)
    args = parser.parse_args()

    if hashlib.md5(args.workbook.read_bytes()).hexdigest() != "a4591b2dcee39591f59e5e25a6ce75fa":
        raise ValueError("Unexpected TCGA-CDR workbook")
    cdr = pd.read_excel(args.workbook, sheet_name="TCGA-CDR",
                        usecols=["bcr_patient_barcode", "type", "OS", "OS.time"])
    cdr = cdr.loc[cdr.type.isin(["GBM", "LGG"])].set_index("bcr_patient_barcode")
    if not cdr.index.is_unique:
        raise ValueError("Duplicate TCGA-CDR patient barcodes")
    legacy = pd.read_csv(args.processed / "clinical_tcga.csv", dtype={"sample": str})
    ids = legacy["sample"].str.upper().str.replace(".", "-", regex=False).str[:12]
    if len(ids) != 664 or not ids.is_unique or not ids.isin(cdr.index).all():
        raise ValueError("Unexpected TCGA benchmark roster or CDR join")
    outcomes = cdr.loc[ids].reset_index(drop=True)
    valid = outcomes["OS"].isin([0, 1]) & np.isfinite(outcomes["OS.time"]) & outcomes["OS.time"].ge(0)
    before = legacy.loc[valid].copy()
    after = before.copy()
    after["os_event"] = outcomes.loc[valid, "OS"].astype(int)
    after["os_months"] = outcomes.loc[valid, "OS.time"] / 30.4375
    if (len(after), int(before.os_event.sum()), int(after.os_event.sum())) != (663, 239, 245):
        raise ValueError("Unexpected paired endpoint counts")
    output = args.staged_repository / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    for cohort in ("rembrandt", "gravendeel"):
        name = f"clinical_{cohort}.csv"
        shutil.copyfile(args.processed / name, output / name)
    for cohort in ("tcga", "rembrandt", "gravendeel"):
        name = f"expr_panel_{cohort}.csv"
        shutil.copyfile(args.processed / name, output / name)
    after.to_csv(output / "clinical_tcga.csv", index=False)
    summary_name = "harmonisation_summary.csv"
    summary = pd.read_csv(args.processed / summary_name)
    tcga_row = summary["cohort"].eq("tcga")
    if tcga_row.sum() != 1 or int(summary.loc[tcga_row, "n_analysed"].iloc[0]) != len(legacy):
        raise ValueError("Benchmark harmonisation counts differ from the legacy roster")
    summary.loc[tcga_row, "n_analysed"] = len(after)
    summary.loc[tcga_row, "excluded_no_outcome"] += len(legacy) - len(after)
    summary.to_csv(output / summary_name, index=False)


    # Use the repository's binary endpoint implementation at all three horizons.
    import sys
    sys.path.insert(0, str(args.staged_repository.resolve()))
    from src.endpoints import make_binary_endpoint

    results = []
    for endpoint, data in (("legacy_paired", before), ("tcga_cdr", after)):
        for horizon in (12, 24, 60):
            _, summary = make_binary_endpoint(data, horizon)
            results.append(dict(endpoint=endpoint, deaths=int(data.os_event.sum()), **summary))
    pd.DataFrame(results).to_csv(args.audit_csv, index=False)


if __name__ == "__main__":
    main()
