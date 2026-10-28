"""Inventory completed local runs, retaining duplicate and exclusion evidence."""

import argparse
from pathlib import Path

from ra_lsbo.catalog import export_catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = parser.parse_args()
    destination, report = export_catalog(args.roots, args.output)
    print("Catalog:", destination / "report.json")
    print("Unique terminal runs:", report["unique_terminal_runs"])
    print("Unique evaluations:", report["unique_evaluations"])
    for identity, paths in report["compatible_groups"].items():
        print("Contract", identity[:12], "runs", len(paths))
