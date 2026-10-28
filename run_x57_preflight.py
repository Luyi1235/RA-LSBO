"""Inspect physical-contract and calibration compatibility; never execute MDA."""

import argparse
from pathlib import Path

from ra_lsbo.preflight import export_preflight


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-project", type=Path, default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--active-run", type=Path)
    parser.add_argument("--output", type=Path, default=root / "runs")
    args = parser.parse_args()
    folder, report = export_preflight(
        args.output, source=args.source, base_project=args.base_project, config_path=args.config,
        calibration_report=args.calibration_report, active_run=args.active_run)
    print("Preflight report:", folder / "report.json")
    print("Historical data directly reusable:", report["historical_data_reusable_without_revalidation"])
    print("Real MDA enabled:", report["real_mda_enabled"])
    for issue in report["readiness_issues"]:
        print(" -", issue)
