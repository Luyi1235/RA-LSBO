"""Export risk labels from one archived LSBO run; never launches the simulator."""

import argparse
from pathlib import Path

from ra_lsbo.audit import export_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Run with manifest, journal and summary")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = parser.parse_args()
    destination, report = export_snapshot(args.source, args.output)
    print("Report:", destination / "report.json")
    print("Evaluations:", report["evaluations"])
    print("States:", report["states"])
    print("Valid objectives:", report["valid_objectives"])
    print("No MDA calls, no model fitting, no source writes.")


if __name__ == "__main__":
    main()
