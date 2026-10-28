"""Prepare or explicitly execute the frozen eight-call prospective pilot."""

import argparse
from pathlib import Path
import torch

from ra_lsbo.x57_pilot import execute_pilot, prepare_pilot


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path, metavar="SOURCE_RUN")
    mode.add_argument("--execute", type=Path, metavar="REQUEST")
    parser.add_argument("--base-project", type=Path, default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--output", type=Path, default=root / "runs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        folder, plan = prepare_pilot(args.prepare, args.base_project, args.output)
        print("Unapproved pilot:", folder / "request.json")
        print("Planning only; MDA calls:", plan["planning_reservations"]["charged_calls"])
    else:
        report = execute_pilot(args.execute)
        print("Pilot:", report["status"], report["ledger"])
