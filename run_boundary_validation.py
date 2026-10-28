"""Prepare the 120-call boundary validation or execute its separately approved request."""

import argparse
from pathlib import Path

import torch

from ra_lsbo.boundary_validation import execute_boundary_validation, prepare_boundary_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path, help="Terminal previous pilot directory")
    mode.add_argument("--execute", type=Path, help="Separately approved request.json")
    parser.add_argument("--transfer", type=Path, help="Same-contract transfer report used for duplicate exclusion")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare:
        if args.transfer is None:
            parser.error("--prepare requires --transfer")
        folder, plan = prepare_boundary_validation(args.prepare, args.transfer, args.output)
        print("Prepared unapproved request:", folder / "request.json")
        print("Actual MDA calls: 0; planned calls:", len(plan["proposals"]))
        print("Estimated solver hours:", plan["estimated_solver_hours_from_previous_pilot"])
    else:
        report = execute_boundary_validation(args.execute)
        print("Campaign:", report["status"], report["ledger"])
