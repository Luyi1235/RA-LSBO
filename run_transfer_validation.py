"""Run a predeclared, frozen cross-policy risk-probability check without MDA."""

import argparse
from pathlib import Path
import torch

from ra_lsbo.transfer_validation import run_transfer


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--targets", type=Path, nargs="+", required=True)
    parser.add_argument("--train-count", type=int, default=180)
    parser.add_argument("--calibration-count", type=int, default=60)
    parser.add_argument("--base-project", type=Path, default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--output", type=Path, default=root / "runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--trees", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_transfer(args.source, args.targets, args.base_project, args.output,
                 train_count=args.train_count, calibration_count=args.calibration_count,
                 seed=args.seed, epochs=args.epochs, trees=args.trees)
