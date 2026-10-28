"""Validate frozen risk calibration on three chronological blocks; no MDA."""

import argparse
from pathlib import Path

import torch

from ra_lsbo.calibration_replay import run_calibration


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--calibration-count", type=int, required=True)
    parser.add_argument("--base-project", type=Path,
                        default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--output", type=Path, default=root / "runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--trees", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_calibration(args.source, args.base_project, args.output, args.train_count,
                    args.calibration_count, seed=args.seed, epochs=args.epochs, trees=args.trees)


if __name__ == "__main__":
    main()
