"""Fit risk and GP heads chronologically on a completed LSBO journal; no MDA."""

import argparse
from pathlib import Path

import torch

from ra_lsbo.replay import run_replay


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-project", type=Path,
                        default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--output", type=Path, default=root / "runs")
    parser.add_argument("--initial", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gp-steps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-per-class", type=int, default=3)
    parser.add_argument("--trees", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_replay(args.source, args.base_project, args.output, initial=args.initial,
               epochs=args.epochs, gp_steps=args.gp_steps, seed=args.seed,
               min_per_class=args.min_per_class, trees=args.trees)


if __name__ == "__main__":
    main()
