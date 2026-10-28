"""Exercise independent cores on existing controlled records, without new truth."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.risk_core.adapter import global_pool, observable_row
from ra_lsbo.risk_core.optimizer import CoreSelector, FAMILIES
from ra_lsbo.system_experiment import save_new
from tools.run_risk_core_selection import BASE, lines, problem_for, read


def check(output):
    torch.set_num_threads(1)
    previous = ROOT/"runs/discovery_policy_20261025"
    problem = problem_for("partial_sparse")
    checks = []
    for seed in (2001, 2004, 2008):
        initial = read(previous/("initial_"+str(seed))/"rows.json")
        online = lines(previous/("posterior_full_"+str(seed))/"observations.jsonl")
        for length in (4, len(initial)+len(online)):
            history = [observable_row(r, True) for r in (initial+online)[:length]]
            pool = global_pool(problem, seed+900000, 192, set())
            for family in FAMILIES:
                for integrated in (False, True):
                    selector = CoreSelector(family, problem, BASE, seed,
                                            latent=integrated, stages=integrated)
                    candidate, metadata = selector.propose(history, pool, 0)
                    checks.append({"seed": seed, "prefix_length": length, "family": family,
                                   "integrated": integrated, "candidate_key": metadata["candidate_key"],
                                   "fallback": metadata["fallback"]})
        print("Existing-history checks", seed, flush=True)
    save_new(output, {"status": "passed", "checks": checks,
                      "new_truth_evaluations": 0, "new_physical_mda_calls": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output.resolve())
