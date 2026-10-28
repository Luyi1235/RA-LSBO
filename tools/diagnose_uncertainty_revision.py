"""Quantify acquisition failure signatures from immutable prior experiment files."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.budget import digest
from ra_lsbo.system_experiment import save_new


def lines(path):
    return [load_json(line) for line in path.read_bytes().splitlines()]


def main():
    studies = [
        ROOT/"runs/adaptive_research_20261021T071809Z_6fba5696",
        ROOT/"runs/margin_research_20261021T072847Z_79e09810"]
    results = []
    for study in studies:
        for case in ("five_mode", "sparse_interior"):
            for method in ("adaptive_full", "adaptive_no_information"):
                proposals, observations, endpoints = [], [], []
                for path in sorted((study/case).glob(method+"_*")):
                    if not path.is_dir():
                        continue
                    p, r = lines(path/"proposals.jsonl"), lines(path/"observations.jsonl")
                    if len(p) != len(r):
                        raise ValueError("Unequal prediction/outcome history")
                    proposals += p
                    observations += r
                    endpoints.append(load_json((path/"summary.json").read_bytes()))
                modes = {}
                for mode in ("information", "development"):
                    indices = [i for i, p in enumerate(proposals) if p["selection"]["mode"] == mode]
                    modes[mode] = {
                        "calls": len(indices),
                        "feasible": sum(observations[i]["risk"]["feasible_label"] is True for i in indices),
                        "predicted_feasibility_below_0_1": sum(
                            proposals[i]["selection"]["used_final"] < .1 for i in indices)}
                results.append({
                    "study": study.name, "case": case, "method": method,
                    "modes": modes,
                    "mean_improvement_percent": float(np.mean([r["improvement_percent"] for r in endpoints])),
                    "warning": "Mode groups are adaptively selected; not a randomized causal comparison"})
    historical = studies[1]/"physical_retrospective_normalized_v2/joint_full_forecasts.json"
    data = load_json(historical.read_bytes())
    marginals = np.asarray(data["raw"]["constraint_marginals"])
    labels = np.asarray([r["label"] for r in data["test"]])
    union = np.maximum(0., 1.-(1.-marginals).sum(1))
    product = marginals.prod(1)
    unique = []
    seen = set()
    for i, row in enumerate(data["test"]):
        key = digest(row["candidate"])
        if key not in seen:
            seen.add(key)
            unique.append(i)
    summaries = {}
    for scope, indices in (("all_forecasts", np.arange(len(labels))),
                           ("unique_designs", np.asarray(unique))):
        zero = union[indices] <= 1e-12
        summaries[scope] = {
            "count": len(indices), "union_zero_count": int(zero.sum()),
            "actually_feasible_in_union_zero": int(labels[indices][zero].sum()),
            "union_mean": float(union[indices].mean()),
            "product_mean": float(product[indices].mean()),
            "product_positive_in_union_zero": int((product[indices][zero] > 1e-12).sum())}
    output = ROOT/"runs/uncertainty_rootcause_20261021"
    output.mkdir(exist_ok=False)
    save_new(output/"diagnostics.json", {
        "additional_real_mda_calls": 0, "new_controlled_evaluations": 0,
        "acquisition_modes": results, "historical_constraint_combination": summaries,
        "constraint_names": data["model"]["diagnostics"]["constraint_names"],
        "constraint_count": marginals.shape[1],
        "correction": "Existing training tables already exclude analytical constraint targets.",
        "union_caveat": "A lower-bound expression on estimated marginals, not the joint posterior probability",
        "product_caveat": "Independent-head model assumption; not validated physical independence",
        "sources": [str(s) for s in studies]+[str(historical)]})
    print(json.dumps(load_json((output/"diagnostics.json").read_bytes()), indent=2))


if __name__ == "__main__":
    main()
