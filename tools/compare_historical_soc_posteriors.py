"""Compare posterior beliefs on historical queries, never new outcomes."""

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_bo.selector import PartialBOSettings, feasibility_probability
from ra_lsbo.soc_revision.acquisition import average, choose_acquisition
from ra_lsbo.soc_revision.optimizer import RevisionSettings
from ra_lsbo.soc_revision.posterior import RiskPosterior
from tools.run_partial_bo_development import BASE, lines, read, sha, write_new
from tools.run_partial_refinement_study import problem_for


def main():
    folder = ROOT/"archive/historical_research/2026-10-26/results/partial_refinement_20261026/optimization/aircraft_140km/ra_grouped_local_8202"
    destination = ROOT/"runs/soc_diagnosis_20261026/historical_prefix_comparison.json"
    history = [DesignRecord.from_dict(row) for row in read(folder/"manifest.json")["initial"]]
    initial_kinds = {kind: sum(row.observation.constraint("soc").kind == kind for row in history)
                     for kind in ("exact", "interval", "missing")}
    records = lines(folder/"observations.jsonl")
    problem, rows = problem_for("aircraft_140km"), []
    for iteration, (proposal, packet) in enumerate(zip(lines(folder/"proposals.jsonl"), records)):
        if proposal["prefix_digest"] != digest([row.to_dict() for row in history]):
            raise ValueError("Historical prefix changed")
        posterior = RiskPosterior(problem, BASE, history, 8202, RevisionSettings())
        prediction = posterior.predict([proposal["candidate"]])
        mixture = prediction["constraints"]["soc"]
        probability = float(average(mixture, feasibility_probability)[0])
        old = proposal["metadata"]
        index = next(index for index, candidate in enumerate(old["pool"]) if candidate == proposal["candidate"])
        previous = old["forecasts"]["constraints"]["soc"]
        prior_probability = float(feasibility_probability(previous["mean"], previous["variance"])[index])
        diagnostic = posterior.constraints["soc"].diagnostics()
        weights = np.asarray(diagnostic["weights"])
        rows.append({"iteration": iteration, "raw_history_digest": posterior.history_digest,
            "old_soc_probability": prior_probability, "mixture_soc_probability": probability,
            "termination": packet["diagnostics"]["termination_reason"],
            "kernel_weights": weights.tolist(), "selected_kernel": diagnostic["selected"],
            "selected_kernel_max_weight": float(weights.max()),
            "all_components_ep_converged": all(head.model.diagnostics["converged"]
                for head in posterior.constraints["soc"].heads if head is not None),
            "hypothetical_query_only": True})
        history.append(DesignRecord.from_dict(packet))
    write_new(destination, {"rows": rows, "initial_soc_observation_kinds": initial_kinds,
        "new_truth_calls": 0, "new_rcaide_mda_calls": 0,
        "source_bindings": {str(folder/name): sha(folder/name) for name in (
            "manifest.json", "proposals.jsonl", "observations.jsonl")},
        "tool_sha256": sha(__file__),
        "interpretation": "Fixed historical queries and prefixes; not counterfactual optimization outcomes"})
    for row in rows:
        print(row["iteration"]+1, row["termination"], round(row["old_soc_probability"], 4),
              round(row["mixture_soc_probability"], 4), row["all_components_ep_converged"], flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
