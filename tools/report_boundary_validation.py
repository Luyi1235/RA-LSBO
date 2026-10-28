"""Write a terminal research report, including campaigns that stop before testing."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json


def number(value, digits=5):
    return "Unavailable" if value is None else f"{value:.{digits}f}"


def render_report(result, audit, analysis=None):
    evaluated = [item for item in result["results"] if item["risk"] is not None]
    states = Counter(item["risk"]["state"] for item in evaluated)
    known = [item for item in evaluated if item["risk"]["feasible_label"] is not None]
    feasible = [item for item in known if item["risk"]["feasible_label"] is True]
    violations = Counter(name for item in evaluated for name, value in item["risk"]["constraints"].items()
                         if item["risk"]["constraint_observed"][name] and value > 0)
    lines = [
        "# Boundary Validation Results", "",
        f"Campaign status: `{result['status']}`; stop reason: `{result.get('reason', 'none')}`.",
        f"Plan SHA-256: `{result['plan_sha256']}`.",
        f"Physical contract: `{result['contract_id']}`.", "",
        f"Real charged calls: {result['ledger']['charged_calls']}/120; "
        f"settled calls: {result['ledger']['settled_calls']}; "
        f"unresolved started calls: {result['ledger']['unresolved_started_calls']}.",
        f"Returned evaluations: {len(evaluated)}; known feasibility labels: {len(known)}; "
        f"unknown labels: {len(evaluated)-len(known)}; "
        f"feasible: {len(feasible)}; known nonfeasible: {len(known)-len(feasible)}.",
        f"State counts: `{json.dumps(dict(states), sort_keys=True)}`.",
        f"Observed constraint violations: `{json.dumps(dict(violations), sort_keys=True)}`. "
        "Multiple constraints can be violated by one design.", "",
        "| Phase | Returned | Feasible | Known nonfeasible | Unknown | Unexecuted |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in ("calibration", "test"):
        group = [item for item in evaluated if item["phase"] == phase]
        labels = [item["risk"]["feasible_label"] for item in group]
        lines.append(f"| {phase} | {len(group)} | {sum(y is True for y in labels)} | "
                     f"{sum(y is False for y in labels)} | {sum(y is None for y in labels)} | "
                     f"{60-len(group)} |")
    lines += ["", "Calibration outcomes fit the local sigmoid and are not its validation set.", ""]
    if analysis is None:
        lines += ["No held-out analysis is available. No claim of improved calibration is made.", ""]
    else:
        lines += [
            "## Held-Out Forecasts", "",
            "| Model | Scored test points | Brier | Log loss | Accepted at 0.8 | Known infeasible accepted | Unknown accepted |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, metrics in analysis["metrics"].items():
            gate = analysis["acceptance_at_0_8"][name]
            lines.append(f"| {name} | {metrics['samples']} | {number(metrics['brier'])} | "
                         f"{number(metrics['log_loss'])} | {gate['accepted']} | "
                         f"{gate['infeasible_among_accepted']} | {gate['unknown_accepted']} |")
        lines += ["", "A zero scored count means unavailable forecasts, not zero prediction error.", "",
                  "## Paired Loss Differences", "",
                  "Comparison minus reference; negative is better. Intervals are exploratory "
                  "stratified bootstrap ranges, not safety bounds.", ""]
        for pair in analysis["paired_brier_differences"]:
            if not pair["samples"]:
                continue
            low, high = pair["bootstrap_interval"]
            lines.append(
                f"- {pair['comparison']} vs {pair['reference']}: "
                f"{number(pair['mean_brier_difference'])}, [{number(low)}, {number(high)}], "
                f"n={pair['samples']}.")
        lines += ["", "See `analysis.json` for all strata, fixed acceptance counts, missingness and seed settings.",
                  "See `boundary_validation.png` for the verified figure.", ""]
    if feasible:
        best = min(feasible, key=lambda item: item["risk"]["objective"])
        lines += [f"Lowest observed feasible energy: {number(best['risk']['objective'], 4)} Wh "
                  f"at frozen candidate {best['index']}. This is descriptive; the campaign "
                  "is not an optimization comparison.", ""]
    lines += [
        "## Verification And Scope", "",
        f"Raw worker responses matched: {audit['raw_responses_matched']}; "
        f"unavailable error responses: {audit['unavailable_error_responses']}.",
        f"Frozen source files verified: {audit['frozen_sources_verified']}. "
        "Ledger replay, approved plan and current physical contract matched.", "",
        "The target is the frozen balanced boundary mixture. Do not extrapolate to the entire "
        "design space, a different evaluator, hidden failure mechanisms, or optimizer superiority.",
        "Positive-slope sigmoid calibration changes forecast values, not underlying score ranking.",
        "Unknown and unexecuted outcomes are not imputed. No started call was retried.", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    result, audit = [load_json((folder / name).read_bytes()) for name in ("result.json", "audit.json")]
    if (result["status"] not in ("complete", "stopped_by_policy", "stopped_evaluator_error")
            or audit["status"] != "passed" or audit["ledger"] != result["ledger"]):
        raise ValueError("Require a terminal audited campaign")
    for filename, expected in audit["input_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Audited input changed")
    analysis_path = folder / "analysis.json"
    analysis = load_json(analysis_path.read_bytes()) if analysis_path.exists() else None
    if analysis is not None:
        for filename, expected in analysis["analysis_source_sha256"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Analysis source changed")
    with (folder / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write(render_report(result, audit, analysis))
    print("Report:", folder / "RESULTS.md")
