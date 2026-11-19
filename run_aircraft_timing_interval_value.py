"""Run a prospective, equal-cost comparison of three first-feedback selectors."""

import argparse
import json
from pathlib import Path
from statistics import mean

from aircraft_timing_information import PROBABILITY_FLOOR, score_information, supports_timing
from aircraft_timing_decision_value import DECISION_ARMS, decision_choices
from aircraft_timing_interval_value import deficit_edges
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from ra_lsbo.partial_bo.problems import candidate_pool
from run_aircraft_initial_budget import (
    BenchmarkAircraft, brier_by_configuration, candidate_roster, evidence_hashes, nested_indices,
    paired_interval, read_ledger,
)
from run_aircraft_initial_timing_diagnostic import completion_probabilities, fit_completion_heads
from run_aircraft_risk_cold_start import discovery_outcome, read, select, source_commit
from run_aircraft_timing_first_feedback import probability
from run_aircraft_timing_online import structural_margins


ARMS = ("greedy", "sign_value", "interval_value")
HEAD = "timing_interval"
PROFILES = {
    "timing-value-smoke": (4., 0., (28760300,), 16),
    "timing-value-headwind": (4., 0., tuple(range(28780100, 28784900, 100)), 48),
    "timing-value-payload": (0., 24., tuple(range(28780100, 28784900, 100)), 48),
}
DECISION_PROFILES = {
    # One agreeing seed and one separating the two-call score from greed.
    "timing-decision-smoke": (4., 0., (28760300, 28780900), 16),
    "timing-decision-headwind": (4., 0., tuple(range(28800100, 28809700, 100)), 48),
    "timing-decision-payload": (0., 24., tuple(range(28800100, 28809700, 100)), 48),
}


def plan(name):
    wind, payload, seeds, panel = {**PROFILES, **DECISION_PROFILES}[name]
    sizes = [8, 16]
    decision = name in DECISION_PROFILES
    arms = DECISION_ARMS if decision else ARMS
    calls = 2 if decision else 1
    spec = {"name": name, "headwind_m_s": wind, "payload_delta_kg": payload,
            "seeds": list(seeds), "pool": 128, "heldout": panel, "structural_screen": True,
            "total_budget": max(sizes), "initial_sizes": sizes, "next_calls_per_arm": calls,
            "arms": list(arms), "probability_floor": PROBABILITY_FLOOR,
            "max_aircraft_calls": len(seeds)*(max(sizes)+panel+len(arms)*len(sizes)*calls),
            "primary": {"initial": 8, "reference": "interval_value", "control": "sign_value",
                        "metric": "screened_heldout_brier_after_feedback"}}
    if decision:
        spec["second_pool"] = 64
        spec["primary"] = {"initial": 8, "reference": "decision_value", "control": "greedy",
                           "metric": "any_usable_by_two"}
    return spec


def select_arms(head, query, candidates, settings):
    p, greedy, eligible, sign_scores = score_information(head, query, candidates)
    edges = [deficit_edges(candidate, settings) for candidate in candidates]
    _, _, interval_eligible, interval_scores = score_information(head, query, candidates, edges)
    if not (eligible == interval_eligible).all():
        raise ValueError("Information arms must have the same probability band")
    shortlisted = [candidates[i] for i in eligible]
    indices = (greedy, eligible[select(shortlisted, sign_scores)],
               eligible[select(shortlisted, interval_scores)])
    choices = {arm: {"index": int(index), "probability": float(p[index])}
               for arm, index in zip(ARMS, indices)}
    for arm, scores in (("sign_value", sign_scores), ("interval_value", interval_scores)):
        choices[arm]["model_expected_brier_reduction"] = float(
            scores[eligible.tolist().index(choices[arm]["index"])])
    return choices, len(eligible)


def first_feedback(problem, seed, history, pool, panel, obtain, second=()):
    initial = len(history)
    row = {"seed": seed, "initial": initial}
    if any(record.usable_feasible for record in history):
        return {**row, "state": "initially_solved"}
    if not all(map(supports_timing, history)):
        return {**row, "state": "unsupported_initial_feedback"}
    used = {record.observation.design_id for record in history}
    candidates = [candidate for candidate in pool if digest(candidate) not in used]
    targets = panel+list(second)
    boundary = len(candidates)
    try:
        heads, query, _ = fit_completion_heads(problem, history, candidates+targets, seed, methods=(HEAD,))
        head = heads[HEAD]
        if second:
            choices, in_band = decision_choices(head, query[:boundary], query[boundary+len(panel):],
                                                candidates, problem.settings)
        else:
            choices, in_band = select_arms(head, query[:boundary], candidates, problem.settings)
        before = probability(head, query[boundary:]).tolist()
    except (ModelNumericalFailure, FloatingPointError):
        return {**row, "state": "initial_selection_model_failure"}
    observed = {}
    for arm, info in choices.items():
        candidate = candidates[info.pop("index")]
        key = digest(candidate)
        if key not in observed:
            record = obtain(candidate, f"{seed}/{initial}/{arm}/first", "next")
            outcome = discovery_outcome(record.to_dict())
            after = before
            update = "unsupported_feedback"
            if supports_timing(record):
                try:
                    predictions, _ = completion_probabilities(problem, history+[record], targets, seed,
                                                               methods=(HEAD,))
                    after = predictions[HEAD].tolist()
                    update = "updated"
                except ModelNumericalFailure:
                    update = "update_model_failure"
            observed[key] = {"design_id": key, "outcome": outcome, "usable": record.usable_feasible,
                             "update": update, "forecast": after[:len(panel)]}
            if second:
                index = select(second, after[len(panel):])
                observed[key]["second_choice"] = second[index]
                observed[key]["second_probability"] = after[len(panel)+index]
        info.update(observed[key])
    if second:
        second_observed = {}
        for arm, info in choices.items():
            candidate = info.pop("second_choice")
            key = digest(candidate)
            if key not in second_observed:
                record = obtain(candidate, f"{seed}/{initial}/{arm}/second", "second")
                second_observed[key] = {"design_id": key, "outcome": discovery_outcome(record.to_dict()),
                                        "usable": record.usable_feasible}
            info["second"] = {**second_observed[key], "probability": info.pop("second_probability")}
            info["any_usable_by_two"] = info["usable"] or info["second"]["usable"]
    return {**row, "state": "paired", "candidates": len(candidates), "in_band": in_band,
            "before_forecast": before[:len(panel)], "arms": choices}


def evaluate_panel(problem, row, panel):
    if row["state"] != "paired":
        return
    raw = [discovery_outcome(record.to_dict()) for record in panel]
    labels = [None if outcome == "unknown" else outcome != "abort" for outcome in raw]
    screened = [label if all(margin <= 0. for margin in structural_margins(problem, ref.candidate).values())
                else None for label, ref in zip(labels, panel)]
    before = row.pop("before_forecast")
    row["screened_panel_count"] = sum(label is not None for label in screened)
    row["heldout_brier_before"] = {key: brier_by_configuration(before, panel, target)
                                  for key, target in (("all", labels), ("screened", screened))}
    scores = {}
    for item in row["arms"].values():
        forecast = item.pop("forecast")
        if item["design_id"] not in scores:
            scores[item["design_id"]] = {key: brier_by_configuration(forecast, panel, target)
                                         for key, target in (("all", labels), ("screened", screened))}
        item["heldout_brier_after"] = scores[item["design_id"]]


def campaign(spec, problem, obtain):
    rows = []
    for seed in spec["seeds"]:
        pool, panel = candidate_roster(problem, spec, seed)
        second = []
        if "second_pool" in spec:
            raw = candidate_pool(problem, seed+3, spec["second_pool"], map(digest, pool+panel))[0]
            second = [candidate for candidate in raw if all(
                margin <= 0. for margin in structural_margins(problem, candidate).values())]
            if not second:
                raise ValueError("No structurally eligible follow-up candidate")
        order = nested_indices(problem, pool, max(spec["initial_sizes"]))
        initial = [obtain(pool[i], f"{seed}/initial/{j}", "initial") for j, i in enumerate(order)]
        pairs = [first_feedback(problem, seed, initial[:n], pool, panel, obtain, second)
                 for n in spec["initial_sizes"]]
        labels = [obtain(candidate, f"{seed}/heldout/{j}", "heldout") for j, candidate in enumerate(panel)]
        for row in pairs:
            evaluate_panel(problem, row, labels)
            if second:
                row["second_pool_count"] = len(second)
        rows.extend(pairs)
        print(spec["name"], seed, [row["state"] for row in pairs], flush=True)
    return rows


def effect(values):
    return {"n": len(values), "mean": mean(values) if values else None,
            "paired_95": paired_interval(values)}


def contrast(rows, reference, control):
    pairs = [(row["arms"][reference], row["arms"][control]) for row in rows]
    brier = {}
    for target in ("all", "screened"):
        losses = [(left["heldout_brier_after"][target], right["heldout_brier_after"][target])
                  for left, right in pairs]
        brier[target] = effect([a-b for a, b in losses if a is not None and b is not None])
    result = {"reference": reference, "control": control,
            "changed_first": sum(a["design_id"] != b["design_id"] for a, b in pairs),
            "brier_after_delta": brier,
            "first_usable_delta": effect([int(a["usable"])-int(b["usable"]) for a, b in pairs])}
    if pairs and "second" in pairs[0][0]:
        result["changed_second"] = sum(a["second"]["design_id"] != b["second"]["design_id"]
                                       for a, b in pairs)
        result["second_usable_delta"] = effect([int(a["second"]["usable"])-int(b["second"]["usable"])
                                                for a, b in pairs])
        result["any_usable_by_two_delta"] = effect([int(a["any_usable_by_two"])-int(b["any_usable_by_two"])
                                                    for a, b in pairs])
    return result


def run(spec, sha, folder):
    source_commit(sha, clean=True)
    if Path(__file__).resolve().parent in (folder, *folder.parents):
        raise ValueError("Write evidence outside the source worktree")
    folder.mkdir(parents=True)
    problem = BenchmarkAircraft(spec)
    save(folder/"plan.json", {"source_sha": sha, "spec": spec, "contract": problem.contract_id})
    ledger = Ledger(folder, spec["max_aircraft_calls"])
    campaign(spec, problem, lambda candidate, tag, role: ledger.evaluate(problem, candidate, tag, role))
    save(folder/"seal.json", evidence_hashes(folder))


def verify(folder, sha):
    source_commit(sha, clean=True)
    header = read(folder/"plan.json")
    spec = plan(header["spec"]["name"])
    problem = BenchmarkAircraft(spec)
    if (header != {"source_sha": sha, "spec": spec, "contract": problem.contract_id}
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("Plan or evidence seal differs")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) > spec["max_aircraft_calls"]:
        raise ValueError("Evaluator budget exceeded")
    position = 0

    def obtain(candidate, tag, role):
        nonlocal position
        position += 1
        number, record, recorded_role = calls[tag]
        if (position, candidate, role) != (number, record.candidate, recorded_role):
            raise ValueError("Ledger differs from the replayed acquisition")
        return record

    rows = campaign(spec, problem, obtain)
    if position != len(calls):
        raise ValueError("Unexpected calls outside the registered campaign")
    summary = {}
    comparisons = (("decision_value", "greedy"), ("gated_decision", "greedy"),
                   ("decision_value", "gated_decision")) if "second_pool" in spec else (
                   ("interval_value", "sign_value"), ("interval_value", "greedy"), ("sign_value", "greedy"))
    for n in spec["initial_sizes"]:
        matching = [row for row in rows if row["initial"] == n]
        paired = [row for row in matching if row["state"] == "paired"]
        summary[str(n)] = {
            "n": len(paired), "initially_solved": sum(row["state"] == "initially_solved" for row in matching),
            "unpaired_failures": sum(row["state"] not in ("paired", "initially_solved") for row in matching),
            "single_candidate_in_band": sum(row["in_band"] == 1 for row in paired),
            "comparisons": [contrast(paired, a, b) for a, b in comparisons]}
    status = "VERIFIED_TIMING_DECISION_VALUE" if "second_pool" in spec else "VERIFIED_TIMING_INTERVAL_VALUE"
    return {"status": status, "plan": spec, "source_sha": sha,
            "actual_calls": position, "rcaide_mda_calls": 0, "summary": summary, "per_seed": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--profile", choices=(*PROFILES, *DECISION_PROFILES))
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task in ("plan", "run") and not args.profile:
        parser.error("--profile is required for plan and run")
    if args.task == "plan":
        print(json.dumps(plan(args.profile), indent=2))
        return
    if not args.expected_sha or args.output is None:
        parser.error("--expected-sha and --output are required")
    output = args.output.resolve()
    if args.task == "run":
        run(plan(args.profile), args.expected_sha, output)
    else:
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide --input and write outside the sealed run")
        result = verify(args.input, args.expected_sha)
        save(output, result)
        print(result["status"], result["plan"]["name"], result["actual_calls"])


if __name__ == "__main__":
    main()
