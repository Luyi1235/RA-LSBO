"""Test whether current bounds inferred from stopped traces aid usable-design ranking."""

import argparse
from collections import Counter
import math
from pathlib import Path
from statistics import mean

import numpy as np

from aircraft_current_state_transfer import STATE_TRANSFER, transferred_current_probability
from aircraft_risk_discovery import fit_stage_head
from aircraft_timing_information import supports_timing
from aircraft_timing_usable_value import FEASIBILITY, TIMING, forecast
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import save
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure, StableGroupedHead
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_initial_feedback_diagnostic import (
    archived_seed_panels, first_ranking, load_pr122_profile, passes_structural_screen,
)
from run_aircraft_initial_timing_diagnostic import fit_completion_heads
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, read, source_commit
from run_aircraft_timing_decision_audit import INPUT_HASHES, INPUT_RESULT, INPUT_SOURCE
from run_aircraft_timing_interval_value import (
    BenchmarkAircraft, brier_by_configuration, effect, evidence_hashes, plan, read_ledger,
)
from run_aircraft_timing_usable_value import DERATED_CURRENT, with_current_limit


PR122_RESULT = "aac24f5a96a39ef26908ac21414382783e89dc73"
METHODS = ("interval_completion", "completed_sign", "completed_exact", "exact_plus_prefix")
PROFILES = {
    "pr135-headwind": "timing-decision-headwind",
    "pr135-payload": "timing-decision-payload",
    "pr122-headwind": "pilot-headwind-48",
    "pr122-payload": "pilot-payload-48",
}


def peak_current_lower(trace, initial_soc, capacity_ah):
    """Peak current is at least the largest observed segment-average current."""
    if not math.isfinite(capacity_ah) or capacity_ah <= 0. or not 0. <= initial_soc <= 1.:
        raise ValueError("Invalid initial charge")
    last_t, last_soc, lower = 0., initial_soc, 0.
    for stage in trace:
        time, soc = stage["elapsed_s"], stage["soc"]
        if not (math.isfinite(time) and last_t <= time and 0. <= soc <= last_soc+1e-12):
            raise ValueError("Invalid discharge trace")
        if time == last_t:
            if not math.isclose(last_soc, soc, abs_tol=1e-12, rel_tol=0.):
                raise ValueError("Discharge without elapsed time")
        else:
            lower = max(lower, capacity_ah*3600.*(last_soc-soc)/(time-last_t))
        last_t, last_soc = time, soc
    return lower


def current_observations(problem, records):
    """Build exact current margins and optional lower-censored prefix margins."""
    exact_only, with_prefix = [], []
    nominal_limit = problem.battery.CELL_MAX_A
    limit = DERATED_CURRENT*nominal_limit
    capacity = problem.battery.CELL_AH*problem.scenario.capacity_retention
    for record in records:
        key = record.observation.evidence_id + ":current_margin_audit"
        missing = OutputObservation("missing", key, reason="peak_current_not_observed")
        diagnostics = record.diagnostics
        lower = peak_current_lower(diagnostics["trace"], problem.scenario.initial_soc, capacity)
        if diagnostics["mission_completed"]:
            exact = record.observation.constraint("current")
            if exact.kind != "exact":
                raise ValueError("Completed trajectory lacks exact peak current")
            amps = (1.+exact.value)*nominal_limit
            if lower > amps+1e-8:
                raise ValueError("Trace-derived lower bound exceeds measured peak current")
            obs = OutputObservation("exact", key, value=amps/limit-1.)
            exact_only.append(obs)
            with_prefix.append(obs)
        else:
            exact_only.append(missing)
            with_prefix.append(OutputObservation("interval", key, lower=(lower-1e-8)/limit-1.)
                               if lower > 0. else missing)
    return exact_only, with_prefix


def prediction_metrics(points, candidates, labels, completed_labels, completion, feasibility):
    scores = completion*feasibility
    return {
        **first_ranking(candidates, scores, labels),
        "usable_brier": brier_by_configuration(scores, points, [label == "usable" for label in labels]),
        "conditional_brier": brier_by_configuration(feasibility, points, completed_labels),
    }


def evaluate_prefix(problem, seed, history, panel, *, transfer_audit=False):
    current_history = [with_current_limit(r, DERATED_CURRENT) for r in history]
    observed = [discovery_outcome(r.to_dict()) for r in current_history]
    header = {"seed": seed, "initial": len(history), "cold": "usable" not in observed,
              "initial_outcomes": dict(Counter(observed))}
    if not all(map(supports_timing, history)):
        return {**header, "state": "unsupported_initial_feedback"}
    kept = [r for r in panel if passes_structural_screen(r) and supports_timing(r)]
    if not kept:
        return {**header, "state": "empty_panel"}
    labels = [discovery_outcome(with_current_limit(r, DERATED_CURRENT).to_dict()) for r in kept]
    if set(labels)-{"abort", "violation", "usable"}:
        raise ValueError("Panel metric requires settled evaluation outcomes")
    # Validate the bound against independently held-out exact peaks without passing
    # any of these labels or traces into either model fit.
    current_observations(problem, kept)
    exact, prefix = current_observations(problem, history)
    candidates = [r.candidate for r in kept]
    features = PrefixFeatures(problem, BASE, history, seed, latent=False)
    x = features.transform([r.candidate for r in history])
    try:
        timing, query, _ = fit_completion_heads(problem, history, candidates, seed, methods=(TIMING,),
                                                 features=features)
        p_complete = forecast(timing[TIMING], query)[-1]
        stage = fit_stage_head(FEASIBILITY, history, x, observed, seed, len(problem.space.continuous))
        p_sign = forecast(stage, query, probit=True)[-1]
        values = [np.ones_like(p_complete), p_sign]
        exact_head = None
        for name, observations in zip(METHODS[2:], (exact, prefix)):
            head = StableGroupedHead(x, observations, seed=seed, name=name, reference_scale=.25,
                                    continuous_count=len(problem.space.continuous))
            values.append(forecast(head, query)[-1])
            if name == "completed_exact":
                exact_head = head
    except ModelNumericalFailure as error:
        return {**header, "state": "model_failure", "head": error.head}
    completed_labels = [None if y == "abort" else y == "usable" for y in labels]
    scores = {name: prediction_metrics(kept, candidates, labels, completed_labels, p_complete, q)
              for name, q in zip(METHODS, values)}
    transfer_failures = []
    if transfer_audit:
        for name, rho in STATE_TRANSFER.items():
            try:
                q = transferred_current_probability(exact_head, x, query, prefix, seed, rho)
                if not np.isfinite(q).all():
                    raise FloatingPointError("Nonfinite transfer forecast")
                scores[name] = prediction_metrics(kept, candidates, labels, completed_labels, p_complete, q)
            except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
                transfer_failures.append(name)
    return {
        **header, "state": "ranked", "panel_outcomes": dict(Counter(labels)),
        "exact_initial": sum(o.kind == "exact" for o in exact),
        "prefix_bounds": sum(o.kind == "interval" for o in prefix),
        "prefix_proves_current_violation": sum(o.kind == "interval" and o.lower > 0. for o in prefix),
        "arms": scores,
        **({"transfer_model_failures": transfer_failures} if transfer_audit else {}),
    }


def compare(pairs):
    return {
        "deltas": {
            metric: effect([float(a[metric])-float(b[metric]) for a, b in pairs
                            if a[metric] is not None and b[metric] is not None])
            for metric in ("usable_brier", "conditional_brier", "top_one_usable")},
        "changed_top_one": sum(a["best_id"] != b["best_id"] for a, b in pairs),
        "wins": sum(a["top_one_usable"] and not b["top_one_usable"] for a, b in pairs),
        "losses": sum(b["top_one_usable"] and not a["top_one_usable"] for a, b in pairs),
    }


def summarize(rows):
    ranked = [r for r in rows if r["state"] == "ranked"]
    comparisons = {control: compare([(r["arms"]["exact_plus_prefix"], r["arms"][control])
                                     for r in ranked]) for control in METHODS[:3]}
    panel = Counter()
    for row in ranked:
        panel.update(row["panel_outcomes"])
    return {"states": dict(Counter(r["state"] for r in rows)), "panel_outcomes": dict(panel),
            "prefix_bound_count": sum(r["prefix_bounds"] for r in ranked),
            "prefix_proven_violations": sum(r["prefix_proves_current_violation"] for r in ranked),
            "mean_usable_brier": {name: mean(r["arms"][name]["usable_brier"] for r in ranked)
                                  for name in METHODS} if ranked else {},
            "top_one_hits": {name: sum(r["arms"][name]["top_one_usable"] for r in ranked) for name in METHODS},
            "exact_plus_prefix_vs": comparisons}


def summarize_transfer(rows):
    ranked = [r for r in rows if r["state"] == "ranked"]
    paired = [r["arms"] for r in ranked if all(name in r["arms"] for name in STATE_TRANSFER)]
    return {
        "ranked": len(ranked),
        "transfer_paired": len(paired),
        "model_failures": dict(Counter(name for r in ranked for name in r["transfer_model_failures"])),
        "comparisons": {
            f"{active}_vs_{control}": compare([(arms[active], arms[control]) for arms in paired])
            for active, control in (("prefix_fixed_half", "prefix_fixed_full"),
                                    ("prefix_fixed_half", "completed_exact"),
                                    ("prefix_fixed_full", "completed_exact"),
                                    ("prefix_fixed_full", "exact_plus_prefix"),
                                    ("prefix_fixed_half", "interval_completion"))
        },
    }


def load(profile, root):
    name = PROFILES[profile]
    folder = root/name
    if profile.startswith("pr122"):
        return PR122_RESULT, load_pr122_profile(folder)
    analysis = root/"results/aircraft-timing-decision-value-9a8d841/analyses"/(name+".json")
    spec = plan(name)
    problem = BenchmarkAircraft(spec)
    header = read(folder/"plan.json")
    if (header != {"source_sha": INPUT_SOURCE, "spec": spec, "contract": problem.contract_id}
            or (file_hash(folder/"seal.json"), file_hash(analysis)) != INPUT_HASHES[name]
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("PR135 input identity mismatch")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) != read(analysis)["actual_calls"]:
        raise ValueError("PR135 call count differs")
    return INPUT_RESULT, (spec, problem, calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--initial", type=int, choices=(8, 16), default=8)
    parser.add_argument("--transfer-audit", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if any(p in (output, *output.parents)
           for p in (Path(__file__).resolve().parent, args.input.resolve())):
        parser.error("Write outside the source worktree and sealed inputs")
    sha = source_commit(args.expected_sha, clean=True)
    result, (spec, problem, calls) = load(args.profile, args.input)
    rows = [evaluate_prefix(problem, seed, history, panel, transfer_audit=args.transfer_audit)
            for seed, history, panel in archived_seed_panels(spec, calls, args.initial)]
    cold = [r for r in rows if r["cold"]]
    report = {"status": "VERIFIED_CURRENT_PREFIX_AUDIT", "source_sha": sha, "input_result_sha": result,
              "profile": args.profile, "initial": args.initial, "new_evaluator_calls": 0, "rcaide_mda_calls": 0,
              "summary": {"all": summarize(rows), "cold": summarize(cold),
                          "cold_no_exact_current": summarize([r for r in cold if r.get("exact_initial") == 0])},
              "rows": rows}
    if args.transfer_audit:
        report["state_transfer"] = {
            "fixed_correlations": STATE_TRANSFER,
            "cold": summarize_transfer(cold),
            "cold_no_exact_current": summarize_transfer([r for r in cold if r.get("exact_initial") == 0]),
        }
    save(output, report)
    print(report["status"], args.profile, args.initial, len(rows))


if __name__ == "__main__":
    main()
