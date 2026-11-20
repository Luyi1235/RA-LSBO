"""Audit whether a predicted benefit from a query pays for pursuing it."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from aircraft_risk_acquisition import sign_covariance
from aircraft_risk_discovery import fit_stage_head
from aircraft_timing_information import supports_timing
from aircraft_timing_interval_value import deficit_edges
from aircraft_timing_usable_value import FEASIBILITY, correlation, forecast, timing_joint_cells, two_call_usable
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import save
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from ra_lsbo.partial_bo.models import PrefixFeatures
from run_aircraft_coverage_factorial import FixedTimingHead
from run_aircraft_current_prefix_audit import PROFILES, load
from run_aircraft_initial_feedback_diagnostic import archived_seed_panels
from run_aircraft_risk_cold_start import discovery_outcome, select, source_commit
from run_aircraft_timing_interval_value import effect
from run_aircraft_timing_usable_value import DERATED_CURRENT, passes_screen, with_current_limit


ARMS = ("interval_greedy", "usable_greedy", "decision_gate")
EPSILON = 1e-12  # Numerical tie tolerance, not an estimated intervention threshold.


def split_panel(problem, panel):
    """Freeze probe and continuation pools with no use of evaluation labels."""
    qualified = [r for r in panel if passes_screen(problem, r.candidate)]
    return qualified[::2], qualified[1::2]


def fit(problem, seed, history, candidates):
    features = PrefixFeatures(problem, BASE, history, seed, latent=False)
    x = features.transform([r.candidate for r in history])
    q = features.transform(candidates)
    timing = FixedTimingHead(problem, x, history, seed)
    outcomes = [discovery_outcome(r.to_dict()) for r in history]
    feasible = fit_stage_head(FEASIBILITY, history, x, outcomes, seed, len(problem.space.continuous))
    return timing, feasible, q


def continuation_values(problem, timing, feasible, features, pair_features, probes):
    query = np.vstack((features, pair_features))
    mean, sd, z, c = forecast(timing, query)
    _, sf, zf, f = forecast(feasible, query, probit=True)
    n = len(probes)
    completion_rho = correlation(timing, query, n, sd)
    feasibility_rho = correlation(feasible, query, n, sf)
    joint_f = f[:n, None]*f[None, n:] + sign_covariance(zf[:n], zf[n:], feasibility_rho)
    joint_f = np.clip(joint_f, np.maximum(0., f[:n, None]+f[None, n:]-1.),
                      np.minimum(f[:n, None], f[None, n:]))
    values = []
    for i, probe in enumerate(probes):
        cells = timing_joint_cells(mean[i], sd[i], z[n:], completion_rho[i],
                                   deficit_edges(probe, problem.settings))
        adaptive, fixed = two_call_usable(c[i], f[i], f[n:], joint_f[i], cells)
        values.append({"p_complete": float(c[i]), "p_usable": float(c[i]*f[i]),
                       "best_fixed": float(fixed), "adaptive": float(adaptive),
                       "decision_information": float(max(adaptive-fixed, 0.))})
    return values


def gate(probes, values, greedy):
    base = values[greedy]
    more_informative = [i for i, row in enumerate(values)
                        if row["decision_information"] > base["decision_information"]+EPSILON]
    net = [values[i]["adaptive"]-base["adaptive"] for i in more_informative]
    admissible = [i for i, gain in zip(more_informative, net) if gain > EPSILON]
    chosen = greedy
    if admissible:
        chosen = admissible[select([probes[i] for i in admissible],
                                    [values[i]["adaptive"] for i in admissible])]
    reason = ("activated" if admissible else
              "gain_cannot_offset_cost" if more_informative else "no_additional_decision_information")
    return chosen, {"reason": reason, "more_informative_count": len(more_informative),
                    "admissible_count": len(admissible), "best_net_among_more_informative": max(net, default=None)}


def observed_action(problem, seed, history, first, targets, before):
    outcome = discovery_outcome(first.to_dict())
    receipt = {"first_id": digest(first.candidate), "first_outcome": outcome, "after": before,
               "update": "first_usable"}
    if first.usable_feasible:
        return receipt
    if not supports_timing(first):
        return {**receipt, "update": "unsupported_feedback"}
    try:
        timing, feasible, query = fit(problem, seed, history+[first], targets)
        complete = forecast(timing, query)[-1]
        constraint = forecast(feasible, query, probit=True)[-1]
        return {**receipt, "update": "refit",
                "after": {"interval": complete.tolist(), "usable": (complete*constraint).tolist()}}
    except (ModelNumericalFailure, np.linalg.LinAlgError, FloatingPointError, RuntimeError):
        return {**receipt, "update": "model_failure"}


def settle(first, pair, targets, receipt, objective, before):
    if first.usable_feasible:
        return {"hit": True, "second_usable": None, "selected": None, "rank_correction": None}
    index = select(targets, receipt["after"][objective])
    old = select(targets, before[objective])
    second = pair[index].usable_feasible
    return {"hit": second, "second_usable": second, "selected": digest(targets[index]),
            "rank_correction": int(second)-int(pair[old].usable_feasible)}


def audit_history(problem, seed, history, panel):
    header = {"seed": seed}
    if any(r.usable_feasible for r in history):
        return {**header, "state": "initially_solved"}
    if not all(map(supports_timing, history)):
        return {**header, "state": "unsupported_initial_feedback"}
    probes, followups = split_panel(problem, panel)
    if len(probes) < 2 or len(followups) < 2:
        return {**header, "state": "insufficient_panel"}
    probe_x = [r.candidate for r in probes]
    follow_x = [r.candidate for r in followups]
    try:
        timing, feasible, features = fit(problem, seed, history, probe_x+follow_x)
        complete = forecast(timing, features)[-1]
        constraint = forecast(feasible, features, probit=True)[-1]
        usable = complete*constraint
        n = len(probe_x)
        # Fix A and B before either policy sees any feedback from a probe.
        ordered = sorted(range(n, n+len(follow_x)),
                         key=lambda i: (usable[i], digest(follow_x[i-n])), reverse=True)
        indices = ordered[:2]
        targets = [follow_x[i-n] for i in indices]
        before = {"interval": complete[indices].tolist(), "usable": usable[indices].tolist()}
        values = continuation_values(problem, timing, feasible, features[:n], features[indices], probe_x)
        greedy = select(probe_x, usable[:n])
        active, decision = gate(probe_x, values, greedy)
        picked = {"interval_greedy": select(probe_x, complete[:n]),
                  "usable_greedy": greedy, "decision_gate": active}
    except (ModelNumericalFailure, np.linalg.LinAlgError, FloatingPointError, RuntimeError):
        return {**header, "state": "forecast_failure"}
    pair = [followups[i-n] for i in indices]
    choices = {}
    cache = {}
    for name, i in picked.items():
        if i not in cache:
            cache[i] = observed_action(problem, seed, history, probes[i], targets, before)
        obj = "interval" if name == "interval_greedy" else "usable"
        choices[name] = {**values[i], **cache[i], **settle(probes[i], pair, targets, cache[i], obj, before)}
    return {**header, "state": "paired", "probe_count": n, "followup_candidates": len(followups),
            "pair": [digest(x) for x in targets], "pair_usable": [r.usable_feasible for r in pair],
            "before": before, "activated": active != greedy, "gate": decision, "arms": choices}


def contrast(rows, active, control):
    diffs = [int(r["arms"][active]["hit"])-int(r["arms"][control]["hit"]) for r in rows]
    wins, losses = diffs.count(1), diffs.count(-1)
    return {**effect(diffs), "wins": wins, "losses": losses, "ties": diffs.count(0),
            "sign_test_two_sided": float(binomtest(wins, wins+losses).pvalue) if wins+losses else None}


def summarize(rows):
    paired = [r for r in rows if r["state"] == "paired"]
    triggered = [r for r in paired if r["activated"]]
    result = {"states": dict(Counter(r["state"] for r in rows)), "activated": len(triggered),
              "gate_reasons": dict(Counter(r["gate"]["reason"] for r in paired)),
              "informative_pair_truths": sum(sum(r["pair_usable"]) == 1 for r in paired)}
    for name, cohort in (("all_paired", paired), ("triggered_before_query", triggered)):
        result[name] = {
            "vs_usable_greedy": contrast(cohort, "decision_gate", "usable_greedy"),
            "vs_interval_greedy": contrast(cohort, "decision_gate", "interval_greedy"),
            "predicted_net_vs_usable_greedy": effect([
                r["arms"]["decision_gate"]["adaptive"]-r["arms"]["usable_greedy"]["adaptive"]
                for r in cohort]),
            "arms": {},
        }
        for arm in ARMS:
            receipt = [r["arms"][arm] for r in cohort]
            correction = [r["rank_correction"] for r in receipt if r["rank_correction"] is not None]
            result[name]["arms"][arm] = {
                "hits": sum(r["hit"] for r in receipt),
                "immediate_hits": sum(r["first_outcome"] == "usable" for r in receipt),
                "corrective_swaps": correction.count(1), "harmful_swaps": correction.count(-1),
                "followup_attempts": len(correction),
                "updates": dict(Counter(r["update"] for r in receipt)),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--initial", choices=(8, 16), default=8, type=int)
    parser.add_argument("--current-limit", choices=("nominal", "derated"), default="nominal")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if any(p in (output, *output.parents)
           for p in (Path(__file__).resolve().parent, args.input.resolve())):
        parser.error("Write outside the source worktree and sealed inputs")
    sha = source_commit(args.expected_sha, clean=True)
    source, (spec, problem, calls) = load(args.profile, args.input)
    fraction = DERATED_CURRENT if args.current_limit == "derated" else 1.
    rows = [audit_history(problem, seed, [with_current_limit(r, fraction) for r in history],
                          [with_current_limit(r, fraction) for r in panel])
            for seed, history, panel in archived_seed_panels(spec, calls, args.initial)]
    report = {"status": "VERIFIED_DECISION_VALUE_AUDIT", "source_sha": sha, "input_result_sha": source,
              "profile": args.profile, "initial": args.initial, "cell_current_limit_fraction": fraction,
              "new_evaluator_calls": 0, "rcaide_mda_calls": 0, "summary": summarize(rows), "rows": rows}
    save(output, report)
    print(report["status"], args.profile, args.initial, report["summary"]["states"], flush=True)


if __name__ == "__main__":
    main()
