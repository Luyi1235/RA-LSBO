"""Small read-only check of sharing-model value on an existing aircraft archive."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import ndtr, softmax

from aircraft_risk_prior import SharedRiskEvidenceGP, layout_features
from aircraft_risk_transfer_value import comparison_value, feedback_probabilities, interval_probability
from aircraft_timing_information import supports_timing
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.partial_refinement.posterior import group_divisors
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from run_aircraft_initial_budget import coverage_index
from run_aircraft_risk_cold_start import file_hash, read, select, source_commit
from run_aircraft_risk_timing_replay import MARGIN_SCALE, timing_feedback
from run_aircraft_timing_decision_audit import INPUT_HASHES, INPUT_RESULT, INPUT_SOURCE
from run_aircraft_timing_interval_value import (
    BenchmarkAircraft, evidence_hashes, plan, read_ledger, save, structural_margins,
)


SHARING = ("global", "layout")
LENGTHS = (.4, 1.5)
INITIAL = 8
MAX_SEEDS = 16
ARMS = ("transfer_value", "greedy", "coverage")


def observe(problem, record):
    return timing_feedback(record.to_dict(), problem.settings, use_interval=True,
                           allow_hidden_reserve_stop=True)


def model_predictions(problem, history, candidates, seed):
    features = PrefixFeatures(problem, BASE, history, seed, latent=False)
    past = [record.candidate for record in history]
    x = layout_features(features.transform(past), past)
    query = layout_features(features.transform(candidates), candidates)
    divisor = group_divisors(x.shape[1], len(problem.space.continuous), *LENGTHS)
    observations = [observe(problem, record) for record in history]
    scaled = [replace(o, lower=None if o.lower is None else o.lower/MARGIN_SCALE,
                      upper=None if o.upper is None else o.upper/MARGIN_SCALE)
              for o in observations]
    settings = MixedGPSettings(length_scale=1., exact_noise=1e-3, interval_noise=0.,
                               jitter=1e-8, stable_tail_moments=True, record_ep_trace=False)
    models = [SharedRiskEvidenceGP(x/divisor, scaled, seed=seed, settings=settings, sharing=mode)
              for mode in SHARING]
    forecast = [model.predict(query/divisor) for model in models]
    means = MARGIN_SCALE*np.array([p["mean"] for p in forecast])
    sd = MARGIN_SCALE*np.sqrt(np.maximum(np.array([p["variance"] for p in forecast]), 1e-14))
    evidence = np.array([model.log_evidence() for model in models])
    if not (np.isfinite(means).all() and np.isfinite(sd).all() and np.isfinite(evidence).all()):
        raise FloatingPointError("Nonfinite fixed-sharing GP forecast")
    return means, sd, softmax(evidence)


def screen(problem, records):
    return [r for r in records if all(v <= 0. for v in structural_margins(problem, r.candidate).values())]


def audit_seed(problem, seed, history, panel):
    row = {"seed": seed, "state": "initially_solved"}
    if any(record.usable_feasible for record in history):
        return row
    if not all(map(supports_timing, history)):
        return {**row, "state": "unsupported_initial_feedback"}
    # Archive roster order fixes the two partitions before inspecting feedback.
    probe, decision = screen(problem, panel[::2]), screen(problem, panel[1::2])
    layouts = ("distributed", "wingtip")
    branch_indices = [[i for i, r in enumerate(decision)
                       if r.candidate["discrete"]["prop_layout"] == layout] for layout in layouts]
    if not probe or any(not group for group in branch_indices):
        return {**row, "state": "insufficient_panel_support"}
    candidates = [r.candidate for r in probe+decision]
    means, sd, weights = model_predictions(problem, history, candidates, seed)
    n = len(probe)
    p = ndtr(-means/sd)
    mixture = weights @ p
    actions = []
    for group in branch_indices:
        indices = [i+n for i in group]
        j = select([candidates[i] for i in indices], mixture[indices])
        actions.append(indices[j])
    fixed_actions = [candidates[i] for i in actions]
    pair = p[:, actions]
    disagreement = bool(np.min(pair[:, 0]-pair[:, 1]) < 0. < np.max(pair[:, 0]-pair[:, 1]))
    cells = feedback_probabilities(means[:, :n], sd[:, :n])
    score = comparison_value(weights, pair, cells)
    signs = np.stack((cells[..., 0], cells[..., 1:].sum(axis=-1)), axis=-1)
    sign_score = comparison_value(weights, pair, signs)
    observed = {}
    greedy = select(candidates[:n], mixture[:n])
    active = select(candidates[:n], score) if score.max() > 1e-12 else greedy
    covered = coverage_index(problem, candidates[:n], [r.candidate for r in history])
    choices = {}
    for arm, index in zip(ARMS, (active, greedy, covered)):
        if index not in observed:
            record = probe[index]
            post = weights @ pair
            posterior_weights = weights
            updated = supports_timing(record)
            if updated:
                obs = observe(problem, record)
                likelihood = interval_probability(
                    means[:, index], sd[:, index], -np.inf if obs.lower is None else obs.lower,
                    np.inf if obs.upper is None else obs.upper)
                posterior_weights = weights*likelihood
                if not posterior_weights.sum() > 0.:
                    raise FloatingPointError("Realized feedback has no predictive mass")
                posterior_weights /= posterior_weights.sum()
                mu, sigma, _ = model_predictions(problem, history+[record], fixed_actions, seed)
                post = posterior_weights @ ndtr(-mu/sigma)
            pick = select(fixed_actions, post)
            observed[index] = {"probe_id": digest(candidates[index]), "probe_usable": record.usable_feasible,
                               "updated": updated, "post_weights": posterior_weights.tolist(),
                               "recommendation": pick, "recommendation_usable": decision[actions[pick]-n].usable_feasible}
        choices[arm] = {**observed[index], "transfer_value": float(score[index]),
                        "sign_value": float(sign_score[index])}
    return {"seed": seed, "state": "paired", "disagreement": disagreement, "weights": weights.tolist(),
            "action_completion": pair.tolist(),
            "action_ids": [digest(x) for x in fixed_actions],
            "action_usable": [decision[i-n].usable_feasible for i in actions],
            "before_choice": select(fixed_actions, weights @ pair), "choices": choices}


def audit_profile(folder):
    header = read(folder/"plan.json")
    spec = header["spec"]
    name = spec["name"]
    seal = folder/"seal.json"
    if (header["source_sha"] != INPUT_SOURCE or spec != plan(name)
            or file_hash(seal) != INPUT_HASHES[name][0] or read(seal) != evidence_hashes(folder)):
        raise ValueError("Expected the original sealed PR135 evidence")
    problem = BenchmarkAircraft(spec)
    if header["contract"] != problem.contract_id:
        raise ValueError("Evaluator contract differs")
    calls = read_ledger(folder, problem.contract_id)
    rows = []
    for seed in spec["seeds"][:MAX_SEEDS]:
        history = [calls[f"{seed}/initial/{i}"][1] for i in range(INITIAL)]
        panel = [calls[f"{seed}/heldout/{i}"][1] for i in range(spec["heldout"])]
        rows.append(audit_seed(problem, seed, history, panel))
    paired = [r for r in rows if r["state"] == "paired"]
    a, g = "transfer_value", "greedy"
    return {"name": name, "seeds": len(rows), "paired": len(paired),
            "sharing_rank_disagreements": sum(r["disagreement"] for r in paired),
            "positive_value": sum(r["choices"][a]["transfer_value"] > 1e-12 for r in paired),
            "changed_probe_vs_greedy": sum(r["choices"][a]["probe_id"] != r["choices"][g]["probe_id"]
                                           for r in paired), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    sha = source_commit(args.expected_sha, clean=True)
    output = args.output.resolve()
    if any(folder.resolve() in (output, *output.parents) for folder in args.input):
        parser.error("Write outside the sealed inputs")
    result = {"status": "VERIFIED_RISK_TRANSFER_PILOT", "source_sha": sha,
              "input_result_sha": INPUT_RESULT, "new_evaluator_calls": 0, "profiles": []}
    for folder in args.input:
        row = audit_profile(folder)
        result["profiles"].append(row)
        print(result["status"], row["name"], row["paired"], row["sharing_rank_disagreements"], flush=True)
    save(output, result)


if __name__ == "__main__":
    main()
