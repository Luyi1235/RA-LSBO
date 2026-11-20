"""Separate cold-start budget allocation from timing-feedback information."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.special import ndtr

from aircraft_timing_information import supports_timing
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.partial_bo.models import EvidenceGP, PrefixFeatures
from ra_lsbo.partial_refinement.posterior import group_divisors
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from run_aircraft_initial_budget import brier_by_configuration, paired_interval
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, read, select, source_commit
from run_aircraft_risk_timing_replay import MARGIN_SCALE, timing_feedback
from run_aircraft_timing_decision_audit import INPUT_HASHES, INPUT_RESULT, INPUT_SOURCE
from run_aircraft_timing_interval_value import (
    BenchmarkAircraft, evidence_hashes, plan, read_ledger, save, structural_margins,
)


BUDGET_RESULT = "2e21ba084026552cedb586662e059507190c9816"
BUDGET_HASHES = {
    "timing-budget-headwind.json": "1b8251398847995c70d717b2d6fb4f5ce58eb883d273a619f250913e19e2eabf",
    "timing-budget-payload.json": "8f4484a82f10268467cf72cde5cfbd70c2238632a9517f0cc619b9c1a81ab456",
}
LENGTHS = (.4, 1.5)
SIZES = (8, 16)
STRATA = ("all", "seen_configuration", "unseen_configuration")


def effect(values):
    return {"n": len(values), "mean": mean(values) if values else None,
            "paired_95": paired_interval(values)}


def budget_contrasts(path):
    if file_hash(path) != BUDGET_HASHES[path.name]:
        raise ValueError("Expected the fixed PR131 analysis")
    rows = defaultdict(dict)
    for record in read(path)["per_seed"]:
        rows[record["seed"]][record["arm"]] = record["capped_first"]
    contrasts = defaultdict(list)
    for row in rows.values():
        i8, i16, s8, s16 = (row[arm] for arm in (
            "timing_interval-8", "timing_interval-16", "timing_sign-8", "timing_sign-16"))
        contrasts["interval8_minus_interval16"].append(i8-i16)
        contrasts["sign8_minus_sign16"].append(s8-s16)
        contrasts["interval_minus_sign_at8"].append(i8-s8)
        contrasts["interval_minus_sign_at16"].append(i16-s16)
        contrasts["feedback_by_initial_interaction"].append((i8-s8)-(i16-s16))
    return {"name": path.name, "contrasts": {k: effect(v) for k, v in contrasts.items()}}


def completion_forecasts(problem, history, candidates, seed):
    features = PrefixFeatures(problem, BASE, history, seed, latent=False)
    x = features.transform([record.candidate for record in history])
    query = features.transform(candidates)
    divisor = group_divisors(x.shape[1], len(problem.space.continuous), *LENGTHS)
    settings = MixedGPSettings(length_scale=1., exact_noise=1e-3, interval_noise=0.,
                               jitter=1e-8, stable_tail_moments=True, record_ep_trace=False)
    result = {}
    for method in ("sign", "interval"):
        observations = [timing_feedback(record.to_dict(), problem.settings,
                                        use_interval=method == "interval",
                                        allow_hidden_reserve_stop=True) for record in history]
        scaled = [replace(o, lower=None if o.lower is None else o.lower/MARGIN_SCALE,
                          upper=None if o.upper is None else o.upper/MARGIN_SCALE)
                  for o in observations]
        model = EvidenceGP(x/divisor, scaled, seed=seed, settings=settings)
        p = model.predict(query/divisor)
        result[method] = ndtr(-p["mean"]/np.sqrt(np.maximum(p["variance"], 1e-14)))
    return result


def score_subpanel(panel, outcomes, selected, forecasts):
    records = [panel[i] for i in selected]
    candidates = [r.candidate for r in records]
    labels = [None if outcomes[i] == "unknown" else outcomes[i] != "abort" for i in selected]
    results = {}
    for name, all_scores in forecasts.items():
        scores = all_scores[selected]
        best = select(candidates, scores)
        results[name] = {"best_id": digest(candidates[best]), "best_usable": records[best].usable_feasible,
                         "completion_brier": brier_by_configuration(scores, records, labels)}
    return {"candidates": len(records), "has_usable": any(r.usable_feasible for r in records),
            "methods": results}


def compare_history(problem, seed, history, panel, outcomes):
    cold = not any(r.usable_feasible for r in history)
    seen = {digest(r.candidate["discrete"]) for r in history}
    groups = {"all": list(range(len(panel))), "seen_configuration": [], "unseen_configuration": []}
    for i, record in enumerate(panel):
        group = "seen_configuration" if digest(record.candidate["discrete"]) in seen else "unseen_configuration"
        groups[group].append(i)
    row = {"seed": seed, "initial": len(history), "cold": cold, "groups": {}}
    if not panel or not all(map(supports_timing, history)):
        return row
    forecasts = completion_forecasts(problem, history, [r.candidate for r in panel], seed)
    row["groups"] = {name: score_subpanel(panel, outcomes, members, forecasts)
                     for name, members in groups.items() if members}
    return row


def summarize(rows, stratum):
    data = [r["groups"][stratum] for r in rows if stratum in r["groups"]]
    pairs = [(r["methods"]["interval"], r["methods"]["sign"]) for r in data]
    hits = [int(a["best_usable"])-int(b["best_usable"]) for a, b in pairs]
    brier = [(a["completion_brier"], b["completion_brier"]) for a, b in pairs]
    return {"n": len(pairs), "with_usable": sum(r["has_usable"] for r in data),
            "changed_first": sum(a["best_id"] != b["best_id"] for a, b in pairs),
            "interval_hits": sum(a["best_usable"] for a, _ in pairs),
            "sign_hits": sum(b["best_usable"] for _, b in pairs),
            "wins": hits.count(1), "losses": hits.count(-1),
            "usable_delta": effect(hits),
            "brier_delta": effect([a-b for a, b in brier if a is not None and b is not None])}


def panel_replay(folder):
    header = read(folder/"plan.json")
    spec = header["spec"]
    name = spec["name"]
    seal = folder/"seal.json"
    if (header["source_sha"] != INPUT_SOURCE or spec != plan(name)
            or file_hash(seal) != INPUT_HASHES[name][0] or read(seal) != evidence_hashes(folder)):
        raise ValueError("PR135 sealed evidence differs")
    problem = BenchmarkAircraft(spec)
    if header["contract"] != problem.contract_id:
        raise ValueError("Evaluator contract differs")
    calls = read_ledger(folder, problem.contract_id)
    rows = []
    outcomes_seen = Counter()
    for seed in spec["seeds"]:
        history = [calls[f"{seed}/initial/{i}"][1] for i in range(max(SIZES))]
        raw = [calls[f"{seed}/heldout/{i}"][1] for i in range(spec["heldout"])]
        panel = [r for r in raw if all(v <= 0. for v in structural_margins(problem, r.candidate).values())]
        if {digest(r.candidate) for r in history} & {digest(r.candidate) for r in panel}:
            raise ValueError("Training and held-out designs intersect")
        outcomes = [discovery_outcome(record.to_dict()) for record in panel]
        outcomes_seen.update(outcomes)
        rows.extend(compare_history(problem, seed, history[:n], panel, outcomes) for n in SIZES)
    summary = {}
    for n in SIZES:
        group = [r for r in rows if r["initial"] == n]
        cold = [r for r in group if r["cold"]]
        summary[str(n)] = {
            key: {stratum: summarize(subset, stratum) for stratum in STRATA}
            for key, subset in (("all", group), ("cold", cold))}
    return {"name": name, "seeds": len(spec["seeds"]), "panel_outcomes": dict(outcomes_seen),
            "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--budget-analyses", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    sha = source_commit(args.expected_sha, clean=True)
    output = args.output.resolve()
    if any(folder.resolve() in (output, *output.parents) for folder in args.input):
        parser.error("Write outside the sealed inputs")
    result = {"status": "VERIFIED_TIMING_FEEDBACK_MECHANISM", "source_sha": sha,
              "budget_result": BUDGET_RESULT, "panel_result": INPUT_RESULT,
              "new_evaluator_calls": 0, "length_scales": LENGTHS,
              "budget": [budget_contrasts(path) for path in args.budget_analyses], "profiles": []}
    for folder in args.input:
        profile = panel_replay(folder)
        result["profiles"].append(profile)
        print(result["status"], profile["name"], len(profile["rows"]), flush=True)
    save(output, result)


if __name__ == "__main__":
    main()
