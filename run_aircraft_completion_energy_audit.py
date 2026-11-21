"""Audit whether completion rankings retain low-energy candidates on archived rosters."""

import argparse
from math import ceil
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.special import ndtr
import torch

from aircraft_timing_information import PROBABILITY_FLOOR, score_information, supports_timing
from aircraft_timing_interval_value import deficit_edges
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import save
from ra_lsbo.features import load_base
from ra_lsbo.structure_corpus import native_observations
from run_aircraft_coverage_factorial import FixedTimingHead
from run_aircraft_external_full_budget import POOL_ARMS, SCENARIOS, read, read_ledger
from run_aircraft_native_risk_lsbo import ENERGY_TARGET_WH
from run_aircraft_risk_cold_start import file_hash, select, source_commit
from run_aircraft_roster_energy_audit import (
    PR149_RESULT, UsableAircraft, audit_plan, original_data, usable_energy, verify,
)


PR150_RESULT = "44140fb6a8e3f03101dda564264222ce86698418"
PR150_CODE = "df181f3b53bdf7af66b7b194f71353bf5a840c1f"
PR150_MANIFEST = "daf7db68dcdafce5552c2fb388a24dd9c6746c321f5ccfda9dc1523264ba153e"
ARM_NAMES = ("sign_greedy", "interval_greedy", "interval_active")
N_INITIAL = 8
PRETRAIN_EPOCHS = 60
LATENT_DIM = 4


def archived_rows(root, scenario, seed, extra, contract):
    folder = root/scenario/str(seed)
    ledger = read_ledger(folder, contract)
    prefix = [ledger[f"initial/{i}"][1] for i in range(1, N_INITIAL+1)]
    measured = {record.observation.design_id: record for tag, (_, record, role) in ledger.items()
                if role == "online" and tag.split("/")[1] in POOL_ARMS}
    measured.update({tag.split("/")[1]: value[1] for tag, value in extra.items()
                     if tag.startswith(f"{seed}/")})
    candidates = read(folder/"roster.json")["covered"]
    return prefix, [measured[digest(point)] for point in candidates]


def rank_choices(problem, prefix, panel, seed, modules):
    candidates = [r.candidate for r in panel]
    trusted = [r for r in prefix if supports_timing(r)]
    if not trusted:
        return None
    rep = modules["representation"].Representation(problem.space, seed, LATENT_DIM)
    rep.fit(native_observations(prefix, modules, problem.space), PRETRAIN_EPOCHS)
    inputs = rep.features([problem.candidate_class(**r.candidate) for r in trusted]).numpy()
    query = rep.features([problem.candidate_class(**point) for point in candidates]).numpy()
    sign = FixedTimingHead(problem, inputs, trusted, seed, use_interval=False).predict(query)
    sign_p = ndtr(-sign["mean"]/np.sqrt(np.maximum(sign["variance"], 1e-14)))
    interval = FixedTimingHead(problem, inputs, trusted, seed, use_interval=True)
    edges = [deficit_edges(point, problem.settings) for point in candidates]
    interval_p, greedy, eligible, gains = score_information(interval, query, candidates, edges)
    active = eligible[select([candidates[i] for i in eligible], gains)]
    return sign_p, interval_p, (select(candidates, sign_p), greedy, int(active))


def diagnose(problem, prefix, panel, seed, modules, target):
    row = {"seed": seed, "cold": not any(record.usable_feasible for record in prefix)}
    if not row["cold"]:
        return row
    scores = rank_choices(problem, prefix, panel, seed, modules)
    if scores is None:
        return {**row, "scored": False}
    sign_p, interval_p, chosen = scores
    energies = [usable_energy(record) for record in panel]
    feasible = sorted((e, i) for i, e in enumerate(energies) if e is not None)
    low = {i for _, i in feasible[:ceil(len(feasible)/4)]}
    target_indices = {i for e, i in feasible if e <= target}
    labels = np.asarray([record.diagnostics.get("mission_completed") is True for record in panel],
                        dtype=float)
    rankings = {}
    for name, p in (("sign", sign_p), ("interval", interval_p)):
        shortlisted = {i for i, score in enumerate(p) if score >= PROBABILITY_FLOOR*max(p)}
        rankings[name] = {
            "completion_brier": float(np.mean((p-labels)**2)), "shortlist": len(shortlisted),
            "low_energy_retained": len(shortlisted & low),
            "absolute_target_retained": len(shortlisted & target_indices),
        }
    return {**row, "scored": True, "candidates": len(panel), "feasible_candidates": len(feasible),
            "low_energy_candidates": len(low), "absolute_target_candidates": len(target_indices),
            "rankings": rankings,
            "choices": {name: {"candidate_id": panel[i].observation.design_id, "usable_Wh": energies[i],
                               "low_energy": i in low, "absolute_target": i in target_indices}
                        for name, i in zip(ARM_NAMES, chosen)}}


def summary(rows):
    cold = [r for r in rows if r["cold"]]
    scored = [r for r in cold if r["scored"]]
    with_target = [r for r in scored if r["absolute_target_candidates"]]
    return {"cold_prefixes": len(cold), "scored_prefixes": len(scored),
            "absolute_target_present": len(with_target),
            "ranking": {name: {
                "mean_completion_brier": mean(r["rankings"][name]["completion_brier"] for r in scored),
                "shortlist_retains_low_energy": sum(r["rankings"][name]["low_energy_retained"] > 0 for r in scored),
                "shortlist_retains_absolute_target": sum(
                    r["rankings"][name]["absolute_target_retained"] > 0 for r in with_target),
            } for name in ("sign", "interval")},
            "selection": {name: {
                "usable": sum(r["choices"][name]["usable_Wh"] is not None for r in scored),
                "low_energy": sum(r["choices"][name]["low_energy"] for r in scored),
                "absolute_target": sum(r["choices"][name]["absolute_target"] for r in scored),
            } for name in ARM_NAMES}}


def audit(root, supplemental):
    if file_hash(supplemental/"SHA256SUMS.txt") != PR150_MANIFEST:
        raise ValueError("Expected the fixed PR150 supplemental archive")
    torch.set_num_threads(1)
    modules, _ = load_base(BASE)
    sections = {}
    for scenario in SCENARIOS:
        spec, cases = original_data(root, scenario)
        problem = UsableAircraft(spec)
        run = supplemental/scenario
        registration = audit_plan(spec, cases, PR150_CODE)
        if verify(spec, cases, registration, run) != read(supplemental/f"{scenario}-analysis.json"):
            raise ValueError("PR150 report differs from the sealed supplemental ledger")
        extra = read_ledger(run, problem.contract_id)
        target = ENERGY_TARGET_WH[scenario]
        rows = []
        for seed in spec["seeds"]:
            prefix, panel = archived_rows(root, scenario, seed, extra, problem.contract_id)
            rows.append(diagnose(problem, prefix, panel, seed, modules, target))
        sections[scenario] = {"target_Wh": target, "summary": summary(rows), "rows": rows}
    return {"status": "VERIFIED_COMPLETION_ENERGY_ALIGNMENT", "archive_pr149": PR149_RESULT,
            "archive_pr150": PR150_RESULT, "mode": "retrospective shared roster, single query per policy",
            "new_evaluator_calls": 0, "rcaide_mda_calls": 0, "scenarios": sections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr149-archive", type=Path, required=True)
    parser.add_argument("--pr150-archive", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sha = source_commit(args.expected_sha, clean=True)
    original, supplemental, output = (path.resolve() for path in
                                      (args.pr149_archive, args.pr150_archive, args.output))
    if output.exists() or any(root in (output, *output.parents) for root in (
            original, supplemental, Path(__file__).resolve().parent)):
        parser.error("Use a new output outside both archives and the code worktree")
    receipt = audit(original, supplemental)
    save(output, {**receipt, "source_sha": sha})
    print(receipt["status"], receipt["new_evaluator_calls"], flush=True)


if __name__ == "__main__":
    main()
