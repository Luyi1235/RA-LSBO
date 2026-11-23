"""Replay the frozen 45 nmi worker evidence and same-prefix proposals without workers."""

import importlib
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType

import torch

from run_x57_distance_diagnostic import ROOT, read_events
from run_x57_engineering_energy_pilot import ARMS, INITIAL, POLICY, fit_head, points, source_row
from run_x57_initialization_diagnostic import verified_hash
from ra_lsbo.x57_cold_campaign import ObservationAdapter, update_native
from ra_lsbo.x57_cold_feedback import normalize
from ra_lsbo.x57_cold_risk import propose
from ra_lsbo.x57_energy_stop import STATUS, first_crossing, prefix_energy, stopped_metrics
from ra_lsbo.x57_energy_surrogate import energy_training_row
from ra_lsbo.x57_live import X57LiveProblem
from ra_lsbo.x57_native_slot import configure_native_search

ARCHIVE = "bab2b2b0e7fd59ec920ebabd0f04e2584716face"
OLD = "7d106a807559a564d4b4065a935fb8abe8b7e265"
SWITCH = "5830ffde99ca47f56c742d688186f437d248edd6"
SEED = 20261125
LEDGER_HASH = "28d3914d00531a1eee8ee8daa09f3751fc72bd3e6efee3b40e45585ae878c006"
INITIAL_HASH = "728af5c827e82bb764ae0da04a38553d7419b79fdce3d1078240eb83e46928ce"


def archive_rows(archive):
    verified_hash(archive / "calls.jsonl", LEDGER_HASH)
    verified_hash(archive / "shared_initial.json", INITIAL_HASH)
    events = read_events(archive / "calls.jsonl")
    starts = [(r["journal"], r["call_id"], r["candidate_id"])
              for r in events if r["event"] == "worker_started"]
    settled = [r for r in events if r["event"] == "settled"]
    if (len(starts) != 44 or starts != [(r["journal"], r["call_id"], r["candidate_id"]) for r in settled]
            or not all(r["charged"] and r["seed"] == SEED for r in settled)):
        raise ValueError("Archived worker ledger disagrees with the pinned pilot")
    initial = json.loads((archive / "shared_initial.json").read_bytes())
    if ([r["candidate_id"] for r in initial] != [r["candidate_id"] for r in settled[:INITIAL]]
            or [r["worker_result"] for r in initial] != [r["worker_result"] for r in settled[:INITIAL]]):
        raise ValueError("Archived shared initialization does not align with the ledger")
    return initial, settled


def stop_replay(settled):
    stops = []
    for row in settled:
        raw = row["worker_result"]
        metrics = raw["metrics"]
        segments = metrics.get("mission_solver_segments", [])
        if (row["feedback"]["completion"] is not True or metrics.get("energy_audit_status") != "passed"
                or metrics.get("mission_profile", {}).get("cruise_distance_nmi") != 45):
            raise ValueError("Expected 44 complete energy-audited 45 nmi trajectories")
        reconstructed, prefixes = prefix_energy(segments)
        if (len(prefixes) != 11 or not math.isclose(reconstructed, metrics["terminal_energy_Wh"], abs_tol=1e-6)):
            raise ValueError("Converged segment energy disagrees with the archived full energy audit")
        if (stop := first_crossing(segments)) is None:
            continue
        prefix = segments[:stop["completed_segments"]]
        truncated = dict(objective=None, solver_success=False, constraints={}, status=STATUS,
                         metrics=stopped_metrics(prefix, metrics["mission_profile"]))
        feedback = normalize(truncated, row["feedback"]["constraints"], row["precheck"], "rcaide")
        observed = {**row, "feedback": feedback, "worker_result": truncated}
        sign, bound = [energy_training_row(observed, row["journal"] + ":" + row["call_id"],
                                          45, prefix_energy, interval)[0]
                       for interval in (False, True)]
        if (feedback["completion"] is not None or feedback["objective_kind"] != "missing"
                or feedback["known_constraint_violations"] != ["usable_energy"]
                or sign.kind != "interval" or sign.lower != 0 or bound.lower <= 0):
            raise ValueError("Replayed stop leaked full truth or lost its energy constraint")
        stops.append({"candidate_id": row["candidate_id"], "journal": row["journal"], "call_id": row["call_id"],
                      "evaluation": row["evaluation"], "inferred_feedback": feedback,
                      "inferred_worker_result": truncated, "sign_lower": sign.lower,
                      "numeric_lower": bound.lower, "historical_truth": {
                          "objective_Wh": row["feedback"]["objective_Wh"], "completion": True,
                          "terminal_energy_Wh": metrics["terminal_energy_Wh"], "outcome": row["feedback"]["outcome"],
                          "worker_artifact": f"{row['journal']}/workers/{row['call_id']}/response.json"}})
    if len(stops) != 5:
        raise ValueError(f"Expected five pre-final energy stops, got {len(stops)}")
    return {"scanned": len(settled), "new_worker_calls": 0, "triggered": len(stops), "cases": stops}


def load_risk(sha):
    path = "ra_lsbo/x57_cold_risk.py"
    source = subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=ROOT)
    module = ModuleType("ra_lsbo.x57_cold_risk_replay")
    exec(compile(source, f"{sha}:{path}", "exec"), module.__dict__)
    return module


def first_step_cold_trace(module):
    captured = {}
    ordinary_pool = module.latent_pool
    ordinary_constraints = module.fit_cold_constraints

    def pool(native, *args):
        captured["native"] = native
        return ordinary_pool(native, *args)

    def constraints(native, rep, bounds):
        result = ordinary_constraints(native, rep, bounds)
        captured.update(models=result, bounds=bounds, analytical=native.problem.analytical_constraints)
        return result

    module.latent_pool, module.fit_cold_constraints = pool, constraints
    return captured


def cold_parts(rep, point, head, info, captured):
    modeling = captured.get("models")
    if not modeling or head is None or "engineering_log_probability" not in info:
        return None
    names, models = modeling
    offset, scale = captured["bounds"]
    with torch.no_grad():
        features = rep.features([point])
        x = (features - offset) / scale
        probs = {name: 1. for name in captured["analytical"]}
        probs["usable_energy"] = float(head.log_probability(features).exp().item())
        for name in names:
            if name in models:
                model, threshold = models[name]
                post = model.posterior(x)
                probs[name] = float(torch.special.ndtr((threshold - post.mean) /
                                                       post.variance.clamp_min(1e-12).sqrt()).item())
            else:
                probs[name] = .5
    return {"log_expected_improvement": None, "expected_improvement_normalized": None,
            "p_complete": info["p_complete"], "engineering_probabilities": probs,
            "joint_log_score": info["engineering_log_probability"] + math.log(info["p_complete"] or 1.)}


def summarize_branch(trace, winner, space, branch):
    if not trace or "pool" not in trace:
        return {"status": "no_native_pool", "best_raw_rank": None, "locally_optimized": False}
    ids = {p["candidate_id"] for p in trace["pool"] if space.branch_key(p["discrete"]) == branch}
    rank = next((i for i, key in enumerate(trace["raw_rank_order"], 1) if key in ids), None)
    optimized = branch in trace["local_branch_keys"]
    if space.branch_key(winner.discrete) == branch:
        status = "selected"
    elif optimized:
        status = "locally_optimized_but_not_selected"
    elif rank is not None:
        status = "generated_but_not_selected_for_local_search"
    else:
        status = "not_generated"
    return {"status": status, "best_raw_rank": rank, "locally_optimized": optimized,
            "accepted_local_results": sum(p["candidate_id"] in ids and p["from_local_search"]
                                          for p in trace["pool"])}


def proposals(base, config, output, initial, settled):
    problem = X57LiveProblem(base, config, output)
    module = importlib.import_module("lsbo.optimizer")
    by_arm = {arm: [source_row(r, arm) for r in settled if r["arm"] == arm] for arm in ARMS}
    baseline_low = min((r for r in by_arm["lsbo"] if r["feedback"]["usable"]),
                       key=lambda r: r["feedback"]["objective_Wh"])
    branch = problem.space.branch_key(baseline_low["candidate"]["discrete"])
    truth = {}
    for row in settled:
        truth.setdefault(row["candidate_id"], []).append({
            "arm": row["arm"], "evaluation": row["evaluation"], "usable": row["feedback"]["usable"],
            "objective_Wh": row["feedback"]["objective_Wh"], "outcome": row["feedback"]["outcome"]})

    def run(label, risk_func, *, through, reserve, direct, wanted, cold=None):
        directory = output / label
        directory.mkdir()
        arm = "ra_lsbo_energy_sign"
        torch.manual_seed(SEED)
        adapter = ObservationAdapter(problem.problem, module.Evaluation)
        optimizer = module.Optimizer(adapter, module.Settings(seed=SEED, initial=8, budget=20), directory)
        configure_native_search(optimizer, reserve=reserve, joint_final=direct, audit=True)
        history = list(initial)
        for row in history:
            update_native(optimizer, problem.space.canonical(module.Candidate(**row["candidate"])),
                          row, "initial", row["proposal"])
        audit = {}
        for i in range(1, through + 1):
            state = {}

            def fit(rep, bounds):
                head, _ = fit_head(rep, [o.candidate for o in optimizer.observations], history,
                                   SEED, 45, False, bounds)
                state.update(rep=rep, energy=head)
                return head

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED + i)
                candidate, phase, info = risk_func(optimizer, history, POLICY[arm],
                                                   module.mixed_candidate_pool, 512, fit_energy=fit)
            if i in wanted:
                key = optimizer.space.key(candidate)
                trace = info.get("native_trace")
                components = info.get("acquisition_components")
                if components is None and cold is not None and state:
                    components = cold_parts(state["rep"], candidate, state["energy"], info, cold)
                audit[str(i)] = {
                    "candidate_id": key, "candidate": candidate.to_dict(), "phase": phase,
                    "proposal": info, "acquisition": components, "archived_truth": truth.get(key),
                    "low_energy_branch": summarize_branch(trace, candidate, optimizer.space, branch),
                }
            recorded = by_arm[arm][i - 1]
            update_native(optimizer, problem.space.canonical(module.Candidate(**recorded["candidate"])), recorded,
                          recorded["phase"], recorded["proposal"])
            history.append(recorded)
        return audit

    previous = load_risk(OLD)
    switched = load_risk(SWITCH)
    old = run("7d106a8", previous.propose, through=1, reserve=False, direct=False,
              wanted={1}, cold=first_step_cold_trace(previous))
    new = run("5830ffd", switched.propose, through=1, reserve=False, direct=False, wanted={1})
    if (new["1"]["proposal"].get("risk_mode") != "risk_weighted_native_search"
            or not old["1"]["acquisition"] or not new["1"]["acquisition"]
            or old["1"]["candidate_id"] != by_arm["ra_lsbo_energy_sign"][0]["candidate_id"]):
        raise ValueError("The frozen first-step replay failed its acquisition or identity checks")
    steps = set(range(6, 11))
    baseline = run("ordinary_local_slots", propose, through=10, reserve=False, direct=True, wanted=steps)
    reserve = run("reserved_local_slot", propose, through=10, reserve=True, direct=True, wanted=steps)
    changed_slots = 0
    for i in map(str, sorted(steps)):
        left, right = (rows[i]["proposal"].get("native_trace", {}) for rows in (baseline, reserve))
        if (left.get("raw_rank_order") != right.get("raw_rank_order")
                or len(left.get("local_branch_keys", [])) != len(right.get("local_branch_keys", []))):
            raise ValueError(f"Local slot replay changed the raw pool/ranking or search quota at step {i}")
        changed_slots += left.get("local_branch_keys") != right.get("local_branch_keys")
    return {"new_worker_calls": 0, "code": {"old": OLD, "first_incumbent": SWITCH}, "archive": ARCHIVE,
            "ledger_sha256": LEDGER_HASH, "shared_initial_sha256": INITIAL_HASH, "seed": SEED,
            "first_proposal": {"old": old["1"], "new": new["1"]}, "low_energy_branch": branch,
            "archived_low_energy_result": truth[baseline_low["candidate_id"]],
            "joint_final_rule_for_slot_comparison": True, "steps_with_changed_local_slots": changed_slots,
            "ordinary": baseline, "reserved": reserve}
