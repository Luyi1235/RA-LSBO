"""PR70 prospective three-arm support-consistency pilot; stdlib protocol/audit."""
import json
import math
import statistics
from pathlib import Path
from pr69_evidence_verify import require, fingerprint, native_digest as digest, feasible, best
from ra_lsbo.experiments.information_value_stage3.protocol import compare
from ra_lsbo.experiments.information_value_stage3.runner import _decision
from ra_lsbo.experiments.information_value_stage1.io import verify as verify_manifest

PARENT = "fd56186bb75b29b0df96bb2c27d67791c5f0d73b"
CASES = ("branin_ordinary", "branin_sparse", "rosenbrock_ordinary", "rosenbrock_sparse")
ARMS = ("cei_myopic", "cei_two_step", "cei_two_step_next_pool")


def plan(profile):
    require(profile in ("ci", "smoke", "full"), "Unregistered profile")
    cases = ("rosenbrock_sparse",) if profile == "ci" else CASES
    seeds = (998890,) if profile == "ci" else ((998891,) if profile == "smoke" else tuple(range(998901, 998905)))
    steps = 6 if profile == "full" else 2
    groups = len(cases)*len(seeds)
    return {"schema": "pr70_next_pool_isolation_v1", "profile": profile,
        "cases": list(cases), "seeds": list(seeds), "arms": list(ARMS),
        "initial": 12, "steps": steps, "pool": 128, "initial_calls": groups*12,
        "online_calls": groups*len(ARMS)*steps, "total_calls": groups*(12+len(ARMS)*steps),
        "primary_pair": [ARMS[2], ARMS[1]], "reference_pair": [ARMS[2], ARMS[0]],
        "first_action_shortlist": 4, "fantasies_per_region_branch": 4,
        "only_primary_change": "current-pool continuation replaced by exact next deterministic protected candidate pool",
        "candidate_generator_unchanged": True, "future_outcomes_accessed": False,
        "all_real_feedback_preserved": True, "early_stopping": False,
        "four_seed_exploratory_not_confirmatory": True,
        "equal_true_calls_not_equal_wall_time": True,
        "aircraft_calls": 0, "rcaide_calls": 0}


def check_transition(support, raw, model, iteration):
    if support is None:
        return False
    require(support["iteration"] == iteration, "Wrong future iteration")
    require(support["raw_pool_sha256"] == digest(raw), "Forecast next raw pool differs from execution")
    require(support["model_pool_sha256"] == digest(model)
            and support["model_candidates"] == model, "Forecast next protected pool differs from execution")
    return True


def summarize(rows, spec):
    indexed = {(r["case"], r["seed"], r["arm"]): r for r in rows}
    expected = {(c, s, a) for c in spec["cases"] for s in spec["seeds"] for a in ARMS}
    require(len(rows) == len(indexed) and set(indexed) == expected, "Incomplete/duplicate roster")
    result = {}
    for c in spec["cases"]:
        for s in spec["seeds"]:
            require(len({indexed[c, s, a]["initial_hash"] for a in ARMS}) == 1, "Initial evidence mismatch")
        item = {"arms": {}, "pairs": {}}
        for a in ARMS:
            group = [indexed[c, s, a] for s in spec["seeds"]]
            values = [r["final_best"] for r in group if r["final_best"] is not None]
            item["arms"][a] = {"trials": len(group), "final_feasible": len(values),
                "cold_discoveries": sum(r["initial_best"] is None and r["final_best"] is not None for r in group),
                "nonfeasible": sum(r["nonfeasible"] for r in group),
                "online_calls": sum(r["online_calls"] for r in group),
                "conditional_objective_mean": statistics.mean(values) if values else None,
                "conditional_objective_median": statistics.median(values) if values else None,
                **{k: sum(r[k] for r in group) for k in ("proposal_seconds", "fantasy_head_fits_started",
                    "fantasy_head_fits_completed", "fantasy_fallbacks", "model_free_calls", "future_pool_checks")}}
        for left, right in ((ARMS[2], ARMS[1]), (ARMS[2], ARMS[0]), (ARMS[1], ARMS[0])):
            wlt, deltas, details, changed = [0, 0, 0], [], [], 0
            for s in spec["seeds"]:
                a, b = indexed[c, s, left], indexed[c, s, right]
                outcome = compare(a["final_best"], b["final_best"])
                wlt[0 if outcome == 1 else 1 if outcome == -1 else 2] += 1
                delta = a["final_best"] - b["final_best"] if a["final_best"] is not None and b["final_best"] is not None else None
                if delta is not None:
                    deltas.append(delta)
                details.append({"seed": s, "outcome": outcome, "objective_delta": delta})
                changed += a["path_hash"] != b["path_hash"]
            item["pairs"][left+"_vs_"+right] = {"wins_losses_ties": wlt, "both_feasible": len(deltas),
                "objective_delta_mean": statistics.mean(deltas) if deltas else None,
                "changed_paths": changed, "per_seed": details}
        result[c] = item
    return result


def check_receipt(meta, candidate, iteration, steps, arm):
    d = _decision(meta, candidate)
    iv = meta.get("information_value_decision")
    if iv is None:
        require(meta.get("coverage_reason") is not None, "Missing forecast without declared coverage")
        return d
    require(iv["remaining_actual_calls"] == steps-iteration, "Wrong remaining budget")
    pool = meta["pool"]
    index = iv["selected_index"]
    require(pool[index] == candidate and iv["candidate_pool_hash"] == digest(pool), "Root binding mismatch")
    require(iv["quadrature_nodes_per_region_branch"] == 4, "Quadrature changed")
    if iteration == steps-1 or arm == ARMS[0]:
        require(iv["depth"] == 1 and iv["selected_index"] == iv["myopic_index"]
                and not iv["fantasies"] and d["fantasy_head_fits_started"] == 0, "Last/myopic step is not myopic")
    if arm != ARMS[2] or iv["depth"] == 1:
        return d
    supports = iv["future_supports"]
    require(iv["selected_future_support"] == supports[str(index)], "Selected support mismatch")
    for k, support in supports.items():
        require(support["first_candidate_sha256"] == digest(pool[int(k)])
                and support["iteration"] == iteration+1
                and support["generation_uses_observations"] is False, "Future support identity mismatch")
        points = support["model_candidates"]
        require(points and support["model_pool_sha256"] == digest(points)
                and pool[int(k)] not in points and len({digest(x) for x in points}) == len(points), "Invalid future points")
    continuations, masses = {}, {}
    for f in iv["fantasies"]:
        i = f["first_index"]
        obs = f["observation"]
        if obs["aborted"]:
            require(obs["region"] == {"kind": "interval", "lower": 0., "lower_inclusive": False}
                    and obs["objective"] == {"kind": "missing"}, "Hidden abort value in fantasy")
        require(math.isfinite(f["weight"]) and f["weight"] > 0 and math.isfinite(f["continuation"])
                and f["continuation"] >= 0, "Invalid fantasy contribution")
        masses[i] = masses.get(i, 0.) + f["weight"]
        continuations[i] = continuations.get(i, 0.) + f["weight"]*f["continuation"]
        if f["second_index"] is not None:
            second = supports[str(i)]["model_candidates"][f["second_index"]]
            require(f["second_candidate_sha256"] == digest(second), "Second-action binding mismatch")
    if not iv["fallback"]:
        for i in iv["first_action_indices"]:
            require(math.isclose(masses[i], 1., abs_tol=1e-10), "Broken fantasy weights")
            require(math.isclose(iv["scores"][i], iv["myopic_scores"][i]+continuations[i], rel_tol=1e-10, abs_tol=1e-10), "Broken Q2 sum")
        require(index == max(iv["first_action_indices"], key=lambda i: (iv["scores"][i], -i)), "Wrong Q2 winner")
    return d


def verify(output):
    root = Path(output)
    status = verify_manifest(root)
    def read(p):
        return json.loads((root/p).read_text(encoding="utf-8"))
    def lines(p):
        return [json.loads(s) for s in (root/p).read_text(encoding="utf-8").splitlines() if s.strip()]
    spec, report = read("plan.json"), read("report.json")
    require(spec == plan(spec["profile"]), "Unregistered plan")
    require(report["code_sha"] == read("source_receipt.json")["head_sha"], "Source mismatch")
    events, count = lines("evaluations.jsonl"), spec["total_calls"]
    require(len(events) == 2*count, "Incomplete ledger")
    settled = {}
    for i in range(count):
        a, b = events[2*i:2*i+2]
        require(a["ticket"] == b["ticket"] == i and a["event"] == "started" and b["event"] == "settled", "Ticket mismatch")
        feasible(b["record"])
        require(a["candidate"] == b["record"]["candidate"] and a["candidate_id"] == b["candidate_id"] == fingerprint(a["candidate"])
                and a["purpose"] == b["purpose"], "Ledger binding mismatch")
        settled[i] = b
    used, rows = set(), []
    for case in spec["cases"]:
        for seed in spec["seeds"]:
            init = read(Path("shared_initial")/case/(str(seed)+".json"))
            require(len(init["records"]) == len(init["tickets"]) == 12 and init["hash"] == fingerprint(init["records"]), "Initial mismatch")
            for j, (t, rec) in enumerate(zip(init["tickets"], init["records"])):
                require(t not in used and settled[t]["record"] == rec and settled[t]["purpose"] ==
                        {"phase": "initial", "case": case, "seed": seed, "index": j}, "Initial ledger mismatch")
                used.add(t)
            for arm in ARMS:
                folder = Path("trials")/case/str(seed)/arm
                props, obs, row = lines(folder/"proposals.jsonl"), lines(folder/"observations.jsonl"), read(folder/"result.json")
                require(len(props) == len(obs) == spec["steps"], "Online count mismatch")
                history = list(init["records"])
                path, curve, decisions, previous, checks = [], [best(history)], [], None, 0
                for i, (p, o) in enumerate(zip(props, obs)):
                    rec, t = o["record"], o["ticket"]
                    require(t not in used and settled[t]["record"] == rec and settled[t]["purpose"] ==
                            {"phase": "online", "case": case, "seed": seed, "arm": arm, "iteration": i}, "Online ledger mismatch")
                    used.add(t)
                    require(p["iteration"] == o["iteration"] == i and p["candidate"] == rec["candidate"] and
                        p["candidate_hash"] == o["candidate_hash"] == fingerprint(rec["candidate"]) and
                        p["history_hash"] == fingerprint(history), "Prefix binding mismatch")
                    require(len(p["raw_pool"]) == 128 and fingerprint(p["raw_pool"]) == p["pool_hash"]
                            and rec["candidate"] in p["raw_pool"] and not any(r["candidate"] == rec["candidate"] for r in history), "Pool/repeat mismatch")
                    require(not any(c == r["candidate"] for c in p["protected_pool"] for r in history), "History repeats in model support")
                    checks += check_transition(previous, p["raw_pool"], p["protected_pool"], i)
                    meta = p["metadata"]
                    require(meta["pool"] == p["protected_pool"], "Actual model/coverage pool mismatch")
                    decision = check_receipt(meta, rec["candidate"], i, spec["steps"], arm)
                    decision.update(iteration=i, candidate_hash=fingerprint(rec["candidate"]), nonfeasible=int(not feasible(rec)))
                    decisions.append(decision)
                    previous = meta.get("information_value_decision", {}).get("selected_future_support")
                    history.append(rec)
                    path.append(fingerprint(rec["candidate"]))
                    curve.append(best(history))
                require(row["case"] == case and row["seed"] == seed and row["arm"] == arm and row["initial_hash"] == init["hash"]
                        and row["curve"] == curve and row["initial_best"] == curve[0] and row["final_best"] == curve[-1]
                        and row["decisions"] == decisions and row["path_hash"] == fingerprint(path)
                        and row["online_calls"] == spec["steps"] and row["future_pool_checks"] == checks, "Trial mismatch")
                require(row["nonfeasible"] == sum(d["nonfeasible"] for d in decisions)
                        and row["fantasy_fallbacks"] == sum(d["fantasy_fallback"] for d in decisions)
                        and row["model_free_calls"] == sum(not d["forecast_available"] for d in decisions), "Trial counters mismatch")
                for k in ("fantasy_head_fits_started", "fantasy_head_fits_completed"):
                    require(row[k] == sum(d[k] for d in decisions), "Fantasy counters mismatch")
                require(math.isclose(row["proposal_seconds"], sum(p["proposal_seconds"] for p in props), abs_tol=1e-8), "Time mismatch")
                rows.append(row)
    require(used == set(range(count)) and report["trials"] == rows and report["summary"] == summarize(rows, spec), "Campaign mismatch")
    require(report["actual_calls_started"] == report["actual_calls_settled"] == count
            and report["aircraft_calls"] == report["rcaide_calls"] == 0, "Bill mismatch")
    return {**status, "trials": len(rows), "actual_calls_started": count, "actual_calls_settled": count,
        "initial_calls": spec["initial_calls"], "online_calls": spec["online_calls"],
        "future_pool_checks": sum(r["future_pool_checks"] for r in rows),
        "gp_fits_in_this_verification": 0, "new_evaluations_in_this_verification": 0}
