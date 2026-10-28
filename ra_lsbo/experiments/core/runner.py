"""Single-writer experiment runner; every model evaluation is journalled."""

from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ...budget import digest
from ...mechanism_system import SystemSettings
from ...partial_bo.problems import candidate_pool
from ...partial_bo.records import DesignRecord
from ...system_experiment import OriginalLSBOWarmStart
from ...soc_revision.acquisition import quantities
from .methods import POSTERIOR_ARMS, optimizer_for, posterior_for
from .metrics import flight_risk, paired_comparisons, probability_scores, qualification, trajectory_metrics
from .problems import AircraftCase, FlightScenario, make_problem
from .protocol import BASE, build_plan, sha, verify_plan


NATIVE_STOP = "Repeated missing/failed objective returns; stopped before wasting remaining budget"


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(plain(value), sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class Ledger:
    def __init__(self, folder, maximum):
        self.path, self.maximum = Path(folder)/"calls.jsonl", maximum
        if self.path.exists():
            raise RuntimeError("Existing journal: audit/reconcile; never automatically retry a campaign")
        self.count, self.pending, self.tags = 0, None, set()
        self.roles = Counter()

    def evaluate(self, problem, candidate, tag, role):
        if self.pending is not None or tag in self.tags or self.count >= self.maximum:
            raise RuntimeError("Unsettled call, repeated tag, or exhausted complete-study budget")
        canonical = problem.canonical(candidate)
        if canonical != candidate or any(value > 0 for value in problem.precheck(candidate).values()):
            raise ValueError("Evaluation requires a canonical geometry-screened candidate")
        self.count += 1
        self.tags.add(tag)
        call = {"number": self.count, "tag": tag, "role": role, "candidate": candidate,
                "contract": problem.contract_id, "new_rcaide_mda": False}
        self.pending = digest(call)
        append(self.path, {"event": "started", "call_id": self.pending, **call})
        start = time.monotonic()
        record = problem.evaluate(candidate)
        if record.candidate != candidate or record.observation.contract_id != problem.contract_id:
            raise RuntimeError("Evaluator returned evidence for a different design or contract")
        append(self.path, {"event": "settled", "call_id": self.pending, "tag": tag,
            "record": record.to_dict(), "record_digest": digest(record.to_dict()),
            "seconds": time.monotonic()-start})
        self.pending = None
        self.roles[role] += 1
        return record


def prepare(folder, profile="smoke", suite="combined"):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError("Choose a new experiment directory")
    plan = build_plan(profile, suite)
    folder.mkdir(parents=True)
    snapshot = folder/"source_snapshot"
    snapshot.mkdir()
    for path, identity in plan["sources"].items():
        destination = snapshot/(identity+".source")
        if not destination.exists():
            destination.write_bytes(Path(path).read_bytes())
    save(folder/"plan.json", plan)
    save(folder/"registration.json", {"plan_sha256": sha(folder/"plan.json"),
        "maximum_model_calls": plan["budget"]["maximum_model_calls"], "new_model_calls": 0})
    return plan


def load_verified(folder):
    folder = Path(folder)
    plan = read(folder/"plan.json")
    if sha(folder/"plan.json") != read(folder/"registration.json")["plan_sha256"]:
        raise ValueError("Prospective plan changed")
    verify_plan(plan)
    for identity in plan["sources"].values():
        if sha(folder/"source_snapshot"/(identity+".source")) != identity:
            raise ValueError("Source snapshot changed")
    return plan


def panel_scores(problem, base, history, panel, seed, method, profile):
    posterior = posterior_for(problem, base, history, seed, method, profile)
    values = quantities(posterior.predict([row.candidate for row in panel]), posterior.best, 1., .5)
    usable = [None if row.observation.status in {"unknown", "infrastructure_error"}
              else bool(row.usable_feasible) for row in panel]
    output = {"history_digest": digest([row.to_dict() for row in history]),
              "panel_digest": digest([row.to_dict() for row in panel]),
              "joint_usable": probability_scores(values["joint"], usable),
              "completion": probability_scores(values["success"], [None if row.observation.completion_label() is None
                    else bool(row.observation.completion_label()) for row in panel]),
              "constraints": {}, "probability_interpretation": "product of marginal posterior probabilities, not physical flight reliability"}
    for name, probabilities in values["probabilities"].items():
        labels = [None if row.observation.status in {"unknown", "infrastructure_error"}
                  else row.observation.constraint(name).constraint_satisfaction() for row in panel]
        output["constraints"][name] = probability_scores(probabilities, labels)
    output["predictions"] = {"joint": values["joint"].tolist(), "completion": values["success"].tolist(),
        "constraints": {name: value.tolist() for name, value in values["probabilities"].items()}}
    return output


def validate_flight(folder, plan, ledger, case, seed, policy, method, incumbent):
    if incumbent is None:
        return flight_risk([])
    records = []
    for number, scenario in enumerate(plan["tasks"][case]["flight_scenarios"][str(seed)]):
        problem = make_problem(BASE, case, scenario=FlightScenario(**scenario), full_trajectory=True)
        records.append(ledger.evaluate(problem, incumbent.candidate,
            f"{case}/{seed}/{policy}/{method}/flight/{number}", "held_out_flight_scenarios"))
    output = flight_risk(records)
    output["incumbent_design_id"] = incumbent.observation.design_id
    output["nominal_objective_Wh"] = incumbent.observation.objective.value
    save(folder/"flight_validation"/case/f"{seed}_{policy}_{method}.json",
         {"summary": output, "records": [row.to_dict() for row in records]})
    return output


def trial(folder, plan, ledger, case, seed, policy, method, initial, panel):
    profile = plan["profile"]
    problem = make_problem(BASE, case)
    output = folder/"optimization"/case/f"{seed}_{policy}_{method}"
    output.mkdir(parents=True, exist_ok=False)
    native = OriginalLSBOWarmStart(problem, BASE, [row.legacy_row(index) for index, row in enumerate(initial)],
        seed, SystemSettings(calls=profile["horizon"], failure_cap=profile["horizon"],
                             gp_steps=profile["native_gp_steps"]), output) if method == "original_lsbo" else None
    optimizer = optimizer_for(problem, BASE, seed, method, profile) if method in POSTERIOR_ARMS else None
    save(output/"manifest.json", {"case": case, "seed": seed, "initial_policy": policy, "method": method,
        "initial_digest": digest([row.to_dict() for row in initial]),
        "initial": [row.to_dict() for row in initial], "contract": problem.contract_id})
    history, online, stopped, start = list(initial), [], False, time.monotonic()
    for iteration in range(profile["horizon"]):
        pool, rejected = [], 0
        if native:
            proposal, metadata = native.propose()
            candidate = proposal["candidate"]
        else:
            # Same random stream across arms; only already-observed designs differ.
            pool, rejected = candidate_pool(problem, seed*10000+iteration+100000000*profile["cases"].index(case),
                profile["pool"], {row.observation.design_id for row in history})
            candidate, metadata = ((pool[0], {"mode": "uniform_random"}) if method == "random"
                                   else optimizer.propose(history, pool, iteration))
        append(output/"proposals.jsonl", {"iteration": iteration, "candidate": candidate, "pool": pool,
            "exact_screen_rejections": rejected, "metadata": metadata,
            "prefix_digest": digest([row.to_dict() for row in history])})
        record = ledger.evaluate(problem, candidate, f"{case}/{seed}/{policy}/{method}/online/{iteration}", "online")
        append(output/"observations.jsonl", record.to_dict())
        history.append(record)
        online.append(record)
        if native:
            try:
                native.accept(record.legacy_row(len(history)-1), metadata)
            except RuntimeError as error:
                if str(error) != NATIVE_STOP:
                    raise
                stopped = True
                break
        elif optimizer:
            optimizer.accept(record)
    result = {"case": case, "seed": seed, "initial_policy": policy, "method": method,
              **trajectory_metrics(initial, online, profile["horizon"], stopped),
              "optimization_seconds": time.monotonic()-start}
    if method in POSTERIOR_ARMS:
        save(output/"final_panel.json", panel_scores(problem, BASE, history, panel, seed, method, profile))
    if case.startswith("aircraft_"):
        incumbent = min((row for row in history if row.usable_feasible),
                        key=lambda row: row.observation.objective.value, default=None)
        result["flight_risk"] = validate_flight(folder, plan, ledger, case, seed, policy, method, incumbent)
    save(output/"summary.json", result)
    print(f"{case} seed={seed} {policy} {method}: best={result['best_feasible']} calls={len(online)}", flush=True)
    return result


def integration_check(folder, ledger, case, candidate):
    records = [ledger.evaluate(make_problem(BASE, case, integration_step=step), candidate,
                              f"{case}/integration/{step}", "aircraft_step_convergence") for step in (20., 5.)]
    exact = all(row.observation.objective.kind == "exact" for row in records)
    relative = (abs(records[0].observation.objective.value-records[1].observation.objective.value)
                /max(abs(records[1].observation.objective.value), 1.)) if exact else None
    same = (records[0].diagnostics["termination_reason"] == records[1].diagnostics["termination_reason"]
            and records[0].physical_feasible == records[1].physical_feasible)
    result = {"steps_s": [20., 5.], "same_status_and_physical_label": same,
              "both_objectives_exact": exact, "relative_energy_error": relative,
              "status": "passed" if exact and same and relative <= .01 else "inconclusive" if not exact and same else "failed"}
    save(folder/"qualification"/(case+"_integration.json"), result)
    return result


def run(folder, approved_call_limit=500):
    execution_start = time.monotonic()
    folder = Path(folder)
    plan = load_verified(folder)
    maximum = plan["budget"]["maximum_model_calls"]
    if type(approved_call_limit) is not int or approved_call_limit < 1 or maximum > approved_call_limit:
        raise ValueError(f"Complete prospective budget {maximum} exceeds approved limit {approved_call_limit}; no partial campaign started")
    # A retained lock after interruption requires investigation, not a fresh run.
    descriptor = os.open(str(folder/"execution.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    torch.set_num_threads(1)
    ledger = Ledger(folder, maximum)
    profile, results, qualifications, integrations = plan["profile"], [], {}, {}
    for case, task in plan["tasks"].items():
        problem = make_problem(BASE, case)
        anchors = [ledger.evaluate(problem, candidate, f"{case}/anchor/{index}", "qualification_anchors")
                   for index, candidate in enumerate(task["anchors"])]
        panel = [ledger.evaluate(problem, candidate, f"{case}/panel/{index}", "independent_panel")
                 for index, candidate in enumerate(task["panel"])]
        qualifications[case] = {**qualification(panel), "anchors_usable": [bool(row.usable_feasible) for row in anchors]}
        save(folder/"qualification"/(case+".json"), {"summary": qualifications[case],
            "anchors": [row.to_dict() for row in anchors], "panel": [row.to_dict() for row in panel]})
        if case.startswith("aircraft_"):
            integrations[case] = integration_check(folder, ledger, case, task["anchors"][0])
            if integrations[case]["status"] == "failed":
                raise RuntimeError("Aircraft integration check failed; settled calls retained, no optimization started for this case")
        for seed in profile["seeds"]:
            for policy in profile["initial_policies"]:
                initial_spec = task["initial"][str(seed)][policy]
                initial = [ledger.evaluate(problem, candidate, f"{case}/{seed}/{policy}/initial/{index}", "shared_initial")
                           for index, candidate in enumerate(initial_spec["random_candidates"])]
                if initial_spec["shared_anchor_index"] is not None:
                    initial.insert(0, anchors[initial_spec["shared_anchor_index"]])
                for method in ("ra_lsbo", "onehot_local", "no_intervals"):
                    save(folder/"common_prefix_panel"/case/f"{seed}_{policy}_{method}.json",
                         panel_scores(problem, BASE, initial, panel, seed, method, profile))
                for method in plan["arms"]:
                    results.append(trial(folder, plan, ledger, case, seed, policy, method, initial, panel))
    output = {"profile": profile["name"], "summaries": results, "qualifications": qualifications,
              "integration": integrations, "paired_comparisons": paired_comparisons(results),
              "charged_model_calls": ledger.count, "charged_by_role": dict(ledger.roles),
              "maximum_model_calls": maximum, "unused_calls_not_reassigned": maximum-ledger.count,
              "new_rcaide_mda_calls": 0, "publication_confirmation": False,
              "execution_seconds": time.monotonic()-execution_start,
              "charged_model_seconds": sum(row["seconds"] for row in lines(ledger.path) if row["event"] == "settled")}
    save(folder/"results.json", output)
    return output


def audit(folder):
    folder = Path(folder)
    plan = load_verified(folder)
    events = lines(folder/"calls.jsonl")
    starts = [row for row in events if row["event"] == "started"]
    ends = [row for row in events if row["event"] == "settled"]
    if len(starts) != len(ends) or len(events) != 2*len(starts):
        raise ValueError("Unsettled calls: do not retry or silently relabel")
    lookup, roles = {}, Counter()
    for index, (start, end) in enumerate(zip(starts, ends), 1):
        payload = {name: start[name] for name in ("number", "tag", "role", "candidate", "contract", "new_rcaide_mda")}
        record = DesignRecord.from_dict(end["record"])
        if (events[2*index-2] != start or events[2*index-1] != end or start["number"] != index
                or start["call_id"] != digest(payload) or end["call_id"] != start["call_id"]
                or start["tag"] != end["tag"] or start["tag"] in lookup
                or record.candidate != start["candidate"] or record.observation.contract_id != start["contract"]
                or digest(record.to_dict()) != end["record_digest"] or start["new_rcaide_mda"]):
            raise ValueError("Call identity, evidence, or settlement sequence mismatch")
        lookup[start["tag"]] = record
        roles[start["role"]] += 1
    if len(starts) > plan["budget"]["maximum_model_calls"] or any(
            count > plan["budget"].get(role, -1) for role, count in roles.items()):
        raise ValueError("Role or total evaluation budget exceeded")
    summaries, checked = [], 0
    for case, task in plan["tasks"].items():
        panel_ids = {digest(candidate) for candidate in task["panel"]}
        for index, raw in enumerate(task["panel"]):
            if lookup[f"{case}/panel/{index}"].candidate != raw:
                raise ValueError("Independent panel changed")
        panel_records = [lookup[f"{case}/panel/{index}"] for index in range(len(task["panel"]))]
        anchor_records = [lookup[f"{case}/anchor/{index}"] for index in range(len(task["anchors"]))]
        if [row.candidate for row in anchor_records] != task["anchors"]:
            raise ValueError("Declared anchors changed")
        expected_qualification = {**qualification(panel_records),
                                  "anchors_usable": [bool(row.usable_feasible) for row in anchor_records]}
        if read(folder/"qualification"/(case+".json"))["summary"] != expected_qualification:
            raise ValueError("Qualification statistics disagree with the independent panel")
        for seed in plan["profile"]["seeds"]:
            for policy in plan["profile"]["initial_policies"]:
                specification = task["initial"][str(seed)][policy]
                initial = [lookup[f"{case}/{seed}/{policy}/initial/{index}"]
                           for index in range(len(specification["random_candidates"]))]
                if [row.candidate for row in initial] != specification["random_candidates"]:
                    raise ValueError("Shared initial set changed")
                if specification["shared_anchor_index"] is not None:
                    initial.insert(0, lookup[f"{case}/anchor/{specification['shared_anchor_index']}"])
                for method in plan["arms"]:
                    output = folder/"optimization"/case/f"{seed}_{policy}_{method}"
                    manifest = read(output/"manifest.json")
                    if manifest["initial_digest"] != digest([row.to_dict() for row in initial]):
                        raise ValueError("Arm changed the paired initial labels")
                    history, online = list(initial), []
                    for iteration, proposal in enumerate(lines(output/"proposals.jsonl")):
                        record = lookup[f"{case}/{seed}/{policy}/{method}/online/{iteration}"]
                        if (proposal["iteration"] != iteration or proposal["candidate"] != record.candidate
                                or proposal["prefix_digest"] != digest([row.to_dict() for row in history])
                                or record.observation.design_id in {row.observation.design_id for row in history}
                                or any(row.observation.design_id in panel_ids for row in history)):
                            raise ValueError("Proposal/history/evaluation mismatch or panel contamination")
                        history.append(record)
                        online.append(record)
                        checked += 1
                    if lines(output/"observations.jsonl") != plain([row.to_dict() for row in online]):
                        raise ValueError("Trial outcomes differ from settled evidence")
                    summary = read(output/"summary.json")
                    expected = trajectory_metrics(initial, online, plan["profile"]["horizon"], summary["stopped"])
                    if any(summary[key] != value for key, value in expected.items()):
                        raise ValueError("Trajectory metrics disagree with settled evidence")
                    if "flight_risk" in summary and summary["flight_risk"]["n"]:
                        path = folder/"flight_validation"/case/f"{seed}_{policy}_{method}.json"
                        saved = read(path)
                        risk_records = [lookup[f"{case}/{seed}/{policy}/{method}/flight/{index}"]
                                        for index in range(plan["profile"]["scenarios"])]
                        if saved["records"] != plain([row.to_dict() for row in risk_records]):
                            raise ValueError("Flight scenario outcomes differ from settled calls")
                        if summary["flight_risk"] != saved["summary"]:
                            raise ValueError("Trial and held-out flight summaries differ")
                        incumbent = min((row for row in history if row.usable_feasible),
                                        key=lambda row: row.observation.objective.value, default=None)
                        if incumbent is None or saved["summary"]["incumbent_design_id"] != incumbent.observation.design_id:
                            raise ValueError("Post-validation did not use the nominal feasible incumbent")
                        for record, scenario in zip(risk_records, task["flight_scenarios"][str(seed)]):
                            expected_problem = make_problem(BASE, case, scenario=FlightScenario(**scenario), full_trajectory=True)
                            if (record.candidate != incumbent.candidate
                                    or record.observation.contract_id != expected_problem.contract_id):
                                raise ValueError("Post-validation scenario or incumbent changed")
                        risk = flight_risk(risk_records)
                        if any(saved["summary"][key] != value for key, value in risk.items()):
                            raise ValueError("Flight risk metrics mismatch")
                    summaries.append(summary)
    result = read(folder/"results.json")
    if (result["summaries"] != summaries or result["charged_model_calls"] != len(starts)
            or result["charged_by_role"] != dict(roles)
            or result["paired_comparisons"] != paired_comparisons(summaries)):
        raise ValueError("Campaign summaries or call accounting mismatch")
    output = {"passed": True, "settled_calls": len(starts), "checked_proposals": checked,
              "source_bindings": len(plan["sources"]), "new_rcaide_mda_calls": 0,
              "results_sha256": sha(folder/"results.json"), "calls_sha256": sha(folder/"calls.jsonl"),
              "algorithm_replay": False, "scope": "source snapshots, evidence, pairing, histories, budgets and metric reconstruction"}
    save(folder/"audit.json", output)
    return output


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(folder):
    folder = Path(folder)
    load_verified(folder)
    audit_result = read(folder/"audit.json")
    if (not audit_result["passed"] or audit_result["results_sha256"] != sha(folder/"results.json")
            or audit_result["calls_sha256"] != sha(folder/"calls.jsonl")):
        raise ValueError("Audit must pass before reporting")
    result = read(folder/"results.json")
    rows = result["summaries"]
    write_csv(folder/"summary.csv", rows, ["case", "seed", "initial_policy", "method", "initial_kind", "initial_calls",
        "online_calls", "has_feasible", "best_feasible", "first_feasible_total_call", "first_feasible_censored",
        "online_usable", "hidden_failures", "numerical_failures", "terminal_unknown", "optimization_seconds"])
    curves = [{"case": row["case"], "seed": row["seed"], "initial_policy": row["initial_policy"],
               "method": row["method"], **point} for row in rows for point in row["curve"]]
    write_csv(folder/"curves.csv", curves, ["case", "seed", "initial_policy", "method", "online_budget",
        "nominal_total_budget", "actual_calls", "best_feasible", "has_feasible"])
    risks = [{"case": row["case"], "seed": row["seed"], "initial_policy": row["initial_policy"],
              "method": row["method"], **row["flight_risk"]} for row in rows if "flight_risk" in row]
    write_csv(folder/"flight_risk.csv", risks, ["case", "seed", "initial_policy", "method", "status", "n",
        "soc_violations", "soc_unknown", "soc_failure_fraction_bounds", "soc_failure_probability_95_conservative",
        "soc_shortfall_cvar_bounds", "completed_mission_count", "completed_only_mean_energy_Wh"])
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["initial_policy"], row["method"], row["initial_kind"])].append(row)
    title = {"mathematical": "Mathematical benchmark report",
             "simplified_aircraft": "Simplified aircraft report",
             "combined": "Mathematical and reduced-aircraft experiment report"}[read(folder/"plan.json")["suite"]]
    text = [f"# {title}", "",
        f"Profile: {result['profile']}; model calls: {result['charged_model_calls']}/{result['maximum_model_calls']}; RCAIDE calls: 0.",
        "", "Smoke/pilot runs verify software only; no stable optimization advantage or flight certification is claimed.",
        "Nominal feasibility uncertainty and held-out physical scenario risk are reported separately.", "",
        "| Case | Policy | Method | Initial | Runs | Feasible runs | Conditional median objective |", "|---|---|---|---|---:|---:|---:|"]
    for (case, policy, method, kind), group in sorted(groups.items()):
        values = [row["best_feasible"] for row in group if row["has_feasible"]]
        median = "NA" if not values else f"{np.median(values):.6g}"
        text.append(f"| {case} | {policy} | {method} | {kind} | {len(group)} | {len(values)} | {median} |")
    text += ["", "Conditional objective medians exclude unsuccessful runs; feasible-run counts must be read alongside them.",
             "All labels, including shared initial sets, anchors, independent panels, integration checks and scenario tests, are charged.",
             "Unknown final SOC produces CVaR/probability bounds; no exact final value is invented.",
             "Probability scores are in common_prefix_panel/ and optimization/*/*/final_panel.json; no test-panel calibration is fitted.",
             "Full paired comparisons, qualification intervals, unused budget and flight-risk metrics are in results.json."]
    with (folder/"report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(text)+"\n")
    return folder/"report.md"
