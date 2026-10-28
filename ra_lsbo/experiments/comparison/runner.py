"""Auditable single-writer execution for the external comparison roster."""

from collections import Counter
import csv
import os
from pathlib import Path
import time

import numpy as np

from ...budget import digest
from ..core import runner as base_runner
from ..core.metrics import flight_risk, paired_comparisons, qualification, trajectory_metrics
from ..core.problems import FlightScenario, make_problem
from ..core.protocol import BASE
from ..core.runner import Ledger, append, lines, plain, read, save
from ...partial_bo.problems import candidate_pool
from ...partial_bo.records import DesignRecord
from . import ARMS
from .methods import optimizer_for
from .protocol import SCHEMA, sha, verify_plan


def prepare(folder, profile="pilot", suite="mathematical"):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError("Choose a new comparison directory")
    from .protocol import build_plan
    plan = build_plan(profile, suite)
    verify_plan(plan)
    folder.mkdir(parents=True)
    snapshot = folder/"source_snapshot"
    snapshot.mkdir()
    for identity in plan["sources"].values():
        path = next(path for path, source_hash in plan["sources"].items() if source_hash == identity)
        (snapshot/(identity+".source")).write_bytes(Path(path).read_bytes())
    save(folder/"plan.json", plan)
    save(folder/"registration.json", {"plan_sha256": sha(folder/"plan.json"),
        "maximum_model_calls": plan["budget"]["maximum_model_calls"], "new_model_calls": 0})
    return plan


def load_plan(folder):
    folder = Path(folder)
    plan = read(folder/"plan.json")
    registration = read(folder/"registration.json")
    if (sha(folder/"plan.json") != registration["plan_sha256"]
            or registration["maximum_model_calls"] != plan["budget"]["maximum_model_calls"]
            or registration["new_model_calls"] != 0):
        raise ValueError("Prospective comparison plan or registration changed")
    verify_plan(plan)
    for identity in plan["sources"].values():
        if sha(folder/"source_snapshot"/(identity+".source")) != identity:
            raise ValueError("Comparison source snapshot changed")
    return plan


def _flight_validation(folder, plan, ledger, case, seed, method, incumbent):
    if incumbent is None:
        return flight_risk([])
    records = []
    for index, scenario in enumerate(plan["tasks"][case]["flight_scenarios"][str(seed)]):
        problem = make_problem(BASE, case, scenario=FlightScenario(**scenario), full_trajectory=True)
        records.append(ledger.evaluate(problem, incumbent.candidate,
            f"{case}/{seed}/random/{method}/flight/{index}", "held_out_flight_scenarios"))
    result = flight_risk(records)
    result.update(incumbent_design_id=incumbent.observation.design_id,
                  nominal_objective_Wh=incumbent.observation.objective.value)
    save(folder/"flight_validation"/case/f"{seed}_{method}.json",
         {"summary": result, "records": [row.to_dict() for row in records]})
    return result


def _trial(folder, plan, ledger, case, seed, initial, panel, method):
    profile = plan["profile"]
    problem = make_problem(BASE, case)
    output = folder/"optimization"/case/f"{seed}_{method}"
    output.mkdir(parents=True, exist_ok=False)
    optimizer = optimizer_for(problem, BASE, seed, method, profile)
    save(output/"manifest.json", {"case": case, "seed": seed, "method": method,
        "initial_digest": digest([row.to_dict() for row in initial]),
        "initial": [row.to_dict() for row in initial], "contract": problem.contract_id})
    history, online, start = list(initial), [], time.monotonic()
    for iteration in range(profile["horizon"]):
        pool, rejected = candidate_pool(problem,
            seed*10000+iteration+100000000*profile["cases"].index(case),
            profile["pool"], {row.observation.design_id for row in history})
        candidate, metadata = optimizer.propose(history, pool, iteration)
        append(output/"proposals.jsonl", {"iteration": iteration, "candidate": candidate, "pool": pool,
            "exact_screen_rejections": rejected, "metadata": metadata,
            "prefix_digest": digest([row.to_dict() for row in history])})
        record = ledger.evaluate(problem, candidate,
            f"{case}/{seed}/random/{method}/online/{iteration}", "online")
        append(output/"observations.jsonl", record.to_dict())
        history.append(record)
        online.append(record)
        optimizer.accept(record)
    result = {"case": case, "seed": seed, "initial_policy": "random", "method": method,
        **trajectory_metrics(initial, online, profile["horizon"]),
        "optimization_seconds": time.monotonic()-start}
    if case.startswith("aircraft_"):
        incumbent = min((row for row in history if row.usable_feasible),
                        key=lambda row: row.observation.objective.value, default=None)
        result["flight_risk"] = _flight_validation(folder, plan, ledger, case, seed, method, incumbent)
    save(output/"summary.json", result)
    print(f"{case} seed={seed} {method}: best={result['best_feasible']} calls={len(online)}", flush=True)
    return result


def run(folder, approved_call_limit=None):
    folder = Path(folder)
    plan = load_plan(folder)
    maximum = plan["budget"]["maximum_model_calls"]
    if (type(approved_call_limit) is not int or approved_call_limit < 1
            or maximum > approved_call_limit):
        raise ValueError(f"Complete prospective budget {maximum} exceeds approved limit "
                         f"{approved_call_limit}; an explicit --approved-call-limit is required")
    descriptor = os.open(str(folder/"execution.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    ledger = Ledger(folder, maximum)
    results, qualifications, integrations = [], {}, {}
    for case, task in plan["tasks"].items():
        problem = make_problem(BASE, case)
        anchors = [ledger.evaluate(problem, candidate, f"{case}/anchor/{i}", "qualification_anchors")
                   for i, candidate in enumerate(task["anchors"])]
        panel = [ledger.evaluate(problem, candidate, f"{case}/panel/{i}", "independent_panel")
                 for i, candidate in enumerate(task["panel"])]
        qualifications[case] = {**qualification(panel),
                                "anchors_usable": [bool(row.usable_feasible) for row in anchors]}
        save(folder/"qualification"/(case+".json"),
             {"summary": qualifications[case], "anchors": [r.to_dict() for r in anchors],
              "panel": [r.to_dict() for r in panel]})
        if case.startswith("aircraft_"):
            integrations[case] = base_runner.integration_check(folder, ledger, case, task["anchors"][0])
            if integrations[case]["status"] == "failed":
                raise RuntimeError("Aircraft integration check failed; optimization stopped")
        for seed in plan["profile"]["seeds"]:
            specification = task["initial"][str(seed)]["random"]
            initial = [ledger.evaluate(problem, candidate,
                f"{case}/{seed}/random/initial/{index}", "shared_initial")
                for index, candidate in enumerate(specification["random_candidates"])]
            for method in ARMS:
                results.append(_trial(folder, plan, ledger, case, seed, initial, panel, method))
    summary = {"profile": plan["profile"]["name"], "suite": plan["suite"],
        "summaries": results, "qualifications": qualifications, "integration": integrations,
        "paired_comparisons": paired_comparisons(results),
        "charged_model_calls": ledger.count, "charged_by_role": dict(ledger.roles),
        "maximum_model_calls": maximum, "unused_calls_not_reassigned": maximum-ledger.count,
        "new_rcaide_mda_calls": 0, "publication_confirmation": False}
    save(folder/"results.json", summary)
    return summary


def evidence_hashes(folder):
    """Bind all structured evidence, excluding the audit's own self-reference."""
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): sha(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix in {".json", ".jsonl"} and path != folder/"audit.json"}


def audit(folder, *, write=True):
    """Reconstruct the full registered campaign without re-evaluating any model.

    This verifies evidence and candidate-pool generation, NOT optimizer replay.
    Repeated auditing is idempotent only when the sealed evidence is unchanged.
    """
    folder = Path(folder)
    plan = load_plan(folder)
    before = evidence_hashes(folder)
    events = lines(folder/"calls.jsonl")
    if len(events) % 2 or any(events[index]["event"] != "started"
                              or events[index+1]["event"] != "settled"
                              for index in range(0, len(events), 2)):
        raise ValueError("Unsettled comparison call ledger")
    settled, starts, roles = {}, {}, Counter()
    for number in range(0, len(events), 2):
        start, end = events[number], events[number+1]
        payload = {name: start[name] for name in
                   ("number", "tag", "role", "candidate", "contract", "new_rcaide_mda")}
        record = DesignRecord.from_dict(end["record"])
        if (start["number"] != number//2+1 or start["call_id"] != digest(payload)
                or end["call_id"] != start["call_id"] or start["tag"] != end["tag"]
                or record.candidate != start["candidate"]
                or record.observation.contract_id != start["contract"]
                or digest(record.to_dict()) != end["record_digest"] or start["tag"] in settled
                or start["new_rcaide_mda"] is not False):
            raise ValueError("Call identity or settled record mismatch")
        settled[start["tag"]], starts[start["tag"]] = record, start
        roles[start["role"]] += 1
    if (len(settled) > plan["budget"]["maximum_model_calls"] or any(
            count > plan["budget"].get(role, -1) for role, count in roles.items())):
        raise ValueError("Comparison budget exceeded")
    expected_tags, reconstructed, qualifications, integrations = set(), [], {}, {}

    def claim(tag, role, problem, candidate=None):
        if tag not in settled:
            raise ValueError("Missing registered call: "+tag)
        if tag in expected_tags:
            raise ValueError("Repeated registered call: "+tag)
        expected_tags.add(tag)
        row, start = settled[tag], starts[tag]
        values = list(problem.precheck(row.candidate).values())
        if (start["role"] != role or row.observation.contract_id != problem.contract_id
                or row.candidate != problem.canonical(row.candidate)
                or any(not np.isfinite(value) or value > 0 for value in values)
                or (candidate is not None and row.candidate != candidate)):
            raise ValueError("Registered call role, contract or candidate changed: "+tag)
        return row

    checked = 0
    for case in plan["profile"]["cases"]:
        task, problem = plan["tasks"][case], make_problem(BASE, case)
        anchors = [claim(f"{case}/anchor/{i}", "qualification_anchors", problem, raw)
                   for i, raw in enumerate(task["anchors"])]
        panel = [claim(f"{case}/panel/{i}", "independent_panel", problem, raw)
                 for i, raw in enumerate(task["panel"])]
        qualifications[case] = {**qualification(panel),
                               "anchors_usable": [bool(row.usable_feasible) for row in anchors]}
        expected = {"summary": qualifications[case], "anchors": [row.to_dict() for row in anchors],
                    "panel": [row.to_dict() for row in panel]}
        if read(folder/"qualification"/(case+".json")) != plain(expected):
            raise ValueError("Qualification evidence or statistics changed")
        if case.startswith("aircraft_"):
            records = [claim(f"{case}/integration/{step}", "aircraft_step_convergence",
                             make_problem(BASE, case, integration_step=step), task["anchors"][0])
                       for step in (20., 5.)]
            exact = all(row.observation.objective.kind == "exact" for row in records)
            relative = (abs(records[0].observation.objective.value-records[1].observation.objective.value)
                        /max(abs(records[1].observation.objective.value), 1.)) if exact else None
            same = (records[0].diagnostics["termination_reason"] == records[1].diagnostics["termination_reason"]
                    and records[0].physical_feasible == records[1].physical_feasible)
            integration = {"steps_s": [20., 5.], "same_status_and_physical_label": same,
                "both_objectives_exact": exact, "relative_energy_error": relative,
                "status": "passed" if exact and same and relative <= .01
                else "inconclusive" if not exact and same else "failed"}
            if (integration["status"] == "failed"
                    or read(folder/"qualification"/(case+"_integration.json")) != integration):
                raise ValueError("Aircraft integration evidence failed or changed")
            integrations[case] = integration
        for seed in plan["profile"]["seeds"]:
            specification = task["initial"][str(seed)]["random"]
            initial = [claim(f"{case}/{seed}/random/initial/{i}", "shared_initial", problem, raw)
                       for i, raw in enumerate(specification["random_candidates"])]
            for method in ARMS:
                output = folder/"optimization"/case/f"{seed}_{method}"
                expected_manifest = {"case": case, "seed": seed, "method": method,
                    "contract": problem.contract_id, "initial": [row.to_dict() for row in initial],
                    "initial_digest": digest([row.to_dict() for row in initial])}
                if read(output/"manifest.json") != plain(expected_manifest):
                    raise ValueError("Methods do not share identical starting labels or trial identity")
                history, online = list(initial), []
                proposals = lines(output/"proposals.jsonl")
                if len(proposals) != plan["profile"]["horizon"]:
                    raise ValueError("Incomplete registered trial horizon")
                for iteration, proposal in enumerate(proposals):
                    record = claim(f"{case}/{seed}/random/{method}/online/{iteration}", "online", problem)
                    seen = {row.observation.design_id for row in history}
                    pool, rejected = candidate_pool(problem,
                        seed*10000+iteration+100000000*plan["profile"]["cases"].index(case),
                        plan["profile"]["pool"], seen)
                    if (proposal["iteration"] != iteration or proposal["candidate"] != record.candidate
                            or proposal["prefix_digest"] != digest([row.to_dict() for row in history])
                            or record.observation.design_id in seen or proposal["pool"] != pool
                            or proposal["exact_screen_rejections"] != rejected
                            or proposal["metadata"]["method"] != method):
                        raise ValueError("Proposal/evaluation/history/pool mismatch")
                    history.append(record)
                    online.append(record)
                    checked += 1
                if lines(output/"observations.jsonl") != plain([row.to_dict() for row in online]):
                    raise ValueError("Trial observations differ from ledger")
                saved = read(output/"summary.json")
                expected = {"case": case, "seed": seed, "initial_policy": "random", "method": method,
                            **trajectory_metrics(initial, online, plan["profile"]["horizon"])}
                if any(saved[key] != value for key, value in expected.items()):
                    raise ValueError("Trial metrics differ from settled evidence")
                if case.startswith("aircraft_"):
                    incumbent = min((row for row in history if row.usable_feasible),
                                    key=lambda row: row.observation.objective.value, default=None)
                    records = []
                    if incumbent is not None:
                        for i, scenario in enumerate(task["flight_scenarios"][str(seed)]):
                            flight_problem = make_problem(BASE, case, scenario=FlightScenario(**scenario),
                                                          full_trajectory=True)
                            records.append(claim(f"{case}/{seed}/random/{method}/flight/{i}",
                                "held_out_flight_scenarios", flight_problem, incumbent.candidate))
                    risk = flight_risk(records)
                    if incumbent is not None:
                        risk.update(incumbent_design_id=incumbent.observation.design_id,
                                    nominal_objective_Wh=incumbent.observation.objective.value)
                        expected_flight = {"summary": risk, "records": [row.to_dict() for row in records]}
                        if read(folder/"flight_validation"/case/f"{seed}_{method}.json") != plain(expected_flight):
                            raise ValueError("Flight scenario evidence or risk statistics changed")
                    elif (folder/"flight_validation"/case/f"{seed}_{method}.json").exists():
                        raise ValueError("Flight validation without a nominal feasible incumbent")
                    if saved["flight_risk"] != risk:
                        raise ValueError("Trial flight-risk summary differs from scenario evidence")
                reconstructed.append(saved)
    if set(settled) != expected_tags:
        raise ValueError("Unregistered or unconsumed call ledger entries")
    result = read(folder/"results.json")
    expected_result = {"profile": plan["profile"]["name"], "suite": plan["suite"],
        "summaries": reconstructed, "qualifications": qualifications, "integration": integrations,
        "paired_comparisons": paired_comparisons(reconstructed),
        "charged_model_calls": len(starts), "charged_by_role": dict(roles),
        "maximum_model_calls": plan["budget"]["maximum_model_calls"],
        "unused_calls_not_reassigned": plan["budget"]["maximum_model_calls"]-len(starts),
        "new_rcaide_mda_calls": 0, "publication_confirmation": False}
    if result != plain(expected_result):
        raise ValueError("Results differ from reconstructed campaign evidence")
    if evidence_hashes(folder) != before:
        raise ValueError("Evidence changed while being audited")
    audit_result = {"passed": True, "settled_calls": len(starts), "checked_proposals": checked,
        "source_bindings": len(plan["sources"]), "algorithm_replay": False,
        "evidence_hashes": before,
        "scope": "sources, full registered horizon, shared initialization, candidate pools, "
                 "ledger, metrics, qualification, integration, scenarios and evidence seal"}
    if write:
        path = folder/"audit.json"
        if path.exists():
            if read(path) != audit_result:
                raise ValueError("Evidence differs from the existing audit seal; do not overwrite history")
        else:
            save(path, audit_result)
    return audit_result


def report(folder):
    folder = Path(folder)
    plan = load_plan(folder)
    audit_result = read(folder/"audit.json")
    if not audit_result["passed"] or audit_result != audit(folder, write=False):
        raise ValueError("Comparison audit must pass on unchanged evidence before reporting")
    result = read(folder/"results.json")
    rows = result["summaries"]
    from .visualization import make_figures
    figures = make_figures(result, folder/"figures")
    with (folder/"summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["case", "seed", "method", "initial_calls", "online_calls", "has_feasible",
                  "best_feasible", "first_feasible_total_call", "first_feasible_censored",
                  "online_usable", "hidden_failures", "explicit_violations"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "explicit_violations": row["explicit_violations"]})
    grouped = {}
    for row in rows:
        grouped.setdefault((row["case"], row["method"]), []).append(row)
    text = ["# External Algorithm Comparison", "",
        f"Suite: {plan['suite']}; profile: {plan['profile']['name']}; "
        f"calls: {result['charged_model_calls']}/{result['maximum_model_calls']}.",
        "Exploratory screening only; adapters are not reference-code reproductions. No RCAIDE MDA calls.",
        "", "| Case | Method | Runs | Feasible runs | Median feasible objective |",
        "|---|---|---:|---:|---:|"]
    for (case, method), group in sorted(grouped.items()):
        values = sorted(row["best_feasible"] for row in group if row["has_feasible"])
        median = "NA" if not values else f"{float(np.median(values)):.6g}"
        text.append(f"| {case} | {method} | {len(group)} | {len(values)} | {median} |")
    text.extend(["", "Unknown and failed outcomes are retained. Conditional objective summaries are shown "
        "with feasible-run counts; use `results.json` for paired outcomes and full trajectories."])
    text.extend(["", "## Figures", "", "Convergence medians and final objectives condition on runs with a "
        "feasible solution; the feasible-run fraction and counts are shown separately. "
        "SOC risk bounds reflect unknown outcomes, are conditional on an available incumbent, "
        "and are not confidence intervals or flight certification."])
    for figure in figures:
        text.extend(["", f"![{figure.stem}](figures/{figure.name})"])
    with (folder/"report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(text)+"\n")
    return folder/"report.md"
