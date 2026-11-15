"""Read-only feedback audit of the frozen Compact MDA qualification archive."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ra_lsbo.audit import load_json
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.records import finite_number
from ra_lsbo.risk_core.observations import OutputObservation

ARCHIVE_HASHES = {
    "plan.json": "087f587efcec17f966ffb9723821919f7fd52860a3e3a356933f893e5b381bdf",
    "submissions.jsonl": "b289acf2b6b6fe3565af602ea572ce8297fa497231d6417af5ed099d3c194d6d",
    "observations.jsonl": "dd5d7e6234d759c7053d8500203d3a90c54391a777b6514341407daa8974e28a",
    "task_65/budget.jsonl": "084e5e1bb9c6e7475051a9b76c0460adea1c83a9853736ea2eacf5d499a26301",
}
TASK_COUNTS = {55: (30, 30), 65: (7, 6)}


def feedback_from_settlement(risk):
    """Use only the archived trusted values and explicitly verified intervals."""
    state, evidence = risk["state"], risk["evidence"]
    trusted = state not in {"unknown", "infrastructure_error", "precheck_rejected"}
    complete = risk["objective_observed"] if trusted else None
    usable = risk["feasible_label"] if trusted else None
    bounds = evidence.get("verified_constraint_bounds", {}) if trusted else {}
    if not set(bounds) <= risk["constraints"].keys() or any(risk["constraint_observed"][name] for name in bounds):
        raise ValueError("Archived bound has an unknown head or overlaps an exact value")
    constraints = {}
    for name, value in sorted(risk["constraints"].items()):
        if trusted and risk["constraint_observed"][name]:
            obs = OutputObservation("exact", "archived_exact:" + name, value=value)
        elif name in bounds:
            obs = OutputObservation(**{"kind": "interval", **bounds[name]})
            if obs.kind != "interval":
                raise ValueError("Archived constraint bound must be an interval")
        else:
            constraints[name] = {"kind": "missing", "sign": None, "analytical": name in risk["analytical_constraints"]}
            continue
        constraints[name] = {"kind": obs.kind, "sign": obs.constraint_satisfaction(),
                             "analytical": name in risk["analytical_constraints"]}
    value = evidence.get("solver_discharge_margin_W")
    diagnostic = None
    if finite_number(value):
        diagnostic = {"margin_W": value, "source": evidence.get("solver_discharge_margin_source")}
    return {"raw_status": risk["raw_status"], "state": state, "failure_modes": risk["failure_modes"],
            "completed_with_exact_objective": complete, "usable": usable, "constraints": constraints,
            "discharge_diagnostic": diagnostic}


def summarize(rows):
    groups = []
    for distance in sorted({r["distance_nmi"] for r in rows}):
        outcomes = [r for r in rows if r["distance_nmi"] == distance]
        states = Counter(r["state"] for r in outcomes)
        hidden = [r for r in outcomes if r["state"] == "hidden_failure"]
        modes = Counter(mode for r in hidden for mode in r["failure_modes"] if not mode.startswith("constraint:"))
        heads = Counter((name, obs["kind"], str(obs["sign"])) for r in hidden
                        for name, obs in r["constraints"].items() if not obs["analytical"])
        diagnostics = Counter(str(r["discharge_diagnostic"]["source"])
                              for r in hidden if r["discharge_diagnostic"] is not None)
        groups.append({"distance_nmi": distance, "charged": len(outcomes), "states": dict(states),
                       "completed_with_exact_objective": sum(r["completed_with_exact_objective"] is True for r in outcomes),
                       "usable": sum(r["usable"] is True for r in outcomes), "hidden_failure_modes": dict(modes),
                       "hidden_failure_nonanalytical_feedback": [
                           {"constraint": name, "kind": kind, "sign": {"True": True, "False": False, "None": None}[sign],
                            "count": count}
                           for (name, kind, sign), count in sorted(heads.items())],
                       "hidden_failure_with_any_nonanalytical_sign": sum(
                           any(obs["sign"] is not None and not obs["analytical"] for obs in r["constraints"].values())
                           for r in hidden),
                       "hidden_failure_discharge_diagnostic_sources": dict(diagnostics)})
    return groups


def audit(source):
    files = set(ARCHIVE_HASHES) | {"task_55/budget.jsonl"}
    raw = {name: (source/name).read_bytes() for name in sorted(files)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    if any(hashes[name] != expected for name, expected in ARCHIVE_HASHES.items()):
        raise ValueError("Input does not match the frozen Compact MDA recovery snapshot")
    plan = load_json(raw["plan.json"])
    tasks = {task["distance_nmi"]: task for task in plan["tasks"]}
    if plan["schema"] != "compact_snapshot_recovery_v1" or list(tasks) != list(TASK_COUNTS):
        raise ValueError("Unexpected qualification task roster")
    submissions = [load_json(line) for line in raw["submissions.jsonl"].splitlines()]
    observations = [load_json(line) for line in raw["observations.jsonl"].splitlines()]
    if [r["phase_call"] for r in submissions] != list(range(1, 38)) or [
            r["evaluation"] for r in observations] != list(range(1, 37)):
        raise ValueError("Unexpected historical call inventory")
    by_index = {row["evaluation"]: row for row in observations}
    output = []
    for distance, (charged, settled) in TASK_COUNTS.items():
        task = tasks[distance]
        records = [row for row in submissions if row["task"] == distance]
        with RiskLedger(source/f"task_{distance}/budget.jsonl") as ledger:
            tickets = list(ledger.tickets.values())
        if (len(tickets) != charged or len(records) != charged
                or sum(ticket["state"] == "settled" for ticket in tickets) != settled):
            raise ValueError("Task ledger and historical call counts differ")
        for i, (submission, ticket) in enumerate(zip(records, tickets)):
            candidate = submission["proposal"]["candidate"]
            key = digest(candidate)
            if (candidate != ticket["candidate"] or candidate != task["candidates"][i]
                    or submission["candidate_key"] != key or ticket["candidate_key"] != key):
                raise ValueError("Qualification candidate binding differs")
            call = submission["phase_call"]
            if ticket["state"] == "settled":
                obs = by_index.pop(call)
                if (obs["candidate"] != candidate or obs["evaluator_contract_sha256"] != task["contract_id"]
                        or obs["distance_nmi"] != distance or obs["risk"] != ticket["risk"]):
                    raise ValueError("Observation and ledger settlement differ")
                feedback = feedback_from_settlement(obs["risk"])
            elif ticket["state"] == "started" and call == 37:
                feedback = {"raw_status": None, "state": "unresolved_started", "failure_modes": [],
                            "completed_with_exact_objective": None, "usable": None, "constraints": {},
                            "discharge_diagnostic": None}
            else:
                raise ValueError("Unexpected nonterminal ticket")
            output.append({"phase_call": call, "distance_nmi": distance, "token": ticket["token"],
                           "candidate_sha256": key, "contract": task["contract_id"], **feedback})
    if by_index or any((source/name).read_bytes() != data for name, data in raw.items()):
        raise ValueError("Historical observations were unbound or a source changed during the audit")
    return {"status": "AUDITED_ARCHIVED_X57_FEEDBACK", "source_sha256": hashes,
            "new_aircraft_calls": 0, "new_rcaide_mda_calls": 0,
            "cases": summarize(output), "calls": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original qualification_recovered directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report outside the archive")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents or output.exists():
        parser.error("Choose a new output JSON outside the qualification archive")
    result = audit(source)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(result["status"], len(result["calls"]), "archived calls; 0 new MDA calls")


if __name__ == "__main__":
    main()
