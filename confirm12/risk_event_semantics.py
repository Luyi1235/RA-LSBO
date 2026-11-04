"""Read-only forensic labels for the frozen PR45 reduced-aircraft campaign.

This module never imports, constructs or invokes an aircraft evaluator. It does
NOT change the original GP completion labels or optimize a new candidate.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

SCHEMA = "ra_lsbo_pr45_risk_event_semantics_v1"
FROZEN_SOURCE = "386082beeacac8ae43d0dfb1ccda5c5ad3b356cf"
FROZEN_PLAN_SHA256 = "ac5f1424f3c8d52eff9ffd835cb5004224be965afee8ec15069e1fea38c43012"
FROZEN_SQLITE_SHA256 = "a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7"
EXPECTED_CALLS = 151006
EXPECTED_BY_CASE = {"aircraft_ordinary": 75520, "aircraft_sparse": 75486}
REQUIRED = frozenset(("span", "packing", "stall", "root_bending", "soc", "current", "voltage", "power"))
PHYSICAL_ABORTS = frozenset((
    "physical_soc_reserve_abort", "physical_battery_power_domain", "physical_discharge_domain",
))
FIELDS = (
    "call_id", "case", "seed", "role", "arm", "idx", "design_sha256", "status",
    "status_completion", "mission_completed", "objective_kind", "objective_exact",
    "physical_feasibility", "usable_feasible", "mission_and_usable_feasible",
    "known_violations", "unknown_constraints", "termination_reason", "failure_cause",
    "status_completed_mission_incomplete", "semantic_issues",
)
GROUP_FIELDS = (
    "case", "role", "arm", "seed_groups", "calls", "status_completed", "status_hidden_failure",
    "mission_true", "mission_false", "mission_unknown", "objective_exact",
    "physical_feasible_true", "physical_feasible_false", "physical_feasible_unknown",
    "usable_feasible", "mission_and_usable_feasible", "status_completed_mission_incomplete",
    "objective_interval", "known_constraint_violation", "unknown_constraint_rows", "semantic_issue_rows",
)


def digest(value):
    """Original ledger's sort-key JSON SHA256 identity, not a repr hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf8")).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def constraint_sign(output):
    """Tri-state truth about g<=0 from exact, one-sided or bounded evidence."""
    kind = output["kind"]
    if kind == "exact":
        value = output["value"]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Nonfinite or nonnumeric exact constraint")
        return value <= 0
    if kind == "interval":
        low, high = output.get("lower"), output.get("upper")
        if high is not None and high <= 0:
            return True
        if low is not None and (low > 0 or (low == 0 and output.get("lower_inclusive", True) is False)):
            return False
        return None
    if kind == "missing":
        return None
    raise ValueError("Unsupported observation kind")


def label_record(record):
    """Derive separate *descriptive* labels, never reclassify physical outputs.

    status_completion reproduces RunObservation.completion_label semantics.
    mission_completed comes only from an explicit diagnostic or known final
    termination code; unknown remains null. Physical feasibility is tri-state.
    """
    packet, diag = record["observation"], record["diagnostics"]
    status = packet["status"]
    if status not in ("completed", "hidden_failure", "infrastructure_error", "unknown"):
        raise ValueError("Unregistered observation status")
    reason = diag.get("termination_reason")
    stated = diag.get("mission_completed")
    if stated is not None and type(stated) is not bool:
        raise ValueError("Nonboolean mission_completed diagnostic")
    mission = (stated if stated is not None else True if reason == "complete"
               else False if reason in PHYSICAL_ABORTS else None)
    completion = 1 if status == "completed" else 0 if status == "hidden_failure" else None
    objective = packet["objective"]
    kind = objective["kind"]
    if kind not in ("exact", "interval", "missing"):
        raise ValueError("Unregistered objective evidence")
    outputs = packet["constraints"]
    if not isinstance(outputs, (list, tuple)) or len(outputs) != len(REQUIRED) or {n for n, _ in outputs} != REQUIRED:
        raise ValueError("Expected the frozen eight named aircraft constraints")
    heads = {name: (output["kind"], constraint_sign(output)) for name, output in outputs}
    signs = [s for _, s in heads.values()]
    violated = sum(s is False for s in signs)
    unknown = sum(s is None for s in signs)
    physical = (False if diag.get("certified_physical_infeasible") is True or violated
                else True if not unknown else None)
    usable = status == "completed" and physical is True and kind == "exact"
    issues = []
    if mission is True and (status != "completed" or kind != "exact"):
        issues.append("mission_true_without_completed_exact_target")
    if mission is False and kind == "exact":
        issues.append("mission_false_but_exact_target")
    if stated is not None and reason in PHYSICAL_ABORTS | {"complete"} and stated != (reason == "complete"):
        issues.append("mission_diagnostic_termination_disagreement")
    if usable and mission is not True:
        issues.append("legacy_usable_without_confirmed_mission")
    if status == "completed" and reason == "complete" and mission is not True:
        issues.append("completed_termination_but_mission_not_true")
    return {
        "status": status, "status_completion": completion, "mission_completed": mission,
        "objective_kind": kind, "objective_exact": kind == "exact",
        "physical_feasibility": physical, "usable_feasible": usable,
        "mission_and_usable_feasible": usable and mission is True,
        "known_violations": violated, "unknown_constraints": unknown,
        "termination_reason": reason, "failure_cause": packet.get("failure_cause"),
        "status_completed_mission_incomplete": status == "completed" and mission is False,
        "semantic_issues": issues, "constraint_heads": heads,
    }


def _stat(counter, labels):
    counter["calls"] += 1
    counter["status_completed"] += labels["status"] == "completed"
    counter["status_hidden_failure"] += labels["status"] == "hidden_failure"
    for value, key in ((True, "mission_true"), (False, "mission_false"), (None, "mission_unknown")):
        counter[key] += labels["mission_completed"] is value
    counter["objective_exact"] += labels["objective_exact"]
    counter["objective_interval"] += labels["objective_kind"] == "interval"
    for value, key in ((True, "physical_feasible_true"), (False, "physical_feasible_false"), (None, "physical_feasible_unknown")):
        counter[key] += labels["physical_feasibility"] is value
    counter["usable_feasible"] += labels["usable_feasible"]
    counter["mission_and_usable_feasible"] += labels["mission_and_usable_feasible"]
    counter["status_completed_mission_incomplete"] += labels["status_completed_mission_incomplete"]
    counter["known_constraint_violation"] += labels["known_violations"] > 0
    counter["unknown_constraint_rows"] += labels["unknown_constraints"] > 0
    counter["semantic_issue_rows"] += bool(labels["semantic_issues"])


def _compact(counter):
    return {field: int(counter[field]) for field in GROUP_FIELDS[4:]}


def _csv_scalar(value):
    if value is None:
        return ""
    if type(value) is bool:
        return int(value)
    if isinstance(value, list):
        return ";".join(value)
    return value


def _read_only_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def run(source, output, *, max_records=None, strict_identity=True):
    """Read the original frozen SQLite, writing ONLY into a new output folder.

    strict_identity=False exists solely to exercise tiny SQLite test fixtures;
    the public one-click entry never exposes that switch.
    """
    root, dest = Path(source).resolve(), Path(output).resolve()
    if dest == root or root in dest.parents or dest.exists():
        raise ValueError("A fresh output path outside the original source is required")
    if max_records is not None and (type(max_records) is not int or max_records < 1):
        raise ValueError("Positive max_records or full run required")
    plan_path, sqlite_path = root / "plan.json", root / "real_calls.sqlite"
    plan_bytes_sha = sha_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if strict_identity and (plan_bytes_sha != FROZEN_PLAN_SHA256 or plan.get("source_commit") != FROZEN_SOURCE):
        raise ValueError("Wrong original PR45 source or plan identity")
    db_sha = sha_file(sqlite_path)
    if strict_identity and db_sha != FROZEN_SQLITE_SHA256:
        raise ValueError("Wrong original PR45 SQLite identity")
    original_stat = sqlite_path.stat()
    with _read_only_db(sqlite_path) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        total = db.execute("SELECT count(*) FROM calls").fetchone()[0]
    if (meta.get("scope") != "REAL_AIRCRAFT_USER_INITIATED" or
            meta.get("plan_sha") != digest(plan)):
        raise ValueError("Ledger source and plan do not match")
    if strict_identity and total != EXPECTED_CALLS:
        raise ValueError("Missing original 151006 charged calls")
    dest.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(Counter)
    seeds = defaultdict(set)
    heads = defaultdict(Counter)
    combinations = Counter()
    reasons = Counter()
    issues = Counter()
    n = 0
    with _read_only_db(sqlite_path) as db, (dest / "risk_event_records.csv").open("x", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        sql = ("SELECT id,case_name,seed,role,arm,idx,candidate_hash,contract,state,record,record_hash "
               "FROM calls ORDER BY rowid")
        for cid, case, seed, role, arm, idx, candidate_hash, contract, state, raw, record_hash in db.execute(sql):
            if max_records is not None and n >= max_records:
                break
            if raw is None or state != "settled":
                raise ValueError("An original call has no trusted settled record")
            record = json.loads(raw)
            obs = record["observation"]
            if (record_hash != digest(record) or candidate_hash != digest(record["candidate"]) or
                    obs["design_id"] != candidate_hash or obs["contract_id"] != contract or
                    record["diagnostics"].get("confirm12_real_call", {}).get("call_id") != cid):
                raise ValueError("Untrusted original physical call/record identity")
            if case not in EXPECTED_BY_CASE or role not in ("initial", "online", "verification"):
                raise ValueError("Unregistered case/physical role")
            labels = label_record(record)
            name = arm if arm else "shared_initial"
            group = case, role, name
            _stat(grouped[group], labels)
            seeds[group].add(seed)
            for head, (kind, sign) in labels["constraint_heads"].items():
                h = heads[(*group, head)]
                h[kind] += 1
                h["sign_true" if sign is True else "sign_false" if sign is False else "sign_unknown"] += 1
            combinations[(*group, labels["status"], str(labels["mission_completed"]),
                          labels["objective_kind"], str(labels["physical_feasibility"]))] += 1
            reasons[(*group, str(labels["termination_reason"]))] += 1
            issues.update(labels["semantic_issues"])
            row = dict(call_id=cid, case=case, seed=seed, role=role, arm=name, idx=idx,
                       design_sha256=candidate_hash, **{k: labels[k] for k in FIELDS[7:]})
            writer.writerow({k: _csv_scalar(row[k]) for k in FIELDS})
            n += 1
            if n % 10000 == 0:
                print("read-only risk labels:", n, "/", total, flush=True)
    if max_records is None and strict_identity:
        by_case = Counter({case: sum(v["calls"] for (c, _, _), v in grouped.items() if c == case)
                           for case in EXPECTED_BY_CASE})
        if n != EXPECTED_CALLS or by_case != Counter(EXPECTED_BY_CASE):
            raise ValueError("Incomplete original task/role coverage")
    if sha_file(sqlite_path) != db_sha or sqlite_path.stat().st_mtime_ns != original_stat.st_mtime_ns:
        raise RuntimeError("Original SQLite identity or modification time changed")
    with (dest / "risk_semantics_by_group.csv").open("x", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        for (case, role, arm), c in sorted(grouped.items()):
            writer.writerow(dict(case=case, role=role, arm=arm, seed_groups=len(seeds[(case,role,arm)]), **_compact(c)))
    result = {
        "schema": SCHEMA, "status": "PARTIAL_DIAGNOSTIC" if max_records else "COMPLETE_READ_ONLY_LABEL_AUDIT",
        "data_scope": "PR45 previously evaluated genuine reduced-aircraft records; retrospective semantics",
        "original_source_sha": plan.get("source_commit"), "plan_sha256": plan_bytes_sha,
        "sqlite_sha256_unchanged": db_sha, "original_ledger_state_unchanged": meta.get("state"),
        "source_read_only": True, "new_aircraft_calls": 0, "new_rcaide_calls": 0,
        "scientifically_qualified": False, "source_identity_verified": bool(strict_identity),
        "calls_scanned": n, "original_calls": total, "partial_first_n_only": max_records is not None,
        "by_case_role_arm": {"/".join(k): {**_compact(v), "seed_groups": len(seeds[k])}
                             for k, v in sorted(grouped.items())},
        "constraint_evidence_by_group": {"/".join(k): dict(sorted(v.items())) for k,v in sorted(heads.items())},
        "status_mission_objective_constraint_cross_tab": [
            dict(case=k[0], role=k[1], arm=k[2], status=k[3], mission_completed=k[4],
                 objective_kind=k[5], physical_feasibility=k[6], calls=v)
            for k,v in sorted(combinations.items())],
        "termination_reasons": [dict(case=k[0], role=k[1], arm=k[2], termination_reason=k[3], calls=v)
                                for k,v in sorted(reasons.items())],
        "semantic_issue_counts": dict(sorted(issues.items())),
        "limitations": [
            "Status completed is not evidence of mission completion or final feasibility.",
            "Final physical feasibility can remain unknown after censored or missing constraints.",
            "No probability calibration, hidden-failure generalization or optimization benefit is established.",
            "Shared initial samples and method-selected online trajectories have different selection mechanisms.",
            "The 140 km task has not passed independent sparse-domain qualification.",
        ],
    }
    with (dest / "risk_semantics_report.json").open("x", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return result
