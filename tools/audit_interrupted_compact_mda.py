"""Audit one interrupted historical MDA call without importing project code."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/compact_mda_20261021T081006Z_20eb608d"
PHASE = STUDY / "qualification_recovered_20261025"
OUTPUT = ROOT / "runs/compact_mda_interruption_audit_20261026"
WORKER_ID = "128c10d7e34d4a159aba376ab8062967"
CANDIDATE_KEY = "61f3bbb27e0cba1e1a2f4d126bd99766c7830b5f3217adc55841c15500451413"
CONTRACT = "a16f28b756f61e8f35c9f7e45394c15b0c33b350a3c40791a7c3d45866d24378"
HIDDEN_MODES = (
    "geometry_generation_failure", "network_configuration_failure",
    "motor_design_failure", "mission_solver_nonconvergence", "energy_audit_failure",
)
CONSTRAINTS = (
    "soc", "stall_speed", "flutter", "structure", "span_min", "span_max",
    "usable_energy", "prop_disk_clearance", "prop_fuselage_clearance",
    "prop_wing_clearance", "battery_current", "battery_voltage",
)
EXPECTED = {
    "plan.json": "087f587efcec17f966ffb9723821919f7fd52860a3e3a356933f893e5b381bdf",
    "submissions.jsonl": "b289acf2b6b6fe3565af602ea572ce8297fa497231d6417af5ed099d3c194d6d",
    "observations.jsonl": "dd5d7e6234d759c7053d8500203d3a90c54391a777b6514341407daa8974e28a",
    "task_65/budget.jsonl": "084e5e1bb9c6e7475051a9b76c0460adea1c83a9853736ea2eacf5d499a26301",
    "execution_console.log": "e9f1010a4c1a8f433cd0fcbc73af5a090abf6d422d753b5e7fbf5e3f48d26bee",
    "workers_65/" + WORKER_ID + "/request.json":
        "2c94d13d54581020fbbb158d16c1d376e9677a694a3b7a1e542b7d3674ddb535",
    "workers_65/" + WORKER_ID + "/solver.log":
        "1df2495fda94810fd175ba053c5e661fa18f47b3529be73382e4cb42dee368ab",
    "workers_65/" + WORKER_ID + "/battery_domain_trials.jsonl":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))


def load_json(data):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Nonstandard JSON constant: " + value)

    return json.loads(data, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path):
    require(not path.is_symlink() and not path.is_junction(), "Linked source: " + str(path))
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Source changed while reading: " + str(path))
    return {"path": str(path), "sha256": sha(data), "size_bytes": len(data),
            "mtime_ns": after.st_mtime_ns,
            "last_write_utc": datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat()}


def tree_manifest(root):
    entries = {}
    directories = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink() and not path.is_junction(), "Linked source: " + str(path))
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(relative)
        else:
            record = file_record(path)
            record.pop("path")
            entries[relative] = record
    return {"root": str(root), "file_count": len(entries), "directories": directories,
            "files": entries, "manifest_sha256": digest({"files": entries, "directories": directories})}


class Evidence:
    def __init__(self, manifest):
        self.manifest = manifest
        self.sources = {}

    def read(self, path, expected=None):
        path = path.resolve()
        record = file_record(path)
        data = path.read_bytes()
        require(sha(data) == record["sha256"], "Source changed: " + str(path))
        if path.is_relative_to(STUDY):
            saved = self.manifest["files"][path.relative_to(STUDY).as_posix()]
            require({key: value for key, value in record.items() if key != "path"} == saved,
                    "Source differs from initial manifest: " + str(path))
        if expected is not None:
            require(record["sha256"] == expected, "Pinned SHA-256 mismatch: " + str(path))
            record["expected_sha256"] = expected
        elif str(path) in self.sources and "expected_sha256" in self.sources[str(path)]:
            record["expected_sha256"] = self.sources[str(path)]["expected_sha256"]
        self.sources[str(path)] = record
        return data

    def json(self, path, expected=None):
        return load_json(self.read(path, expected))

    def rows(self, path):
        return [load_json(line) for line in self.read(path).splitlines()]


def replay_ledger(events):
    require(bool(events), "Empty ledger")
    tickets = {}
    previous = None
    for sequence, event in enumerate(events):
        unsigned = {key: value for key, value in event.items() if key != "hash"}
        require(event["sequence"] == sequence and event["previous"] == previous
                and event["hash"] == digest(unsigned), "Invalid ledger chain/sequence")
        kind, payload = event["kind"], event["payload"]
        if sequence == 0:
            require(kind == "policy", "Ledger must start with policy")
        elif kind == "reserve":
            token = payload["token"]
            require(token not in tickets and digest(payload["candidate"]) == payload["candidate_key"],
                    "Duplicate token or invalid candidate digest")
            require(payload["charge"] == 1.0, "Unexpected historical charge")
            tickets[token] = {**payload, "state": "reserved", "reserve_event": event}
        else:
            require(kind in ("start", "settle", "cancel"), "Unknown ledger transition")
            ticket = tickets[payload["token"]]
            expected_state = "started" if kind == "settle" else "reserved"
            require(ticket["state"] == expected_state, "Invalid ledger state transition")
            ticket["state"] = {"start": "started", "settle": "settled", "cancel": "cancelled"}[kind]
            ticket[kind + "_event"] = event
            if kind == "settle":
                ticket["risk"] = payload["risk"]
        previous = event["hash"]
    return tickets


def unknown_outcome():
    return {
        "outcome": "unknown", "physical_success": None, "solver_success": None,
        "feasible_label": None, "hidden_success": None,
        "hidden_labels": dict.fromkeys(HIDDEN_MODES), "objective": None,
        "objective_observed": False, "constraints": dict.fromkeys(CONSTRAINTS),
        "constraint_observed": dict.fromkeys(CONSTRAINTS, False),
        "mda_evaluated": None, "mda_completed": None, "elapsed_seconds": None,
        "original_call_remains_charged": True,
        "administrative_settlement_performed": False, "qualification_completed": False,
        "qualification_gate_satisfied": False,
    }


def log_evidence(path, lines, text):
    matches = [{"path": str(path), "line": index, "text": line}
               for index, line in enumerate(lines, 1) if text in line]
    require(len(matches) == 1, "Expected one log marker: " + text)
    return matches[0]


def audit():
    started = utc_now()
    manifest = tree_manifest(STUDY)
    evidence = Evidence(manifest)
    for relative, expected in EXPECTED.items():
        evidence.read(PHASE / relative, expected)
    plan = evidence.json(PHASE / "plan.json")
    require(plan["schema"] == "compact_snapshot_recovery_v1"
            and Path(plan["study"]).resolve() == STUDY
            and Path(plan["base_project"]).resolve() == PHASE / "frozen_base",
            "Unexpected plan identity")
    require(plan["per_task_calls"] == 30 and plan["phase_max_calls"] == 60
            and "no unknown outcomes" in plan["selection_rule"]
            and "no unknown outcomes" in plan["confirmation_rule"], "Qualification rules differ")
    static_sources = [ROOT / "ra_lsbo/budget.py", ROOT / "tools/recover_compact_qualification.py"]
    static_sources.extend(PHASE / "frozen_base/problems/x57" / name for name in (
        "adapter.py", "worker.py", "mission_solver.py", "domain_solver.py"))
    for path in static_sources:
        evidence.read(path, plan["source_sha256"][str(path)])
    submissions = evidence.rows(PHASE / "submissions.jsonl")
    observations = evidence.rows(PHASE / "observations.jsonl")
    require(len(submissions) == 37 and len(observations) == 36, "Phase counts changed")
    require([row["phase_call"] for row in submissions] == list(range(1, 38))
            and [row["evaluation"] for row in observations] == list(range(1, 37)),
            "Submission/observation sequence differs")
    require(set(PHASE.rglob("budget.jsonl")) == {
        PHASE / "task_55/budget.jsonl", PHASE / "task_65/budget.jsonl"}, "Unexpected phase ledger")
    require([task["distance_nmi"] for task in plan["tasks"]] == [55, 65], "Unexpected tasks")
    all_pending, summaries, worker_bindings = [], [], []
    for task in plan["tasks"]:
        distance = task["distance_nmi"]
        config_path = Path(task["config_path"]).resolve()
        config = evidence.json(config_path, plan["source_sha256"][str(config_path)])
        rows = [row for row in submissions if row["task"] == distance]
        observed = [row for row in observations if row["distance_nmi"] == distance]
        ledger_path = PHASE / ("task_" + str(distance)) / "budget.jsonl"
        events = evidence.rows(ledger_path)
        tickets = list(replay_ledger(events).values())
        require(len(rows) == len(tickets), "Submission/ledger count differs")
        settled = [ticket for ticket in tickets if ticket["state"] == "settled"]
        require(len(observed) == len(settled), "Observation/settlement count differs")
        workers = PHASE / ("workers_" + str(distance))
        requests = {}
        for directory in sorted(workers.iterdir()):
            require(directory.is_dir(), "Unexpected worker root entry")
            request = evidence.json(directory / "request.json")
            key = digest(request["candidate"])
            require(key not in requests and request["config"] == config, "Duplicate/mismatched worker request")
            requests[key] = directory
        require(len(requests) == len(tickets), "Unbound worker request")
        for index, (submission, ticket) in enumerate(zip(rows, tickets)):
            candidate = submission["proposal"]["candidate"]
            key = digest(candidate)
            require(candidate == task["candidates"][index] == ticket["candidate"]
                    and key == submission["candidate_key"] == ticket["candidate_key"],
                    "Plan/submission/reservation candidate binding differs")
            directory = requests[key]
            binding = {"distance_nmi": distance, "candidate_key": key,
                       "phase_call": submission["phase_call"], "token": ticket["token"],
                       "worker_dir": str(directory), "request_path": str(directory / "request.json")}
            if ticket["state"] == "started":
                all_pending.append({**binding, "ticket": ticket, "ledger_path": str(ledger_path)})
            else:
                require(ticket["state"] == "settled", "Unexpected reservation state")
                observation = observed[index]
                require(observation["candidate"] == candidate and observation["risk"] == ticket["risk"]
                        and observation["evaluation"] == submission["phase_call"]
                        and observation["evaluator_contract_sha256"] == task["contract_id"]
                        and Path(observation["risk"]["evidence"]["worker_dir"]).resolve() == directory,
                        "Observation/settlement/request binding differs")
                evidence.json(directory / "response.json")
            worker_bindings.append(binding)
        summaries.append({"distance_nmi": distance, "submissions": len(rows),
                          "charged_started_or_settled": sum(ticket["state"] in ("started", "settled")
                                                           for ticket in tickets),
                          "settled": len(settled), "observations": len(observed),
                          "unresolved_started": sum(ticket["state"] == "started" for ticket in tickets),
                          "ledger_path": str(ledger_path), "ledger_tail_hash": events[-1]["hash"],
                          "hash_chain_verified": True})
    require([(item["submissions"], item["settled"]) for item in summaries] == [(30, 30), (7, 6)],
            "Historical task counts differ")
    require(len(all_pending) == 1, "Expected exactly one pending call")
    pending = all_pending[0]
    ticket = pending.pop("ticket")
    worker = PHASE / "workers_65" / WORKER_ID
    require(pending["candidate_key"] == CANDIDATE_KEY and pending["phase_call"] == 37
            and pending["distance_nmi"] == 65 and pending["token"] == "call_19"
            and Path(pending["worker_dir"]) == worker
            and plan["tasks"][1]["contract_id"] == CONTRACT, "Pending identity differs")
    require(ticket["reserve_event"]["sequence"] == 19 and ticket["start_event"]["sequence"] == 20
            and ticket["start_event"]["hash"] ==
            "209db409cceba4b21ba52ed81ef5a81580fc0c852d2bbc92d440beafa1f1752e",
            "Pending start differs")
    pending.update({"candidate": ticket["candidate"], "contract_sha256": CONTRACT,
                    "original_ledger_state": "started", "charge": ticket["charge"],
                    "reserve_event": ticket["reserve_event"], "start_event": ticket["start_event"]})
    return finish_audit(started, manifest, evidence, plan, pending, summaries, worker_bindings, worker)


def finish_audit(started, manifest, evidence, plan, pending, summaries, bindings, worker):
    missing = [PHASE / "result.json"]
    missing.extend(worker / name for name in (
        "response.json", "response.tmp", "result.json", "solver_error.json",
        "energy_audit.json", "simplified_powered_lift.json"))
    require(all(not path.exists() for path in missing), "A possible terminal artifact now exists")
    top_files = {path.name for path in worker.iterdir() if path.is_file()}
    require(top_files == {"request.json", "solver.log", "battery_domain_trials.jsonl"},
            "Unexpected pending-worker artifact")
    airfoils = worker / "solver_work/Airfoils"
    copied = []
    for path in sorted(worker.rglob("*")):
        if not path.is_file() or path.parent == worker:
            continue
        require(path.is_relative_to(airfoils), "Unrecognized solver output: " + str(path))
        origin = PHASE / "frozen_base/problems/x57/model/Airfoils" / path.relative_to(airfoils)
        require(sha(evidence.read(path)) == sha(evidence.read(origin)), "Solver asset differs from input")
        copied.append({"path": str(path), "original_input_path": str(origin)})
    require(len(copied) == 83, "Copied solver input count changed")
    response_paths = sorted(str(path) for path in PHASE.rglob("response.json"))
    expected_responses = sorted(str(Path(item["worker_dir"]) / "response.json")
                                for item in bindings if item["phase_call"] != 37)
    require(response_paths == expected_responses, "Unaccounted response artifact")
    solver_path = worker / "solver.log"
    solver_lines = evidence.read(solver_path).decode("utf-8", errors="replace").splitlines()
    fallback = "[RCAIDE] DER fallback: evaluations=5, residual_inf=4.45397, rejected=0"
    require(solver_lines[-1] == fallback, "Last solver evidence changed")
    require(not any(marker in "\n".join(solver_lines) for marker in (
        "END procedure.missions.design_mission", "[RCAIDE] Post-processing:",
        "[RCAIDE] Candidate result:", "Traceback", "DomainSolveError")),
        "Potential terminal solver evidence found")
    require(evidence.read(worker / "battery_domain_trials.jsonl") == b"", "Trial log is no longer empty")
    console_path = PHASE / "execution_console.log"
    console_lines = evidence.read(console_path).decode("utf-8", errors="replace").splitlines()
    marker = "[RCAIDE] Solver log: " + str(solver_path)
    console_marker = log_evidence(console_path, console_lines, marker)
    console_tail = [line for line in console_lines[console_marker["line"]:] if line.strip()]
    heartbeat = "[RCAIDE] Waiting for worker response: 15s elapsed (timeout 1800s)"
    require(console_tail[-1] == heartbeat and console_tail[-2] == fallback,
            "Last pending console evidence changed")
    evidence.read(STUDY / "study_active.lock")
    evidence.read(Path(__file__).resolve())
    after = tree_manifest(STUDY)
    require(manifest == after, "Original study changed during audit; refusing to publish")
    for saved in evidence.sources.values():
        current = file_record(Path(saved["path"]))
        require(all(current[key] == saved[key] for key in current), "Evidence changed during audit")
    report = {
        "schema": "compact_mda_interruption_audit_v1",
        "record_type": "read_only_interruption_audit_not_a_settle_event",
        "audit_started_utc": started, "audited_utc": utc_now(),
        "study_path": str(STUDY), "phase_path": str(PHASE),
        **unknown_outcome(), "interruption_cause": None,
        "pending_call": pending, "phase_accounting": summaries,
        "phase_charged_calls": 37, "phase_settled_calls": 36,
        "unique_pending_candidate_binding_verified": True,
        "pinned_artifact_sha256_verified": True,
        "static_source_sha256_matches_plan": True,
        "original_study_unchanged_during_audit": True,
        "original_study_manifest": manifest,
        "post_audit_manifest_sha256": after["manifest_sha256"],
        "evidence_sources": [evidence.sources[key] for key in sorted(evidence.sources)],
        "worker_bindings": bindings,
        "terminal_artifact_search": {
            "recoverable_terminal_result_found": False,
            "can_settle_physical_outcome_from_existing_artifacts": False,
            "checked_absent_paths": [str(path) for path in missing],
            "pending_worker_files": sorted(str(path) for path in worker.rglob("*") if path.is_file()),
            "phase_response_paths": response_paths, "all_responses_bound_to_settled_calls": True,
            "solver_work_copied_input_files": copied,
            "serialized_mission_checkpoint_found": False,
        },
        "execution_evidence": {
            "mda_started_observed": True,
            "last_module": "procedure.missions.design_mission", "last_segment": "DER",
            "module_start": log_evidence(solver_path, solver_lines, "START procedure.missions.design_mission"),
            "last_solver_line": log_evidence(solver_path, solver_lines, fallback),
            "console_worker_binding": console_marker,
            "last_console_line": {"path": str(console_path), "line": len(console_lines), "text": heartbeat},
            "completed_prior_modules": ["procedure.simple_sizing", "procedure.weights"],
            "trial_log_size_bytes": 0, "observed_exit_code": None, "timeout_observed": False,
            "progress_is_not_a_terminal_convergence_or_failure_label": True,
            "rounded_weight_log_excluded_from_training_outputs": True,
        },
        "qualification_rules": {key: plan[key] for key in ("selection_rule", "confirmation_rule")},
        "process_check": {"performed": False, "liveness_at_audit": None},
        "limitations": [
            "File-only audit; no process liveness or termination cause is established.",
            "Last-write times and the 15-second heartbeat do not establish an interruption time or timeout.",
            "DER progress and an empty trial log establish neither physical failure nor success.",
            "No objective, constraints, or hidden failure labels are recovered for the pending call.",
            "This audit neither appends a ledger event nor completes qualification or clears its gate.",
            "The started call keeps its existing charge; no cancellation, refund, retry, or new MDA call.",
        ],
    }
    return report


def validate_report(report):
    for key, value in unknown_outcome().items():
        require(report[key] == value, "Unknown-outcome contract changed: " + key)
    require(report["record_type"] == "read_only_interruption_audit_not_a_settle_event"
            and report["original_study_unchanged_during_audit"] is True
            and report["unique_pending_candidate_binding_verified"] is True
            and report["terminal_artifact_search"]["recoverable_terminal_result_found"] is False,
            "Invalid audit status")


def self_test():
    import copy
    import unittest

    def append_event(events, kind, payload):
        event = {"sequence": len(events), "previous": events[-1]["hash"] if events else None,
                 "kind": kind, "payload": payload}
        events.append({**event, "hash": digest(event)})

    def sample_events(start=True):
        events = []
        append_event(events, "policy", {"policy": {"max_calls": 1}})
        candidate = {"continuous": {"test": 1.0}, "discrete": {}}
        append_event(events, "reserve", {"token": "call_1", "candidate": candidate,
                                         "candidate_key": digest(candidate), "charge": 1.0})
        if start:
            append_event(events, "start", {"token": "call_1"})
        return events

    class AuditTests(unittest.TestCase):
        def test_started_remains_charged_and_unsettled(self):
            ticket = replay_ledger(sample_events())["call_1"]
            self.assertEqual((ticket["state"], ticket["charge"]), ("started", 1.0))
            self.assertNotIn("risk", ticket)

        def test_settlement_is_replayed_without_modifying_input(self):
            events = sample_events()
            append_event(events, "settle", {"token": "call_1", "risk": {"state": "unknown"}})
            original = copy.deepcopy(events)
            self.assertEqual(replay_ledger(events)["call_1"]["state"], "settled")
            self.assertEqual(events, original)

        def test_payload_tampering_is_rejected(self):
            events = sample_events()
            events[1]["payload"]["candidate"]["continuous"]["test"] = 2.0
            with self.assertRaises(ValueError):
                replay_ledger(events)

        def test_rehashed_invalid_candidate_binding_is_rejected(self):
            events = sample_events(False)
            events[1]["payload"]["candidate_key"] = "wrong"
            events[1]["hash"] = digest({key: value for key, value in events[1].items() if key != "hash"})
            with self.assertRaises(ValueError):
                replay_ledger(events)

        def test_broken_previous_link_is_rejected(self):
            events = sample_events()
            events[-1]["previous"] = "wrong"
            events[-1]["hash"] = digest({key: value for key, value in events[-1].items() if key != "hash"})
            with self.assertRaises(ValueError):
                replay_ledger(events)

        def test_duplicate_start_and_unstarted_settlement_are_rejected(self):
            events = sample_events()
            append_event(events, "start", {"token": "call_1"})
            with self.assertRaises(ValueError):
                replay_ledger(events)
            events = sample_events(False)
            append_event(events, "settle", {"token": "call_1", "risk": {}})
            with self.assertRaises(ValueError):
                replay_ledger(events)

        def test_nonstandard_json_is_rejected(self):
            for text in ('{"key": 1, "key": 2}', '{"key": NaN}', '{"key": Infinity}'):
                with self.assertRaises(ValueError):
                    load_json(text)

        def test_physical_and_hidden_labels_remain_unknown(self):
            outcome = unknown_outcome()
            for key in ("physical_success", "solver_success", "feasible_label", "hidden_success", "objective"):
                self.assertIsNone(outcome[key])
            self.assertTrue(all(value is None for value in outcome["hidden_labels"].values()))
            self.assertTrue(all(value is None for value in outcome["constraints"].values()))
            self.assertTrue(outcome["original_call_remains_charged"])
            self.assertFalse(outcome["administrative_settlement_performed"])
            self.assertFalse(outcome["qualification_completed"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true", help="Run in-memory tests; write nothing")
    modes.add_argument("--verify", action="store_true", help="Recompute and verify existing audit; write nothing")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    destination = OUTPUT / "audit.json"
    if not args.verify:
        require(not OUTPUT.exists(), "Output directory already exists; use --verify")
    report = audit()
    validate_report(report)
    if args.verify:
        saved = load_json(destination.read_bytes())
        validate_report(saved)
        ignored = {"audit_started_utc", "audited_utc"}
        require({key: value for key, value in saved.items() if key not in ignored}
                == {key: value for key, value in report.items() if key not in ignored},
                "Saved audit differs from recomputed source evidence")
        action = "verified"
    else:
        require(not OUTPUT.resolve().is_relative_to(STUDY), "Output must be outside original study")
        OUTPUT.mkdir(exist_ok=False)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        action = "created"
    print(json.dumps({"action": action, "path": str(destination),
                      "sha256": sha(destination.read_bytes()), "outcome": report["outcome"],
                      "original_study_files_verified": report["original_study_manifest"]["file_count"],
                      "phase_charged_calls": 37, "phase_settled_calls": 36,
                      "administrative_settlement_performed": False, "qualification_completed": False}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print("Audit failed without changing the original study: " + str(error), file=sys.stderr)
        sys.exit(1)
