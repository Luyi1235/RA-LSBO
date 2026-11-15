import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.legacy import normalize_evaluation
from run_x57_feedback_audit import audit, feedback_from_settlement


def evaluation(success):
    raw = {"status": "ok" if success else "mission_not_converged", "solver_success": success,
           "objective": 42000. if success else None, "constraints": {"span": -.1, "soc": .2 if not success else -.02},
           "metrics": {"mda_evaluated": True}}
    if not success:
        raw["constraint_observed"] = {"span": True}
        raw["metrics"].update(solver_discharge_margin_W=-400.,
                              solver_discharge_margin_source="inadmissible_seed_root_not_a_valid_mission")
    return normalize_evaluation(raw, ("span", "soc"), ("span",)).to_dict()


class X57FeedbackAuditTests(unittest.TestCase):
    def test_failed_solver_diagnostic_is_not_an_engineering_sign(self):
        row = feedback_from_settlement(evaluation(False))
        self.assertIs(row["completed_with_exact_objective"], False)
        self.assertEqual(row["constraints"]["soc"], {"kind": "missing", "sign": None, "analytical": False})
        self.assertEqual(row["constraints"]["span"], {"kind": "exact", "sign": True, "analytical": True})
        self.assertEqual(row["discharge_diagnostic"]["margin_W"], -400.)

        risk = evaluation(False)
        risk["evidence"]["verified_constraint_bounds"] = {
            "soc": {"kind": "interval", "evidence_id": "test", "lower": 0., "lower_inclusive": False}}
        row = feedback_from_settlement(risk)
        self.assertIs(row["constraints"]["soc"]["sign"], False)

    def test_replays_both_tasks_and_keeps_the_unresolved_call_unknown(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, submissions, observations = [], [], []
            for distance, size in ((55, 30), (65, 7)):
                contract = f"contract-{distance}"
                candidates = [{"continuous": {"wing_area": 9.+i/10}, "discrete": {}} for i in range(size)]
                tasks.append({"distance_nmi": distance, "contract_id": contract, "candidates": candidates})
                path = root/f"task_{distance}"/"budget.jsonl"
                with RiskLedger(path, BudgetPolicy(30, 30., 30, 0, 0)) as ledger:
                    for i, candidate in enumerate(candidates):
                        call = len(submissions)+1
                        submissions.append({"phase_call": call, "task": distance, "candidate_key": digest(candidate),
                                            "proposal": {"candidate": candidate}})
                        token = ledger.reserve(candidate, "bootstrap", {"basis": "unavailable"}, "test")
                        ledger.start(token)
                        if call == 37:
                            continue
                        risk = evaluation(success=i % 2 == 0)
                        ledger.settle(token, risk)
                        observations.append({"evaluation": call, "distance_nmi": distance, "risk": risk,
                                             "candidate": candidate, "evaluator_contract_sha256": contract})
            data = {"plan.json": json.dumps({"schema": "compact_snapshot_recovery_v1", "tasks": tasks}),
                    "submissions.jsonl": "\n".join(map(json.dumps, submissions)),
                    "observations.jsonl": "\n".join(map(json.dumps, observations))}
            for name, text in data.items():
                (root/name).write_bytes(text.encode())
            hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in (*data, "task_65/budget.jsonl")}
            with patch("run_x57_feedback_audit.ARCHIVE_HASHES", hashes):
                report = audit(root)
                self.assertEqual(report["new_rcaide_mda_calls"], 0)
                self.assertEqual(len(report["calls"]), 37)
                self.assertEqual(report["calls"][-1]["state"], "unresolved_started")
                self.assertIsNone(report["calls"][-1]["usable"])
                self.assertEqual(report["cases"][0]["hidden_failure_modes"], {"mission_solver_nonconvergence": 15})
                self.assertEqual(report["cases"][0]["hidden_failure_with_any_nonanalytical_sign"], 0)
                with (root/"task_55/budget.jsonl").open("ab") as stream:
                    stream.write(b"{}\n")
                with self.assertRaises((KeyError, ValueError)):
                    audit(root)
