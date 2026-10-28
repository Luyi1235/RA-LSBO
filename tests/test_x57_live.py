import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ra_lsbo.x57_live import execute_smoke

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    from ra_lsbo.synthetic import SyntheticProblem
    from ra_lsbo.x57_live import X57LiveProblem, prepare_smoke
    from test_models import BASE


class AuthorizationTests(unittest.TestCase):
    def test_unapproved_request_never_loads_physical_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps({
                "schema": "ra_x57_single_call_request_v1", "operator_approved": False,
                "max_mda_calls": 1, "max_risk_charge": 1.,
            }))
            with patch("ra_lsbo.x57_live.X57LiveProblem") as adapter:
                with self.assertRaisesRegex(ValueError, "approved"):
                    execute_smoke(path)
                adapter.assert_not_called()

    def test_running_experiment_blocks_even_approved_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text('{"status":"running"}')
            path = root / "request.json"
            path.write_text(json.dumps({
                "schema": "ra_x57_single_call_request_v1", "operator_approved": True,
                "max_mda_calls": 1, "max_risk_charge": 1.,
                "active_run_to_wait_for": str(root),
            }))
            with patch("ra_lsbo.x57_live.X57LiveProblem") as adapter:
                with self.assertRaisesRegex(RuntimeError, "not terminal"):
                    execute_smoke(path)
                adapter.assert_not_called()


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class LiveBoundaryTests(unittest.TestCase):
    def test_prepare_uses_only_cheap_physics_and_evaluate_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = prepare_smoke(BASE, BASE / "configs" / "x57.json", Path(directory) / "request")
            request = json.loads(path.read_text())
            self.assertFalse(request["operator_approved"])
            self.assertEqual(request["max_mda_calls"], 1)
            self.assertNotIn("problems.x57.runtime", sys.modules)
            self.assertFalse((path.parent / "budget.jsonl").exists())
            problem = X57LiveProblem(BASE, BASE / "configs" / "x57.json", Path(directory) / "workers")
            with self.assertRaisesRegex(RuntimeError, "not explicitly"):
                problem.evaluate(request["candidate"])

    def test_mock_physical_smoke_is_exactly_once_and_budgeted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = SyntheticProblem(BASE)
            request = {
                "schema": "ra_x57_single_call_request_v1", "operator_approved": True,
                "max_mda_calls": 1, "max_risk_charge": 1.,
                "base_project": str(BASE), "config_path": "unused_by_mock",
                "active_run_to_wait_for": None, "contract_id": problem.contract_id,
                "candidate": problem.space.default().to_dict(),
            }
            path = root / "request.json"
            path.write_text(json.dumps(request))
            with patch("ra_lsbo.x57_live.X57LiveProblem", return_value=problem):
                result = execute_smoke(path)
                self.assertEqual(result["ledger"]["charged_calls"], 1)
                self.assertEqual(result["ledger"]["risk_charge"], 1.)
                with self.assertRaises(FileExistsError):
                    execute_smoke(path)

    def test_changed_approved_contract_prevents_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = SyntheticProblem(BASE)
            path = root / "request.json"
            path.write_text(json.dumps({
                "schema": "ra_x57_single_call_request_v1", "operator_approved": True,
                "max_mda_calls": 1, "max_risk_charge": 1.,
                "base_project": str(BASE), "config_path": "unused_by_mock",
                "active_run_to_wait_for": None, "contract_id": "different",
                "candidate": problem.space.default().to_dict(),
            }))
            with patch("ra_lsbo.x57_live.X57LiveProblem", return_value=problem):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    execute_smoke(path)
                self.assertFalse((root / "budget.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
