from copy import deepcopy
import unittest

from ra_lsbo.budget import digest
from ra_lsbo.risk_core.online_calibration import OnlineResidualCalibrator
from tools.run_risk_core_selection import (
    assert_linked_truth, clean, linked_truth_index, native_accept, restore_extended,
    study_metrics)


class RiskCoreStudyAuditTests(unittest.TestCase):
    def events(self):
        row = {"evaluation": 5, "candidate": {"x": .1}, "risk": {"objective": 10.}}
        tag = ["screen", "complete", 1, "mixed_eicb", 0]
        return [
            {"event": "started", "call_id": 1, "tag": tag, "candidate": row["candidate"]},
            {"event": "settled", "call_id": 1, "tag": tag,
             "truth_digest": digest(row), "row": row}]

    def test_changed_last_outcome_cannot_pass_truth_binding(self):
        events = self.events()
        index = linked_truth_index(events, 1)
        assert_linked_truth(index, events[1]["tag"], events[1]["row"])
        changed = deepcopy(events[1]["row"])
        changed["risk"]["objective"] = -999.
        with self.assertRaises(ValueError):
            assert_linked_truth(index, events[1]["tag"], changed)

    def test_unsettled_and_out_of_order_calls_are_rejected(self):
        events = self.events()
        with self.assertRaises(ValueError):
            linked_truth_index(events[:1], 1)
        with self.assertRaises(ValueError):
            linked_truth_index(list(reversed(events)), 1)

    def test_first_pending_infinite_calibration_checkpoint_can_restore(self):
        model = OnlineResidualCalibrator()
        model.predict(0., 1., "candidate-1")
        state = restore_extended(clean(model.state_dict()))
        restored = OnlineResidualCalibrator.from_state_dict(state)
        self.assertEqual(restored.state_dict(), model.state_dict())
        self.assertEqual(restored.observe("candidate-1", 1.)["error"], 0)

    def test_nan_evidence_is_not_silently_serialized_as_missing(self):
        with self.assertRaises(ValueError):
            clean({"score": float("nan")})

    def test_only_declared_native_policy_stop_is_caught(self):
        class Native:
            def accept(self, observed, metadata):
                raise RuntimeError("Repeated missing/failed objective returns; "
                                   "stopped before wasting remaining budget")
        self.assertTrue(native_accept(Native(), {}, {}))
        class Broken:
            def accept(self, observed, metadata):
                raise RuntimeError("Numerical model failure")
        with self.assertRaises(RuntimeError):
            native_accept(Broken(), {}, {})

    def test_absorbing_policy_stop_retains_incumbent_without_inventing_calls(self):
        def row(feasible, objective):
            return {"risk": {"feasible_label": feasible, "objective": objective,
                             "hidden_success": int(objective is not None)}}
        found = study_metrics([row(True, 10.)], [row(False, None)], 8, True)
        self.assertEqual(found["endpoint_best"], 10.)
        self.assertEqual(found["online_calls"], 1)
        self.assertFalse(found["complete_horizon"])
        absent = study_metrics([row(False, None)], [row(False, None)], 8, True)
        self.assertEqual(absent["restricted_first_feasible_cost"], 8)
        self.assertIsNone(absent["endpoint_best"])


if __name__ == "__main__":
    unittest.main()
