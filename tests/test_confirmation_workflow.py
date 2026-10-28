import unittest

from tools.advance_compact_confirmation import confirmation_distance


def qualification(first, second, selected):
    return {"status": "complete", "phase_calls": 60,
            "scores": [{"distance_nmi": 55, "count": 30, "feasible": first},
                       {"distance_nmi": 65, "count": 30, "feasible": second}],
            "selected_distance_nmi": selected}


class ConfirmationWorkflowTests(unittest.TestCase):
    def test_shorter_eligible_task_is_required(self):
        self.assertEqual(confirmation_distance(qualification(2, 1, 55)), 55)
        self.assertEqual(confirmation_distance(qualification(0, 1, 65)), 65)
        with self.assertRaises(ValueError):
            confirmation_distance(qualification(2, 1, 65))

    def test_unknown_and_no_eligible_task_cannot_advance(self):
        result = qualification(1, 1, 55)
        result["status"] = "stopped_unknown"
        self.assertIsNone(confirmation_distance(result))
        self.assertIsNone(confirmation_distance(qualification(0, 0, None)))
        self.assertIsNone(confirmation_distance(qualification(8, 9, None)))

    def test_partial_phase_is_not_a_pass(self):
        result = qualification(1, 1, 55)
        result["phase_calls"] = 20
        self.assertIsNone(confirmation_distance(result))


if __name__ == "__main__":
    unittest.main()
