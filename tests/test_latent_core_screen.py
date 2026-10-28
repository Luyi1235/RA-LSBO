import tempfile
import unittest
from pathlib import Path

from tools.run_latent_core_screen import (
    ARMS, FAMILIES, HELDOUT_HORIZON, HELDOUT_SEEDS, INITIAL, MAXIMUM,
    SCREEN_HORIZON, SCREEN_SEEDS, arm_parts, charge, lines, select_family)


def summaries():
    return [{"seed": seed, "method": family+suffix, "endpoint_best": None,
             "restricted_first_feasible_cost": 12, "hidden_failure_count": 6}
            for seed in SCREEN_SEEDS for family in FAMILIES
            for suffix in ("__onehot", "__latent")]


class LatentCoreScreenTests(unittest.TestCase):
    def test_budget_covers_complete_screen_and_confirmation(self):
        self.assertEqual(len(SCREEN_SEEDS)*(INITIAL+len(ARMS)*SCREEN_HORIZON)
                         +len(HELDOUT_SEEDS)*(INITIAL+3*HELDOUT_HORIZON), MAXIMUM)
        self.assertEqual(MAXIMUM, 376)
        self.assertFalse(set(SCREEN_SEEDS) & set(HELDOUT_SEEDS))

    def test_every_family_has_exactly_two_representation_arms(self):
        for family in FAMILIES:
            self.assertEqual(arm_parts(family+"__onehot"), (family, False))
            self.assertEqual(arm_parts(family+"__latent"), (family, True))
        self.assertEqual(len(ARMS), 7)
        with self.assertRaises(ValueError):
            arm_parts("mixed_eicb__stages")

    def test_selection_uses_latent_results_before_bare_core_results(self):
        rows = summaries()
        for row in rows:
            if row["method"] == "ensemble_boundary__onehot":
                row["endpoint_best"] = 1.
            if row["method"] == "mixed_eicb__latent":
                row["endpoint_best"] = 2.
        self.assertEqual(select_family(rows)["selected_family"], "mixed_eicb")

    def test_selection_uses_own_latent_gain_as_first_tie_breaker(self):
        rows = summaries()
        for row in rows:
            row["endpoint_best"] = 3.
            if row["method"] == "calibrated_eicb__onehot":
                row["endpoint_best"] = 4.
        self.assertEqual(select_family(rows)["selected_family"], "calibrated_eicb")

    def test_all_ties_use_frozen_parsimony(self):
        self.assertEqual(select_family(summaries())["selected_family"], "mixed_eicb")

    def test_unresolved_call_cannot_be_retried(self):
        class MissingTruth:
            contract_id = "test"

            def evaluate(self, candidate):
                return {"feasible_label": None}

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(RuntimeError):
                charge(folder, MissingTruth(), {}, ["screen", 1, "test", 0], 1)
            with self.assertRaises(RuntimeError):
                charge(folder, MissingTruth(), {}, ["screen", 1, "test", 1], 2)
            self.assertEqual(len(lines(folder/"calls.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
