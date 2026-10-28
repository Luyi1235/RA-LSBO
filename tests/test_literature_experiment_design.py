import unittest

from tools.prepare_literature_experiments import (
    ABLATION_ARMS, CASES, MAIN_ARMS, build_proposal, comparison_budget)


class LiteratureExperimentDesignTests(unittest.TestCase):
    def test_shared_initial_labels_are_charged_once(self):
        budget = comparison_budget([11], seeds=16, arms=8, online_calls=50)
        self.assertEqual(budget["shared_initial_labels"], 880)
        self.assertEqual(budget["online_labels"], 6400)
        self.assertEqual(budget["total_labels"], 7280)

    def test_new_primary_cases_include_hierarchy_and_negative_control(self):
        names = {case.upstream_class for case in CASES}
        self.assertEqual(len(names), 5)
        self.assertNotIn("FiveModeProblem", names)
        self.assertNotIn("SparseInteriorFiveModeProblem", names)
        self.assertEqual(sum(not case.has_hierarchy for case in CASES), 1)
        self.assertEqual(sum(case.has_hidden_constraints for case in CASES), 4)

    def test_complete_budget_includes_qualification_and_independent_queries(self):
        budget = build_proposal()["budget_estimate"]
        self.assertEqual(budget["public_controlled_labels"]["total_labels"], 35760)
        self.assertEqual(budget["engineering_comparison_mda"]["total_labels"], 4400)
        self.assertEqual(budget["total_physical_mda"], 4592)
        self.assertEqual(budget["total_labels_all_tasks"], 40352)
        self.assertTrue(budget["requires_additional_budget_authorization"])
        self.assertFalse(budget["execution_authorized_by_this_proposal"])
        self.assertEqual(budget["current_task_evaluations"], 0)

    def test_public_and_engineering_seeds_and_arm_names_are_distinct(self):
        proposal = build_proposal()
        public = set(proposal["public_design"]["paired_seeds"])
        engineering = set(proposal["engineering_case"]["paired_seeds"])
        self.assertFalse(public & engineering)
        self.assertEqual(len(public), 16)
        self.assertEqual(len(engineering), 16)
        self.assertEqual(len(set(MAIN_ARMS+ABLATION_ARMS)), 8)

    def test_budget_rejects_invalid_sizes(self):
        for dimensions, seeds, arms, online in (([], 16, 8, 50), ([0], 16, 8, 50),
                                                ([11], 0, 8, 50), ([11], 16, True, 50)):
            with self.assertRaises(ValueError):
                comparison_budget(dimensions, seeds, arms, online)


if __name__ == "__main__":
    unittest.main()
