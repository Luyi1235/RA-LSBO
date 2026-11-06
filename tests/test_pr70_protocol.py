"""Stdlib-only protocol, transition and receipt regression tests."""
from copy import deepcopy
import unittest
from pr70_protocol import plan, ARMS, check_transition, check_receipt
from pr69_evidence_verify import native_digest as digest


class ProtocolTests(unittest.TestCase):
    def test_budgets(self):
        for name, n, initial, online in (("ci",18,12,6),("smoke",72,48,24),("full",480,192,288)):
            p = plan(name)
            self.assertEqual((p["total_calls"],p["initial_calls"],p["online_calls"]),(n,initial,online))
            self.assertEqual(p["aircraft_calls"],0)

    def test_seed_sets_disjoint(self):
        c,s,f = [set(plan(p)["seeds"]) for p in ("ci","smoke","full")]
        self.assertFalse(c&s or c&f or s&f)
        self.assertFalse((c|s|f)&set(range(998801,998805)))

    def test_three_registered_arms(self):
        self.assertEqual(ARMS,("cei_myopic","cei_two_step","cei_two_step_next_pool"))

    def test_nodes_unchanged(self):
        self.assertEqual(plan("full")["fantasies_per_region_branch"],4)
        self.assertEqual(plan("full")["first_action_shortlist"],4)

    def test_bad_profile(self):
        with self.assertRaises(ValueError): plan("confirm")

    def support(self):
        raw=[{"x":1},{"x":2}]; model=[raw[1]]
        return {"iteration":1,"raw_pool_sha256":digest(raw),"model_pool_sha256":digest(model),"model_candidates":model},raw,model

    def test_exact_transition(self):
        s,r,m=self.support()
        self.assertTrue(check_transition(s,r,m,1))

    def test_missing_receipt_not_fabricated(self):
        self.assertFalse(check_transition(None,[],[],1))

    def test_wrong_raw_pool_fails(self):
        s,r,m=self.support()
        with self.assertRaises(ValueError): check_transition(s,list(reversed(r)),m,1)

    def test_wrong_model_pool_fails(self):
        s,r,m=self.support()
        with self.assertRaises(ValueError): check_transition(s,r,r,1)

    def test_wrong_iteration_fails(self):
        s,r,m=self.support()
        with self.assertRaises(ValueError): check_transition(s,r,m,2)

    def test_transition_does_not_mutate(self):
        s,r,m=self.support(); before=deepcopy((s,r,m))
        check_transition(s,r,m,1)
        self.assertEqual((s,r,m),before)

    def test_unexplained_missing_forecast_fails(self):
        with self.assertRaises(ValueError): check_receipt({}, {"x":1},0,2,ARMS[2])

    def test_explicit_model_coverage_has_no_probability(self):
        d=check_receipt({"coverage_reason":"model_numerical_failure","mode":"model_failure_coverage"},{"x":1},0,2,ARMS[2])
        self.assertFalse(d["forecast_available"])
        self.assertIsNone(d["predicted_failure"])


if __name__ == "__main__": unittest.main()
