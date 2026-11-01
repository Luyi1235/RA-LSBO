"""Synthetic tests for the independent read-only D29 EP evidence audit."""
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from tools.audit_ep_d29_static import independent_matern52, independent_group_divisors, candidate_geometry


class D29StaticAuditTests(unittest.TestCase):
    def test_kernel_formula_and_symmetry(self):
        a = np.array([[0., 0.], [.25, .5]])
        K = independent_matern52(a, a)
        assert_allclose(K, K.T, atol=0, rtol=0)
        assert_allclose(np.diag(K), [1., 1.], atol=0, rtol=0)
        distance = math.sqrt(5)*math.hypot(.25, .5)
        self.assertAlmostEqual(K[0,1], (1+distance+distance**2/3)*math.exp(-distance),places=13)

    def test_divisors_are_grouped_as_declared(self):
        d = independent_group_divisors(16, 6, .05, 1.5)
        assert_allclose(d[:6], np.full(6, .05*math.sqrt(6)))
        assert_allclose(d[6:12], np.full(6, math.sqrt(6)))
        assert_allclose(d[12:], np.full(4, 3.))

    def test_geometry_exact_and_interval_conditional_covariance(self):
        rng = np.random.default_rng(2)
        x = rng.random((7,16))
        o = [{"kind": "exact", "value": v} for v in (-.2, -.1, -.05)] + [
            {"kind": "interval", "lower": v} for v in (.001, .002, .003, .004)]
        report = candidate_geometry(x,o,.05,.5,ncont=6)
        self.assertGreater(report["pre_EP_interval_cov_min_eigen"], 0.)
        self.assertGreater(report["pre_EP_interval_cov_cond"], 1.)
        self.assertTrue(math.isfinite(report["pre_EP_exact_kernel_cond"]))
        self.assertGreater(report["minimum_distinct_row_distance"], 0.)
        self.assertEqual(report["zero_distance_nonidentical_rows"],0)

    def test_no_response_or_evaluator_import(self):
        import inspect
        import tools.audit_ep_d29_static as audit
        source = inspect.getsource(audit)
        self.assertNotIn("problem.evaluate(",source)
        self.assertNotIn("optimizer.propose(",source)
        self.assertNotIn("MixedObservationGP(",source)


if __name__ == "__main__":
    unittest.main()
