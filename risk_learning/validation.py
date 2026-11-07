"""Only recomputed report metrics tolerate floating-point reduction roundoff.

Raw files, manifests, IDs, labels, integer counts and model contracts remain
exact. Python versions may implement sum() with different rounding behavior.
"""
import math


def metric_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return (actual.keys() == expected.keys()
                and all(metric_equal(actual[k], expected[k]) for k in actual))
    if isinstance(actual, list):
        return (len(actual) == len(expected)
                and all(metric_equal(a, b) for a, b in zip(actual, expected)))
    if isinstance(actual, float):
        return (math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14))
    return actual == expected
