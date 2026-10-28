"""Mathematical cases exposed by this independently runnable experiment package."""

from ..core.problems import MathematicalCase

CASES = (
    "branin_ordinary",
    "branin_sparse",
    "rosenbrock_ordinary",
    "rosenbrock_sparse",
)


def make_problem(base, case):
    if case not in CASES:
        raise ValueError(f"Unknown mathematical benchmark: {case}")
    family, difficulty = case.split("_")
    return MathematicalCase(base, family, difficulty)
