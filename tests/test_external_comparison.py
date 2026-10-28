from ra_lsbo.budget import digest
from ra_lsbo.experiments.comparison import ARMS
from ra_lsbo.experiments.comparison.methods import (
    ExternalOptimizer, _features, optimizer_for)
from ra_lsbo.experiments.core import protocol
from ra_lsbo.experiments.core.problems import MathematicalCase
from ra_lsbo.partial_bo.problems import candidate_pool


BASE = protocol.BASE


def _history(problem):
    candidates, _ = candidate_pool(problem, 91021, 8)
    records = []
    for index, candidate in enumerate(candidates):
        region = (candidate["continuous"]["x"]-.68)**2 + (candidate["continuous"]["y"]-.25)**2-.1
        records.append(problem.record(candidate, 10.+index, {
            "region": region, "resource": -.2 if candidate["discrete"]["architecture"] == "basic" else
            candidate["continuous"]["trim"]-.4},
            {"termination_reason": "comparison_test"}))
    return records


def test_method_roster_and_feature_encoding_are_stable():
    problem = MathematicalCase(BASE)
    candidates, _ = candidate_pool(problem, 91022, 5)
    x = _features(problem, candidates)
    assert x.shape[0] == 5
    assert x.ndim == 2 and x.shape[1] > 5
    assert ARMS == ("ra_lsbo", "random", "logcei", "scbo_adapted",
                    "sego_utb_adapted", "nsga2_adapted")


def test_external_methods_make_canonical_proposals_without_evaluation(monkeypatch):
    problem = MathematicalCase(BASE)
    history = _history(problem)
    pool, _ = candidate_pool(problem, 91023, 32, {row.observation.design_id for row in history})
    profile = {"name": "smoke", "horizon": 2, "pool": 32, "encoder_epochs": 3}
    for method in ARMS:
        optimizer = optimizer_for(problem, BASE, 17, method, profile)
        monkeypatch.setattr(problem, "evaluate", lambda candidate: (_ for _ in ()).throw(
            AssertionError("proposal logic must not evaluate candidates")))
        candidate, metadata = optimizer.propose(history, pool, 0)
        assert candidate == problem.canonical(candidate)
        assert all(value <= 0 for value in problem.precheck(candidate).values())
        assert digest(candidate) not in {row.observation.design_id for row in history}
        assert metadata["method"] == method
        feedback = problem.record(candidate, 9., {"region": -.1, "resource": -.1},
                                  {"termination_reason": "synthetic_test_feedback"})
        optimizer.accept(feedback)
        monkeypatch.undo()


def test_random_search_is_seed_reproducible():
    problem = MathematicalCase(BASE)
    history = _history(problem)
    pool, _ = candidate_pool(problem, 91024, 24, {row.observation.design_id for row in history})
    profile = {"horizon": 2, "pool": 24}
    first = ExternalOptimizer(problem, 42, "random", profile)
    second = ExternalOptimizer(problem, 42, "random", profile)
    assert first.propose(history, pool, 0)[0] == second.propose(history, pool, 0)[0]

