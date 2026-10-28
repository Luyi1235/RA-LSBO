"""Regression tests for evidence semantics, solver guards and complete audits."""

from dataclasses import replace
from types import SimpleNamespace
import json
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from ra_lsbo.budget import digest
from ra_lsbo.experiments.comparison import methods, protocol, runner
from ra_lsbo.experiments.core import protocol as core_protocol
from ra_lsbo.experiments.core.problems import MathematicalCase
from ra_lsbo.numerics import SLSQP_CLIP_MESSAGE
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.soc_revision import optimizer as local_module

BASE = core_protocol.BASE


def sample_records():
    problem = MathematicalCase(BASE)
    pool, _ = candidate_pool(problem, 12341028, 8)
    feasible = problem.record(pool[0], 5., {"region": -.1, "resource": -.2}, {})
    sign = problem.record(pool[1], OutputObservation("missing", "obj", reason="early_stop"),
        {"region": OutputObservation("interval", "sign", lower=0., lower_inclusive=False),
         "resource": -.2}, {})
    unknown = problem.record(pool[2], OutputObservation("missing", "unknown", reason="not_observed"),
        {"region": OutputObservation("missing", "region", reason="not_observed"), "resource": -.2}, {})
    return problem, pool, feasible, sign, unknown


def test_sign_violation_and_unknown_are_not_ranked_as_feasible():
    _, _, feasible, sign, unknown = sample_records()
    assert sign.observation.constraint("region").constraint_satisfaction() is False
    assert methods._constraint_key(feasible)[0] == 0
    assert methods._constraint_key(sign)[0:2] == (1, 1)
    assert methods._constraint_key(unknown)[0] == 2
    assert methods._constraint_key(feasible) < methods._constraint_key(sign) < methods._constraint_key(unknown)


def test_nsga_never_fills_missing_objectives_with_a_large_penalty():
    problem, _, feasible, sign, unknown = sample_records()
    huge = replace(feasible, observation=replace(feasible.observation,
        objective=OutputObservation("exact", "huge", value=1e16)))
    optimizer = methods.ExternalOptimizer(problem, 17, "nsga2_adapted", {"horizon": 2, "pool": 8})
    order = optimizer._nsga2([sign, unknown, huge], [])
    assert order["crowding_order"][0] == 2
    assert sign.observation.objective.kind == "missing"


def test_objective_gp_uses_observed_infeasible_exact_objectives():
    problem, pool, feasible, _, _ = sample_records()
    infeasible = problem.record(pool[1], 9., {"region": -.1, "resource": .1}, {})
    assert not infeasible.usable_feasible
    model = methods._regressor(problem, [feasible, infeasible], "__objective__")
    assert model is not None and len(model.X_train_) == 2


@pytest.mark.parametrize("status", ["unknown", "infrastructure_error"])
def test_untrusted_outputs_do_not_train_baseline_heads(status):
    problem, _, feasible, _, _ = sample_records()
    untrusted = replace(feasible, observation=replace(feasible.observation, status=status))
    assert methods._classifier(problem, [untrusted], "region") == (None, .5)
    assert methods._regressor(problem, [untrusted, untrusted], "region") is None
    assert methods._regressor(problem, [untrusted, untrusted], "__objective__") is None


def test_scbo_uses_initial_incumbent_and_does_not_regress(monkeypatch):
    problem, pool, feasible, _, _ = sample_records()
    optimizer = methods.ExternalOptimizer(problem, 17, "scbo_adapted", {"horizon": 4, "pool": 8})
    monkeypatch.setattr(optimizer, "_scbo", lambda *args: (np.array([0]), np.array([0.]), {}))
    records = [feasible]
    for iteration, value in enumerate([10., 8., 4.]):
        candidate, _ = optimizer.propose(records, [pool[iteration+1]], iteration)
        record = problem.record(candidate, value, {"region": -.1, "resource": -.2}, {})
        optimizer.accept(record)
        records.append(record)
        if iteration < 2:
            assert optimizer.best_before == 5.
            assert optimizer.successes == 0
            assert optimizer.failures == iteration+1
    assert optimizer.best_before == 4. and optimizer.successes == 1


@pytest.mark.parametrize("eligible", [True, False])
def test_utb_eligibility_is_not_overridden_by_a_fixed_penalty(monkeypatch, eligible):
    problem, pool, feasible, _, _ = sample_records()
    optimizer = methods.ExternalOptimizer(problem, 17, "sego_utb_adapted", {"horizon": 2, "pool": 8})
    monkeypatch.setattr(methods, "_regressor", lambda problem, records, output: output)
    monkeypatch.setattr(methods, "_joint_probability", lambda *args: np.array([.4, .9]))
    def predict(model, problem, pool, fallback=0.):
        return (np.array([2000., 0.]), np.ones(2)) if model == "__objective__" else (
            np.array([-3., 3.]) if eligible else np.array([1., 3.]), np.ones(2))
    monkeypatch.setattr(methods, "_predict", predict)
    score, metadata = optimizer._sego_utb([feasible], pool[:2])
    assert np.argmax(score) == 0
    assert np.isneginf(score[1])
    assert metadata["phase"] == ("utb_feasible_set" if eligible else "utb_support_restoration")


@pytest.mark.parametrize("endpoint_kind", ["roundoff", "nan", "material", "iteration_limit"])
def test_local_solver_endpoints_are_guarded_and_diagnostics_retained(monkeypatch, endpoint_kind):
    problem = MathematicalCase(BASE)
    initial = problem.space.default().to_dict()
    settings = local_module.RevisionSettings(local_branches=1, local_starts=1, local_points=1)
    optimizer = local_module.RiskRefiningBO(problem, BASE, 7, settings)
    posterior = SimpleNamespace(best=1., predict=lambda candidates: candidates)
    monkeypatch.setattr(local_module, "pointwise_acquisition", lambda candidates, *args: (
        np.array([-sum(v*v for v in candidate["continuous"].values()) for candidate in candidates]),
        np.ones((len(candidates), 1))))
    monkeypatch.setattr(problem, "evaluate", lambda *args: pytest.fail("Local search must not evaluate the model"))
    def solver(fun, x0, **kwargs):
        fun(x0)
        warnings.warn(SLSQP_CLIP_MESSAGE, RuntimeWarning)
        if endpoint_kind == "roundoff":
            x = np.ones_like(x0)+1e-14
            fun(x)
            kwargs["constraints"]["fun"](x)
        elif endpoint_kind == "nan":
            x = np.full_like(x0, np.nan)
        elif endpoint_kind == "material":
            x = np.ones_like(x0)+.1
        else:
            x = x0
        return OptimizeResult(x=x, success=endpoint_kind != "iteration_limit",
                              status=9 if endpoint_kind == "iteration_limit" else 0,
                              message="test solver", nfev=2)
    monkeypatch.setattr(local_module, "minimize", solver)
    pool, info = optimizer._augment(posterior, [], [initial], {
        "mode": "feasible_discovery", "fallback": None, "eligible": [True], "kappa": 1.})
    assert pool[0] == initial
    assert all(np.isfinite(list(raw["continuous"].values())).all() for raw in pool)
    report = info["local_reports"][0]
    assert report["slsqp_clipping_warnings"] == 1
    if endpoint_kind in {"nan", "material"}:
        assert report["endpoint"] is None and report["local_error"]
        assert report["success"] is False
    elif endpoint_kind == "iteration_limit":
        assert report["success"] is False and report["status"] == 9
    else:
        assert report["success"] and report["bound_projections"] > 0
    json.dumps(info, allow_nan=False)


def test_confirm_is_rejected_before_building_an_unimplemented_plan(monkeypatch):
    monkeypatch.setattr(protocol, "build_core_plan", lambda *args: pytest.fail("Unsupported profile must fail before planning"))
    with pytest.raises(ValueError, match="only random"):
        protocol.build_plan("confirm", "mathematical")


@pytest.fixture
def toy_campaign(tmp_path, monkeypatch):
    # Synthetic evidence fixture, not an optimization experiment or MDA call.
    profile = {**core_protocol.PROFILES["smoke"], "cases": ("branin_sparse",),
               "seeds": (9601,), "initial": 2, "horizon": 2, "pool": 8, "panel": 2}
    monkeypatch.setitem(core_protocol.PROFILES, "smoke", profile)
    def evaluate(problem, raw):
        return problem.record(raw, sum(raw["continuous"].values()),
                              {"region": -.1, "resource": -.2}, {"termination_reason": "synthetic_fixture"})
    monkeypatch.setattr(MathematicalCase, "evaluate", evaluate)
    class PoolFirst:
        def __init__(self, method):
            self.method = method
        def propose(self, history, pool, iteration):
            return pool[0], {"method": self.method}
        def accept(self, record):
            pass
    monkeypatch.setattr(runner, "optimizer_for", lambda problem, base, seed, method, profile: PoolFirst(method))
    folder = tmp_path/"campaign"
    plan = runner.prepare(folder, "smoke", "mathematical")
    result = runner.run(folder, plan["budget"]["maximum_model_calls"])
    assert result["charged_model_calls"] == 18
    return folder


def overwrite_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_full_audit_is_idempotent_and_report_reaudits(toy_campaign, monkeypatch):
    folder = toy_campaign
    first = runner.audit(folder)
    assert first["settled_calls"] == 18 and first["checked_proposals"] == 12
    assert first["algorithm_replay"] is False
    assert runner.audit(folder) == first
    from ra_lsbo.experiments.comparison import visualization
    monkeypatch.setattr(visualization, "make_figures", lambda *args: [])
    assert runner.report(folder).is_file()


def test_missing_last_proposal_fails_even_with_a_summary(toy_campaign):
    path = toy_campaign/"optimization/branin_sparse/9601_ra_lsbo/proposals.jsonl"
    path.write_text(path.read_text().splitlines()[0]+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="horizon"):
        runner.audit(toy_campaign)


def test_qualification_statistics_are_reconstructed(toy_campaign):
    path = toy_campaign/"qualification/branin_sparse.json"
    content = runner.read(path)
    content["summary"]["n"] += 1
    overwrite_json(path, content)
    with pytest.raises(ValueError, match="Qualification"):
        runner.audit(toy_campaign)


def test_summary_trial_identity_is_checked(toy_campaign):
    path = toy_campaign/"optimization/branin_sparse/9601_ra_lsbo/summary.json"
    content = runner.read(path)
    content["seed"] = 9999
    overwrite_json(path, content)
    with pytest.raises(ValueError, match="Trial metrics"):
        runner.audit(toy_campaign)


def test_report_rejects_stale_seal_before_creating_figures(toy_campaign, monkeypatch):
    runner.audit(toy_campaign)
    overwrite_json(toy_campaign/"added_after_audit.json", {"changed": True})
    from ra_lsbo.experiments.comparison import visualization
    monkeypatch.setattr(visualization, "make_figures", lambda *args: pytest.fail("Must fail before plotting"))
    with pytest.raises(ValueError, match="unchanged evidence"):
        runner.report(toy_campaign)
    assert not (toy_campaign/"summary.csv").exists()


def test_source_snapshot_tampering_is_detected(toy_campaign):
    snapshot = next((toy_campaign/"source_snapshot").glob("*.source"))
    snapshot.write_bytes(snapshot.read_bytes()+b"\n# tampered\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        runner.audit(toy_campaign)


def test_rewritten_candidate_pool_is_detected(toy_campaign):
    path = toy_campaign/"optimization/branin_sparse/9601_ra_lsbo/proposals.jsonl"
    rows = runner.lines(path)
    rows[0]["pool"] = rows[0]["pool"][::-1]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="pool mismatch"):
        runner.audit(toy_campaign)
