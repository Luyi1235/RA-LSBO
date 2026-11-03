"""Genuine six-arm prototype BO engines, nonphysical evidence only."""
import os
from types import SimpleNamespace

import pytest

from confirm12.real_candidate_engines import (
    CANDIDATE_ARMS, ALL_REQUESTED, IntervalMissingHybridCandidate,
    create_engine, qualification_report,
)
from confirm12.offline_candidate_trial import OfflineCandidateTrial
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.risk_core.observations import OutputObservation


def forbidden(*args,**kwargs):
    raise AssertionError("No aircraft eval or VAE fit at engine construction")


@pytest.mark.parametrize("method", CANDIDATE_ARMS)
def test_six_distinct_real_engine_constructors_without_evaluations(
        monkeypatch, tmp_path, method):
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.structure_corpus import CorpusRepresentation
    monkeypatch.setattr(AircraftCase, "evaluate", forbidden)
    monkeypatch.setattr(CorpusRepresentation, "fit", forbidden)
    problem, engine, receipt = create_engine(
        "aircraft_ordinary", 800011, method, tmp_path/method)
    assert problem is not None and callable(engine.propose) and callable(engine.accept)
    assert receipt.method == method
    assert (receipt.initial,receipt.horizon,receipt.pool) == (12,18,128)
    assert not receipt.confirmed_for_experiment
    assert not receipt.authorization_to_evaluate
    # All these are real engines: never the shared synthetic placeholder.
    assert type(getattr(engine,"engine",engine)).__name__ != "SimulationAdapter"


def test_seventh_arm_real_independent_adaptation_still_unqualified(tmp_path):
    problem, engine, receipt = create_engine(
        "aircraft_ordinary",3,"cbob_adapted",tmp_path/"x")
    assert problem is not None and callable(engine.propose)
    assert "EICB_independent" in receipt.origin_h1_method
    assert receipt.authorization_to_evaluate is False
    assert receipt.confirmed_for_experiment is False
    report = qualification_report()
    assert set(report) == set(ALL_REQUESTED) and len(report)==7
    assert all(not row["confirmatory_qualified"] for row in report.values())
    assert all(not row["physical_evaluation_enabled"] for row in report.values())


def test_candidate_arms_have_separate_implementation_origins(tmp_path):
    origins = {}
    for method in CANDIDATE_ARMS:
        _, engine, receipt = create_engine(
            "aircraft_sparse", 800018, method, tmp_path / method)
        origins[method] = receipt.origin_h1_method
    assert origins["ra_lsbo"] == "hybrid_onehot"
    assert origins["full_latent"] == "corpus_stable"
    assert origins["full_onehot"] == "onehot_stable"
    assert origins["random"] == "random"
    assert "reverse_representation" in origins["reverse_representation"]
    assert "interval_as_missing" in origins["interval_as_missing"]
    assert len(set(origins.values())) == len(CANDIDATE_ARMS)


def test_interval_mask_only_model_outputs_not_raw_incumbent(monkeypatch):
    import confirm12.real_candidate_engines as m
    problem = problem_for("aircraft_ordinary")
    a = problem.space.default().to_dict()
    b = problem.space.default().to_dict()
    b["continuous"]["wing_area"] += .1
    # Both are purpose-built records, not observations from evaluate().
    good = problem.record(a, 1234., {"soc":-.2, "stall":-.1},
                          {"synthetic_fixture":True})
    interval = OutputObservation(
        "interval", "synthetic_soc_interval", lower=0.,
        lower_inclusive=False)
    missing = OutputObservation(
        "interval", "synthetic_objective_upper", upper=8000.)
    censored = problem.record(
        b, missing, {"soc":interval}, {"synthetic_fixture":True})
    originals = [good,censored]
    captured = {}
    class FakePosterior:
        def __init__(self, problem_, observed, seed, settings,
                     obj_features, risk_features, **kwargs):
            captured["observed"] = observed
            captured["objective_features"] = obj_features
            captured["risk_features"] = risk_features
            captured["policy"] = kwargs["view_policy"]
            self.best = None
            self.history_digest = None
    monkeypatch.setattr(m, "HeterogeneousRiskPosterior", FakePosterior)
    monkeypatch.setattr(m, "feature_snapshot", lambda *a: "latent")
    monkeypatch.setattr(m, "PrefixFeatures", lambda *a: "explicit")
    instance = object.__new__(IntervalMissingHybridCandidate)
    instance.problem = problem
    instance.base = "unused"
    instance.seed = 10
    instance.settings = object()
    instance.session = object()
    posterior = instance._build_posterior(originals)
    assert posterior.best == 1234.
    assert captured["objective_features"]=="latent"
    assert captured["risk_features"]=="explicit"
    assert captured["policy"]=="onehot_risk_gp_intervals_missing_v1"
    assert captured["observed"][0].observation.objective.kind=="exact"
    assert captured["observed"][1].observation.objective.kind=="missing"
    assert captured["observed"][1].observation.constraint("soc").kind=="missing"
    assert originals[1].observation.objective.kind=="interval"
    assert originals[1].observation.constraint("soc").kind=="interval"
    assert posterior.raw_history_digest != posterior.masked_history_digest


def _fixture(problem, candidate):
    constraints=dict(problem.precheck(candidate))
    constraints.update({k:-.1 for k in problem.required_constraints
                        if k not in problem.analytical_constraints})
    return problem.record(candidate, 29000.+candidate["continuous"]["wing_area"],
                          constraints, {"offline_fixture_without_aircraft":True})


def test_real_candidate_random_closed_loop_no_evaluator(monkeypatch,tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,"evaluate",forbidden)
    problem=problem_for("aircraft_ordinary")
    pool,_=candidate_pool(problem,34561,36)
    trial=OfflineCandidateTrial(
        "aircraft_ordinary",15,"random",tmp_path/"candidate_random",
        [_fixture(problem,c) for c in pool[:12]])
    before=trial.history_digest
    chosen,meta=trial.propose(pool[12:])
    assert meta["method"]=="random"
    trial.accept(_fixture(problem,chosen))
    assert trial.history_digest!=before
    assert len(trial._history)==13


@pytest.mark.skipif(os.getenv("CONFIRM12_RUN_REAL_SIX_GP")!="1",
                    reason="costly true surrogate/encoder fitting is explicit opt-in")
@pytest.mark.parametrize(
    "method", ("full_onehot","full_latent","ra_lsbo",
               "reverse_representation","interval_as_missing","cbob_adapted"))
def test_real_candidate_gp_closed_loop_one_step(monkeypatch,tmp_path,method):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,"evaluate",forbidden)
    problem=problem_for("aircraft_ordinary")
    pool,_=candidate_pool(problem,34771,44)
    records=[_fixture(problem,c) for c in pool[:12]]
    trial=OfflineCandidateTrial(
        "aircraft_ordinary",3462,method,tmp_path/method,records)
    selection,meta=trial.propose(pool[12:])
    assert problem.canonical(selection)==selection
    assert isinstance(meta,dict)
    assert trial._pending==selection
    trial.accept(_fixture(problem,selection))
    assert len(trial._history)==13
    assert trial._pending is None
