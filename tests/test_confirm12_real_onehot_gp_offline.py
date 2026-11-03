"""Actual historical one-hot GP + local acquisition, using fabricated offline records.

Nothing in this test calls the physical evaluator. It is NOT a scientific
result or a qualification of the proposed seven-arm confirmation trial.
"""
import os
import pytest
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool
from confirm12.offline_historical_trial import OfflineHistoricalTrial


@pytest.mark.skipif(os.environ.get("CONFIRM12_TEST_GP_ONEHOT") != "1",
                    reason="optional expensive historical GP/local-search smoke")
def test_real_onehot_one_step_gp_fit_and_accept(monkeypatch,tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.structure_corpus import CorpusRepresentation
    def forbidden(*args,**kwargs):
        raise AssertionError("Forbidden physical aircraft call or VAE training")
    monkeypatch.setattr(AircraftCase,"evaluate",forbidden)
    monkeypatch.setattr(CorpusRepresentation,"fit",forbidden)
    problem=problem_for("aircraft_ordinary")
    pool,_=candidate_pool(problem,800025,44)
    def fake_feedback(candidate):
        values=dict(problem.precheck(candidate))
        values.update({name:-.1 for name in problem.required_constraints
                       if name not in problem.analytical_constraints})
        f=32000.0+100*(candidate["continuous"]["wing_area"]-12.)**2
        return problem.record(candidate,f,values,
                              {"offline_synthetic_GP_smoke_no_aircraft":True})
    initial=[fake_feedback(c) for c in pool[:12]]
    engine=OfflineHistoricalTrial("aircraft_ordinary",800031,"onehot_stable",
                                  tmp_path/"onehot_engine",initial)
    selected, metadata=engine.propose(pool[12:])
    assert selected in pool[12:] or selected==problem.canonical(selected)
    assert isinstance(metadata,dict)
    assert engine._pending==selected
    engine.accept(fake_feedback(selected))
    assert len(engine._history)==13 and engine._pending is None
