"""Six genuine BO arms sharing 12 source-shaped records, one step each, NO evaluate.

The records are synthetic.  This tests the real GP/VAE/SLSQP-to-transaction
seam, not aircraft evaluation, paper method qualification or execution approval.
"""
import os
import pytest

from confirm12 import structured_record_ledger as ledger
from confirm12.offline_ledger_trial import OfflineLedgerTrial
from confirm12.real_candidate_engines import ALL_REQUESTED, create_engine
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool


def _record(problem, candidate):
    constraints = dict(problem.precheck(candidate))
    constraints.update({
        name: -.05 for name in problem.required_constraints
        if name not in problem.analytical_constraints
    })
    return problem.record(
        candidate,
        33000. + 30.*(candidate["continuous"]["wing_area"]-14.)**2,
        constraints,
        {"synthetic_offline_training_not_physical_evaluation": True})


@pytest.mark.skipif(os.environ.get("CONFIRM12_TEST_SIX_ARM_LEDGER") != "1",
                    reason="explicit expensive six real-policy fitting + ledger opt-in")
def test_seven_real_candidate_optimizers_share_ledger_initials(monkeypatch,tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(
        AircraftCase, "evaluate",
        lambda *a,**kw: pytest.fail("FORBIDDEN physical airplane evaluation"))
    ordinary = problem_for("aircraft_ordinary")
    sparse = problem_for("aircraft_sparse")
    seed=801721
    plan={
        "schema": ledger.SCHEMA, "scope":ledger.SCOPE,
        "execution_authorized":False,"phase":"confirmation",
        "max_calls":12+len(ALL_REQUESTED),
        "initial":12,"online":18,"verification":1,
        "cases":{
            "aircraft_ordinary":{
                "seeds":[seed],
                "contracts":{
                    "20s":ordinary.contract_id,
                    "5s":problem_for("aircraft_ordinary",5.).contract_id},
                "arms_by_seed":{str(seed):list(ALL_REQUESTED)}},
            "aircraft_sparse":{
                "seeds":[],
                "contracts":{
                    "20s":sparse.contract_id,
                    "5s":problem_for("aircraft_sparse",5.).contract_id},
                "arms_by_seed":{}},
        },
    }
    db=tmp_path/"truth_shape_only.sqlite"
    ledger.create(db,plan)
    initial,_=candidate_pool(ordinary,801930,12)
    for i,candidate in enumerate(initial):
        ticket=ledger.start(db,"aircraft_ordinary",seed,"initial","",i,
                            candidate,ordinary)
        assert ledger.settle(db,ticket,_record(ordinary,candidate))=="settled"
    initial_digest=None
    selected={}
    for index,method in enumerate(ALL_REQUESTED):
        trial=OfflineLedgerTrial(
            db,plan,"aircraft_ordinary",seed,method,tmp_path/method,
            engine_builder=create_engine)
        if initial_digest is None:
            initial_digest=trial.policy.history_digest
        else:
            assert trial.policy.history_digest == initial_digest
        pool,_=candidate_pool(ordinary,801931+index,32,
                              {r.observation.design_id for r in trial.policy._history})
        candidate,meta=trial.propose(pool)
        assert isinstance(meta,dict)
        trial.accept(_record(ordinary,candidate))
        selected[method]=candidate
    report=ledger.audit(db,plan)
    assert report["ledger_contract_passed"],report
    assert report["states"]=={"settled":19}
    assert report["calls"]==19
    assert set(selected)==set(ALL_REQUESTED)
    assert len({tuple(c["discrete"].items()) for c in selected.values()}) >= 1
    assert not report["production_approved"]
