"""Tests for cross-seed PR45 sample-size learning curve without physical calls."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from confirm12 import sample_size_gp_learning as m
from confirm12.nested_seed_plan import candidate_plan
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool


def _frozen_plan():
    x=candidate_plan()
    cases={}
    for case in m.CASES:
        nested=set(x["auxiliary_by_case"][case])
        seeds=x["seeds_by_case"][case]
        cases[case]={
            "seeds":seeds,
            "arms_by_seed":{
                str(seed):list(x["main_arms"])+(
                    list(x["auxiliary_arms"]) if seed in nested else [])
                for seed in seeds}}
    return {"max_calls":151040,"workers":12,"cases":cases}


def test_fixed_size_roster_no_outcome_selection():
    plan=_frozen_plan()
    tasks=m.fixed_study_plan(plan,folds=8,sizes=m.SIZES)
    assert len(tasks)==16
    assert all(len(t["train_seeds"])==16 and t["train_counts"]==[1,2,4,8,16]
               for t in tasks)
    for case in m.CASES:
        subset=[t for t in tasks if t["case"]==case]
        all_training={s for t in subset for s in t["train_seeds"]}
        all_tests={t["target_seed"] for t in subset}
        assert len(all_training)==128
        assert len(all_tests)==8
        assert all_tests==set(sorted(plan["cases"][case]["seeds"])[256:264])
        assert not all_training.intersection(all_tests)
    with pytest.raises(ValueError):
        m.fixed_study_plan(plan,folds=9)
    with pytest.raises(ValueError):
        m.fixed_study_plan(plan,sizes=(24,12))
    with pytest.raises(ValueError):
        m.fixed_study_plan(plan,sizes=())


def _records(n=36):
    problem=problem_for("aircraft_ordinary")
    pool,_=candidate_pool(problem,4720111,n+8)
    def rec(candidate):
        objective=32000.+float(candidate["continuous"]["wing_area"])
        margins={name:-.1 for name in problem.required_constraints}
        return problem.record(candidate,objective,margins,
                              {"synthetic_fixture_no_aircraft":True})
    return problem,[rec(c) for c in pool]


def test_training_coverage_catches_single_class_completion():
    _,recs=_records(24)
    data=m._label_overview(recs[:12])
    assert data["completion"]["observed"]==12
    assert data["completion"]["positive"]==12
    assert data["completion"]["negative"]==0
    assert data["completion"]["both_classes_observed"] is False
    baseline,count=m._baseline(recs[:12],"completion")
    assert count==12
    assert baseline==13/14


def test_train_only_baselines_and_fixed_truth_boundary():
    _,recs=_records(28)
    a,c=m._baseline(recs[:12],"objective")
    assert c==12 and math.isfinite(a)
    assert a!=0
    boundary=m._fixed_boundary(recs[12:24])
    assert all(len(v)==3 for v in boundary.values())  # ceil(12*0.25)
    assert all(all(0<=i<12 for i in ids) for ids in boundary.values())


def test_no_train_target_overlap_and_paired_models(monkeypatch):
    problem,records=_records(40)
    training=records[:24]
    targets=records[24:36]
    calls=[]
    class GP:
        def __init__(self,name):
            self.view=name
            self.objective=SimpleNamespace(scale=100)
        def predict(self,candidates):
            n=len(candidates)
            shift=0. if self.view=="latent" else 5.
            def head(x,scale=1.):
                return {"mean":np.full(n,x+shift),"variance":np.ones(n),"scale":scale}
            return {"objective":head(32000.,100.),
                    "completion":head(-2.),
                    "constraints":{
                        key:head(-.2,.1) for key in problem.required_constraints
                        if key not in problem.analytical_constraints}}
    def fake_train(prob,seed,train):
        assert prob is problem
        assert seed==707
        assert len(train)==24
        assert all(x is y for x,y in zip(train,training))
        calls.append(True)
        return {"latent":GP("latent"),"onehot":GP("onehot")},{"fit_on_train_only":True}
    monkeypatch.setattr(m,"problem_for",lambda case:problem)
    monkeypatch.setattr(m,"_train_views",fake_train)
    rows,receipt=m.forecast_level(
        "aircraft_ordinary",0,707,909,training,targets)
    assert len(calls)==1
    assert receipt["n"]==24
    assert receipt["training_coverage"]["completion"]["negative"]==0
    assert len({r["candidate_id"] for r in rows})==12
    assert len(rows)==192
    assert all(r["training_size"]==24 and r["fold"]==0 for r in rows)
    c=m._summary(rows)["aircraft_ordinary/24/completion/latent"]
    assert c["resolved_signs"]==12
    assert c["completion_has_both_classes"] is False
    assert c["constant_training_prevalence_brier"] is not None
    comparisons=m._paired_learning(rows)
    assert "24/aircraft_ordinary/objective/squared_error" in comparisons
    with pytest.raises(ValueError,match="overlaps training"):
        m.forecast_level("aircraft_ordinary",0,707,909,training,records[:12])


@pytest.mark.skipif(os.environ.get("RA_SAMPLE_SIZE_GP_REAL_SMOKE")!="1",
                    reason="Opt-in actual 2000-corpus VAE+GP on artificial offline labels")
def test_real_vae_gp_24_training_samples_with_no_aircraft(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    def forbidden(*args,**kwargs):
        pytest.fail("Physical truth call forbidden")
    monkeypatch.setattr(AircraftCase,"evaluate",forbidden)
    problem,records=_records(40)
    rows,receipt=m.forecast_level(
        "aircraft_ordinary",0,33,34,records[:24],records[24:36])
    assert receipt["n"]==24
    assert receipt["representation_initial_epochs"]==30
    assert receipt["both_views_from_same_training_labels"] is True
    assert len(rows)==192
    assert all(np.isfinite(row["mean"]) for row in rows)
