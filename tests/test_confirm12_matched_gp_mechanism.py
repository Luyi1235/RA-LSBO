"""No-truth tests for paired GP view experiment and leakage resistance."""
from types import SimpleNamespace
import math
import numpy as np
import pytest

from confirm12 import matched_gp_mechanism as m
from confirm12.nested_seed_plan import candidate_plan
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool


def _full_plan():
    raw=candidate_plan()
    cases={}
    for case in m.CASES:
        nested=set(raw["auxiliary_by_case"][case])
        seeds=raw["seeds_by_case"][case]
        cases[case]={"seeds":seeds,"arms_by_seed":{
            str(seed):list(raw["main_arms"])+(
                list(raw["auxiliary_arms"]) if seed in nested else [])
            for seed in seeds}}
    return {"max_calls":151040,"workers":12,"cases":cases}


def test_outcome_blind_disjoint_fixed_seed_pairs():
    plan=_full_plan()
    pairs=m.fixed_pairs(plan,8)
    assert len(pairs)==16
    for case in m.CASES:
        task=[x for x in pairs if x[0]==case]
        train={x[1] for x in task}
        target={x[2] for x in task}
        assert not (train & target)
        assert train==set(sorted(plan["cases"][case]["seeds"])[:8])
        assert target==set(sorted(plan["cases"][case]["seeds"])[128:136])
    with pytest.raises(ValueError):
        m.fixed_pairs(plan,0)
    with pytest.raises(ValueError):
        m.fixed_pairs(plan,129)


def test_gaussian_risk_and_completion_probabilities():
    p=m._gaussian_probability([-1.,1.],[1.,1.])
    assert p[0]>.5 and p[1]<.5
    c=m._completion_probability({"mean":np.array([-1.,1.]),
                                 "variance":np.array([0.,0.])})
    assert c[0]>.5 and c[1]<.5
    with pytest.raises(ValueError):
        m._gaussian_probability([0.],[float("nan")])


def test_true_2x2_shared_inputs_no_extra_models_no_evaluator(
        monkeypatch):
    problem=problem_for("aircraft_ordinary")
    pool,_=candidate_pool(problem,4001001,30)
    def record(candidate):
        constraints=dict(problem.precheck(candidate))
        constraints.update({name:-.1 for name in problem.required_constraints
                            if name not in problem.analytical_constraints})
        return problem.record(candidate,29000.,constraints,{"fake_and_offline":True})
    train=[record(x) for x in pool[:12]]
    targets=[record(x) for x in pool[12:24]]
    observed_fit_calls=[]
    class GP:
        def __init__(self,view):
            self.view=view
            self.objective=SimpleNamespace(scale=100.)
        def predict(self,candidates):
            n=len(candidates)
            offset=0 if self.view=="latent" else 2
            def head(mu,scale=1.):
                return {"mean":np.full(n,mu+offset,float),
                        "variance":np.ones(n),"scale":scale}
            return {
                "objective":head(29001.,100.),
                "completion":head(-1.),
                "constraints":{
                    key:head(-.1,.1) for key in problem.required_constraints
                    if key not in problem.analytical_constraints
                },
            }
    def build(problem_,seed,records):
        assert problem_ is problem
        assert seed==9
        assert all(a is b for a,b in zip(records,train))
        observed_fit_calls.append("once")
        return {"latent":GP("latent"),"onehot":GP("onehot")},{
            "fit_on_train_only":True}
    monkeypatch.setattr(m,"problem_for",lambda case:problem)
    monkeypatch.setattr(m,"_train_views",build)
    rows,receipt=m.score_pair(
        "aircraft_ordinary",9,10,train,targets)
    assert len(observed_fit_calls)==1
    assert receipt["fitted_models"]==2
    assert receipt["no_online_arm_outcomes_in_training"]
    assert receipt["actual_representation_allocation"]["ra_lsbo"]=={
        "objective":"latent","risk":"onehot"}
    assert receipt["actual_representation_allocation"]["reverse_representation"]=={
        "objective":"onehot","risk":"latent"}
    assert len({r["candidate_id"] for r in rows})==12
    assert sum(r["head"]=="objective" for r in rows)==24
    assert sum(r["head"]=="completion" for r in rows)==24
    assert all(not math.isnan(r["mean"]) for r in rows)


def test_rejects_training_holdout_overlap_before_GP_fit(monkeypatch):
    problem=problem_for("aircraft_ordinary")
    candidate=problem.space.default().to_dict()
    values={name:-.1 for name in problem.required_constraints}
    row=problem.record(candidate,3.,values,{"fake":True})
    monkeypatch.setattr(m,"problem_for",lambda case:problem)
    monkeypatch.setattr(m,"_train_views",
        lambda *args:pytest.fail("Must not fit after detecting leaked design"))
    with pytest.raises(ValueError,match="seen in training"):
        m.score_pair("aircraft_ordinary",1,2,[row]*12,[row]*12)


def test_aggregate_displays_false_feasible_and_boundary():
    rows=[
        {"case":"aircraft_ordinary","head":"soc","view":"latent",
         "train_seed":1,"brier":.64,"squared_error":.25,
         "near_boundary":True,"false_feasible":True},
        {"case":"aircraft_ordinary","head":"soc","view":"latent",
         "train_seed":1,"brier":.04,"squared_error":.01,
         "near_boundary":False,"false_feasible":False},
    ]
    item=m.aggregate(rows)["aircraft_ordinary/soc/latent"]
    assert item["false_feasible_count_at_0_5"]==1
    assert item["resolved_binary"]==2
    assert item["near_boundary_exact"]==1
    assert math.isclose(item["mean_brier"],.34)
    assert math.isclose(item["rmse"],math.sqrt(.13))
