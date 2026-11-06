"""Actual EP-head tests plus original-PR68 parity; no true evaluator calls."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableGroupedHead
from ra_lsbo.experiments.information_value_stage3.native import FrozenFantasyFactory
from ra_lsbo.experiments.information_value_stage3.planner import Outcome, choose, FantasyFitError
from pr70_future_pool import (CandidateOnly, NextPoolFactory, future_candidates, choose_next, trim_first)
from pr70_protocol import check_transition


class Features:
    def transform(self, candidates):
        return np.array([[c["x"],1.,0.] for c in candidates])


def fixture():
    x=np.array([[v,1.,0.] for v in (0.,.25,.7,1.)])
    def head(values,name,objective=False):
        obs=[OutputObservation("exact",name+str(i),value=v) for i,v in enumerate(values)]
        return StableGroupedHead(x,obs,continuous_count=1,seed=19,name=name,
                                  objective=objective,reference_scale=1.)
    heads={"region":head([.7,.3,-.4,-.2],"region"),"resource":head([-.4,-.2,-.3,-.1],"resource")}
    objective=head([8.,6.,3.,5.],"objective",True)
    features=Features()
    def predict(candidates):
        z=features.transform(candidates)
        return {"constraints":{n:h.predict(z) for n,h in heads.items()},"objective":objective.predict(z)}
    posterior=SimpleNamespace(constraints=heads,objective=objective,risk_features=features,
                              objective_features=features,predict=predict)
    candidates=[{"x":v} for v in (.4,.55,.8,.95)]
    return posterior,candidates,predict(candidates)


class SamePool:
    """Only for parity: supports are current pool minus first, same real EP GP."""
    def __init__(self,posterior,candidates,forecast):
        self.candidates=candidates
        self.real=FrozenFantasyFactory(posterior,candidates,forecast)
    def support(self,i):
        points=[c for j,c in enumerate(self.candidates) if j!=i]
        return {"iteration":1,"model_candidates":points,"model_pool_sha256":digest(points)}
    def condition(self,i,o,**kw):
        f=self.real.condition(i,o,**kw)
        def drop(h):
            return None if h is None else {**h,"mean":np.delete(h["mean"],i),"variance":np.delete(h["variance"],i)}
        return {"constraints":{n:drop(h) for n,h in f["constraints"].items()},"objective":drop(f["objective"])}


class FuturePoolTests(unittest.TestCase):
    def test_coordinate_view_has_no_labels(self):
        c=CandidateOnly({"x":1})
        self.assertFalse(hasattr(c,"observation"))
        self.assertFalse(hasattr(c,"usable_feasible"))

    def test_coordinate_view_is_copied(self):
        x={"x":1}; c=CandidateOnly(x); x["x"]=2
        self.assertEqual(c.candidate,{"x":1})

    def test_same_support_matches_PR68_scores_and_winner(self):
        p,c,f=fixture()
        a,ra=choose(f,5.,2,FrozenFantasyFactory(p,c,f),arm="cei_two_step",seed=123,iteration=0)
        b,rb=choose_next(f,5.,2,SamePool(p,c,f),seed=123,iteration=0)
        self.assertIsNone(ra["fallback"]); self.assertIsNone(rb["fallback"])
        self.assertEqual(a,b)
        np.testing.assert_allclose(ra["scores"],rb["scores"],rtol=1e-10,atol=1e-10)

    def test_last_step_never_accesses_future(self):
        p,c,f=fixture()
        a,ra=choose_next(f,5.,1,None,seed=123,iteration=0)
        b,rb=choose(f,5.,1,None,arm="cei_myopic",seed=123,iteration=0)
        self.assertEqual(a,b); self.assertEqual(ra["scores"],rb["scores"])
        self.assertEqual(ra["future_supports"],{})
        self.assertEqual(ra["fantasies"],[])

    def test_zero_remaining_rejected(self):
        with self.assertRaises(ValueError): choose_next({},None,0,None,seed=1,iteration=0)

    def test_next_query_real_EP_shapes_and_head_counts(self):
        p,c,f=fixture(); future=[{"x":.35},{"x":.65},{"x":.9}]
        with patch("pr70_future_pool.future_candidates",return_value=(future,future)):
            factory=NextPoolFactory(p,c,None,"unused",1,0,[])
            result=factory.condition(0,Outcome(True,None,-.3,None))
        self.assertEqual(len(result["constraints"]["region"]["mean"]),3)
        self.assertEqual(factory.diagnostics()["head_fits_completed"],2)
        self.assertEqual(factory.diagnostics()["real_evaluator_calls"],0)

    def test_next_query_matches_direct_frozen_adapter(self):
        p,c,f=fixture(); future=[{"x":.35},{"x":.65}]
        o=Outcome(False,-.05,.4,4.)
        with patch("pr70_future_pool.future_candidates",return_value=(future,future)):
            factory=NextPoolFactory(p,c,None,"unused",1,0,[])
            got=factory.condition(0,o)
        joined=[c[0]]+future
        expected=trim_first(FrozenFantasyFactory(p,joined,p.predict(joined)).condition(0,o))
        for key in ("region","resource"):
            np.testing.assert_allclose(got["constraints"][key]["mean"],expected["constraints"][key]["mean"])
        np.testing.assert_allclose(got["objective"]["mean"],expected["objective"]["mean"])
        self.assertEqual(factory.diagnostics()["head_fits_completed"],3)

    def test_real_posterior_not_mutated(self):
        p,c,f=fixture(); before={n:digest(h.model.to_dict()) for n,h in p.constraints.items()}
        future=[{"x":.35},{"x":.65}]
        with patch("pr70_future_pool.future_candidates",return_value=(future,future)):
            factory=NextPoolFactory(p,c,None,"unused",1,0,[])
            factory.condition(0,Outcome(True,None,-.3,None))
        self.assertEqual(before,{n:digest(h.model.to_dict()) for n,h in p.constraints.items()})

    def test_no_new_abort_ablation(self):
        p,c,f=fixture()
        with self.assertRaises(ValueError):
            NextPoolFactory(p,c,None,"unused",1,0,[]).condition(0,Outcome(True,None,-.3,None),use_abort_interval=False)

    def test_whole_decision_numerical_fallback(self):
        p,c,f=fixture(); factory=SamePool(p,c,f)
        with patch.object(factory,"condition",side_effect=FantasyFitError("test failure")):
            a,r=choose_next(f,5.,2,factory,seed=123,iteration=0)
        self.assertEqual(a,r["myopic_index"])
        self.assertEqual(r["scores"],r["myopic_scores"])
        self.assertEqual(r["fallback"],"fantasy_numerical_failure_full_myopic")

    def test_programming_error_propagates(self):
        p,c,f=fixture(); factory=SamePool(p,c,f)
        with patch.object(factory,"condition",side_effect=ValueError("programming error")):
            with self.assertRaises(ValueError): choose_next(f,5.,2,factory,seed=123,iteration=0)

    def test_real_candidate_generation_label_independent(self):
        from ra_lsbo.experiments.information_value_stage1.runner import NativeAdapter
        from ra_lsbo.experiments.discovery_stages.policies import filter_near
        api=NativeAdapter(); problem=api.problem("branin_sparse"); seed=998890
        designs=api.initial(problem,"branin_sparse",seed)
        history=[CandidateOnly(c) for c in designs]
        first=api.pool(problem,"branin_sparse",seed,0,history)[0]
        # A future history with labels that raise if inspected gives the same pool.
        class Poison:
            def __getattribute__(self,name): raise AssertionError("future observation read")
        labelled=[SimpleNamespace(candidate=c,observation=Poison()) for c in designs+[first]]
        with patch.object(problem,"evaluate",side_effect=AssertionError("true evaluator called")):
            raw,model=future_candidates(problem,"branin_sparse",seed,0,history,first)
            actual=api.pool(problem,"branin_sparse",seed,1,labelled)
            protected=filter_near(problem,actual,labelled)[0]
        self.assertEqual(raw,actual); self.assertEqual(model,protected)
        self.assertNotIn(first,model)
        self.assertEqual(len(raw),128)

    def test_future_second_hash_belongs_to_future_not_current(self):
        p,c,f=fixture(); future=[{"x":.35},{"x":.65},{"x":.9}]
        with patch("pr70_future_pool.future_candidates",return_value=(future,future)):
            factory=NextPoolFactory(p,c,None,"unused",1,0,[])
            _,r=choose_next(f,5.,2,factory,seed=123,iteration=0)
        self.assertIsNone(r["fallback"])
        valid={digest(x) for x in future}
        for row in r["fantasies"]:
            if row["second_index"] is not None:
                self.assertIn(row["second_candidate_sha256"],valid)


if __name__ == "__main__": unittest.main()
