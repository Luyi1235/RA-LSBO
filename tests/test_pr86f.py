"""Pure regression and synthetic end-to-end tests; never scientific Full data."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from pr86f_protocol import METHODS, plan
from pr86f_acquisition import scores, choose
import pr86f_experiment as ex
from risk_learning.core import candidates, observed
from pr86c_acquisition import score_candidates, entropy


class SyntheticBackend:
    kind="synthetic_test"
    def __init__(self,mode="mixed"):
        self.mode=mode
    def candidates(self,seed,count):
        return candidates(seed,count)
    def evaluate(self,task,c):
        region=1.0 if self.mode=="no_objective" else -1.0 if self.mode=="no_feasible" else c["x"]-.5
        resource=1.0 if self.mode=="no_feasible" else c["y"]-.9
        target=dict(region=region,resource=resource,objective=(c["x"]-.2)**2+c["y"],feasible=region<=0 and resource<=0)
        return target, observed(c,target)
    def risk(self,rows,queries,seed,bill):
        assert all("truth" not in r for r in rows)
        diag=dict(candidates=[dict(status="ok") for _ in range(8)])
        for k in ("gp_head_started","gp_head_completed"):bill[k]+=2
        for k in ("gp_kernel_started","gp_kernel_completed"):bill[k]+=16
        ps=dict(region=[.2+.6*(1-c["x"]) for c in queries],resource=[.3+.6*(1-c["y"]) for c in queries])
        ps["joint"]=[a*b for a,b in zip(ps["region"],ps["resource"])]
        payload=[{k:r[k] for k in ("candidate","candidate_id","region","resource")} for r in rows]
        rr=dict(status="OK",method="native_censored",objective_labels_used=False,privileged_training=False,
            training_rows=len(rows),training_input_sha256=ex.digest(payload),query_ids=[ex.digest(c) for c in queries],
            heads=dict(region=diag,resource=diag))
        return dict(probabilities=ps),rr
    def objective(self,rows,queries,seed,bill):
        assert all("truth" not in r for r in rows)
        exact=[r for r in rows if r["objective"]["kind"]=="exact"]
        receipt=dict(training_rows=len(rows),exact_training_rows=len(exact),privileged_training=False,
            missing_labels_imputed=False,training_input_sha256=ex.digest([(r["candidate_id"],r["objective"]) for r in exact]),
            query_ids=[ex.digest(c) for c in queries])
        if len(exact)<2:
            receipt["status"]="INSUFFICIENT_EXACT_OBJECTIVES"
            return None,receipt
        for k in ("objective_gp_head_started","objective_gp_head_completed"):bill[k]+=1
        for k in ("objective_gp_kernel_started","objective_gp_kernel_completed"):bill[k]+=8
        receipt.update(status="OK",objective_fit_only_exact=True,head=dict(candidates=[dict(status="ok") for _ in range(8)]))
        return dict(mean=[(c["x"]-.2)**2+c["y"] for c in queries],variance=[.1]*len(queries)),receipt


class ProtocolTests(unittest.TestCase):
    def test_budgets(self):
        f,c=plan("full"),plan("ci")
        self.assertEqual((f["math_calls"],f["shared_anchor_calls"],f["online_calls"],f["test_calls"]),(18240,720,2160,15360))
        self.assertEqual((f["trajectories"],f["risk_refits"],f["risk_heads"],f["risk_kernels"]),(90,2250,4500,36000))
        self.assertEqual((c["math_calls"],c["risk_refits"]),(68,15))
        self.assertEqual(f["checkpoints"],[24,36,48])
        self.assertEqual(f["minimum_exact_objectives"],2)
    def test_seed_disjointness(self):
        def seeds(p):return [u[k] for u in p["units"] for k in ("anchor_seed","pool_seed","test_seed")]
        full,ci=seeds(plan("full")),seeds(plan("ci"))
        self.assertEqual(len(set(full)),90)
        self.assertFalse(set(full)&set(ci))
        self.assertTrue(all(s>=1210000 for s in full))
    def test_no_runtime_tuning_profile(self):
        with self.assertRaises(ValueError):plan("smoke")


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.cs=candidates(7,4)
        self.f=dict(probabilities=dict(region=[.1,.5,.8,.9],resource=[1.]*4,joint=[.1,.5,.8,.9]))
        self.o=dict(mean=[1.,2.,3.,4.],variance=[.1]*4)
    def test_frozen_baselines_exact(self):
        for m in METHODS[:2]:
            self.assertEqual(scores(m,self.f,self.o,2.5),score_candidates(m,self.f,self.o,2.5))
    def test_direct_definition(self):
        from pr86c_acquisition import expected_improvement
        ei=[expected_improvement(m,v,2.5) for m,v in zip(self.o["mean"],self.o["variance"])]
        actual,_=scores(METHODS[2],self.f,self.o,2.5)
        self.assertEqual(actual,[.65*(p*e)+.35*(e*entropy(p)) for p,e in zip(self.f["probabilities"]["joint"],ei)])
    def test_cold_definition_no_imputation(self):
        a,_=scores(METHODS[2],self.f,None,None)
        self.assertEqual(a,[.65*p+.35*entropy(p) for p in self.f["probabilities"]["joint"]])
        self.assertEqual(a,scores(METHODS[2],self.f,self.o,None)[0])
    def test_direct_retains_scale(self):
        a=scores(METHODS[2],self.f,self.o,2.5)[0]
        scale=3.
        o=dict(mean=[v*scale for v in self.o["mean"]],variance=[v*scale*scale for v in self.o["variance"]])
        b=scores(METHODS[2],self.f,o,2.5*scale)[0]
        for x,y in zip(a,b):self.assertAlmostEqual(scale*x,y,places=12)
    def test_zero_ei_and_ties(self):
        o=dict(mean=[100.]*4,variance=[0.]*4)
        for m in METHODS:
            i,r=choose(m,self.cs,self.f,o,0.)
            self.assertEqual(i,max(range(4),key=lambda k:ex.digest(self.cs[k])))
            self.assertTrue(all(math.isfinite(v) for v in r["scores"]))
    def test_reject_nonfinite(self):
        f=deepcopy(self.f);f["probabilities"]["joint"][0]=float("nan")
        with self.assertRaises(ValueError):scores(METHODS[2],f,self.o,1.)
    def test_reject_bad_variance(self):
        o=deepcopy(self.o);o["variance"][0]=-1
        with self.assertRaises(ValueError):scores(METHODS[2],self.f,o,1.)
    def test_no_false_advantage_from_proxy(self):
        null=dict(best_exact_feasible=None)
        self.assertEqual(ex.compare(null,null),(0,None))
        self.assertEqual(ex.compare(dict(best_exact_feasible=9.),null),(1,None))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/"run"
        self.sha="a"*40
        self.source=dict(code_sha=self.sha,parent_sha="test",frozen_models_unchanged=True)
    def produce(self,mode="mixed"):
        return ex.run(self.root,"ci",self.source,SyntheticBackend(mode))
    def reseal(self):
        (self.root/"manifest.json").unlink(); ex.seal(self.root)
    def unit(self):
        return next((self.root/"units").rglob("unit.json"))
    def save_unit(self,data):
        self.unit().write_text(json.dumps(data),encoding="utf-8");self.reseal()
    def test_end_to_end_and_shared_anchors(self):
        report=self.produce()
        self.assertEqual(report["bill"]["math_settled"],68)
        v=ex.verify(self.root,self.sha,allow_synthetic=True)
        self.assertEqual(v["decisions"],12)
        a=ex.analyze(self.root,Path(self.tmp.name)/"analysis",self.sha,allow_synthetic=True)
        self.assertEqual((a["outcome_rows"],a["paired_rows"]),(6,6))
        with self.assertRaisesRegex(ValueError,"Synthetic"):
            ex.verify(self.root,self.sha)
        data=ex.read(self.unit())
        digests={t["frames"][0]["history_sha256"] for t in data["trials"]}
        self.assertEqual(len(digests),1)
    def test_cold_zero_objective_no_imputation(self):
        self.produce("no_objective")
        ex.verify(self.root,self.sha,allow_synthetic=True)
        d=ex.read(self.unit())
        for t in d["trials"]:
            self.assertIsNone(t["final_state"]["best_exact_feasible"])
            self.assertTrue(all(f["decision"]["cold_fallback"] for f in t["frames"][:-1]))
    def test_objective_available_without_feasible_incumbent(self):
        self.produce("no_feasible")
        ex.verify(self.root,self.sha,allow_synthetic=True)
        d=ex.read(self.unit())
        for t in d["trials"]:
            self.assertGreaterEqual(t["final_state"]["exact_objective_count"],2)
            self.assertFalse(t["final_state"]["feasible_found"])
            self.assertTrue(t["frames"][0]["decision"]["cold_fallback"])
    def test_existing_run_cannot_overwrite(self):
        self.produce()
        with self.assertRaisesRegex(ValueError,"overwrite"):
            self.produce()
    def test_manifest_mutation_detected(self):
        self.produce();self.unit().write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            ex.verify(self.root,self.sha,allow_synthetic=True)
    def test_resealed_wrong_baseline_scores_detected(self):
        self.produce();d=ex.read(self.unit())
        q=d["trials"][0]["frames"][0]["decision"]
        q["scores"][0]+=10;q["score_digest"]=ex.digest(q["scores"])
        self.save_unit(d)
        with self.assertRaisesRegex(ValueError,"Decision replay"):
            ex.verify(self.root,self.sha,allow_synthetic=True)
    def test_resealed_unobservable_incumbent_detected(self):
        self.produce();d=ex.read(self.unit())
        d["trials"][0]["frames"][0]["state"]["best_exact_feasible"]=-1e9
        self.save_unit(d)
        with self.assertRaisesRegex(ValueError,"endpoint"):
            ex.verify(self.root,self.sha,allow_synthetic=True)
    def test_final_receipt_is_checked(self):
        self.produce();d=ex.read(self.unit())
        d["trials"][0]["frames"][-1]["objective_receipt"]["query_ids"]=["wrong-id"]
        self.save_unit(d)
        with self.assertRaisesRegex(ValueError,"Objective query"):
            ex.verify(self.root,self.sha,allow_synthetic=True)
    def test_numeric_tolerance_not_identity_tolerance(self):
        self.assertTrue(ex.numeric_equal(.1,.1+1e-15))
        self.assertFalse(ex.numeric_equal(.1,.11))
        self.assertFalse(ex.numeric_equal("abc","abd"))
    def test_failure_is_retained(self):
        b=SyntheticBackend()
        with patch.object(b,"risk",side_effect=RuntimeError("synthetic fit failure")):
            with self.assertRaisesRegex(RuntimeError,"synthetic fit"):
                ex.run(self.root,"ci",self.source,b)
        self.assertTrue((self.root/"failure.json").is_file())
        with self.assertRaisesRegex(ValueError,"Failed run"):
            ex.verify(self.root,self.sha,allow_synthetic=True)


if __name__=="__main__":
    unittest.main()
