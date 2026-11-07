import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from risk_learning.core import (HEADS, MODELS, PARENT, TASKS, binary_metrics, candidates,
    canonical, digest, features, label_counts, native_candidate, observed,
    optimum_witness, plan, score, truth)
from risk_learning.io import read, write, seal, check_manifest
from risk_learning.native import predict, training_payload
from risk_learning.study import new_bill, run, verify


class DesignTests(unittest.TestCase):
    def test_frozen_budgets(self):
        for profile,total,rows,heads in [('ci',176,16,24),('smoke',912,48,72),('full',12672,360,540)]:
            p=plan(profile)
            self.assertEqual(p['total_math_calls'],total)
            self.assertEqual(p['model_rows'],rows)
            self.assertEqual(p['gp_head_attempts'],heads)
            self.assertFalse(p['online_optimization'])
    def test_unknown_profile(self):
        with self.assertRaises(ValueError): plan('large')
    def test_all_seeds_disjoint(self):
        seeds=[]
        for profile in ('ci','smoke','full'):
            for unit in plan(profile)['units']:
                seeds.extend([unit['train_seed'],unit['test_seed']])
        self.assertEqual(len(seeds),len(set(seeds)))
    def test_nested_designs(self):
        self.assertEqual(candidates(42,12),candidates(42,192)[:12])
    def test_train_test_disjoint(self):
        a,b=map(lambda s:{digest(c) for c in candidates(s,192)},[42,43])
        self.assertFalse(a & b)
    def test_reserved_tasks_blocked(self):
        for task in plan('full')['reserved_task_ids']:
            with self.assertRaises(ValueError): truth(task,candidates(1,1)[0])
    def test_inactive_not_clipped(self):
        with self.assertRaises(ValueError): canonical(dict(branch=0,x=.5,y=.5,trim=.8))
    def test_invalid_domain(self):
        for v in (-.1,1.1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError): canonical(dict(branch=1,x=v,y=.5,trim=.5))
    def test_invalid_branch(self):
        for b in (-1,4,1.,True):
            with self.assertRaises(ValueError): canonical(dict(branch=b,x=.5,y=.5,trim=.5))
    def test_features_canonical(self):
        c=dict(branch=0,x=.2,y=.3,trim=.5)
        self.assertEqual(features(c),[.2,.3,0.,1.,1.,0.,1.,0.,1.,0.,0.,0.,0.])
        self.assertEqual(native_candidate(c)['discrete']['module'],1)
    def test_tradeoff_branch_winner_switches(self):
        for s,task in enumerate(TASKS[2:]):
            values=[]
            for branch in range(4):
                target=truth(task,optimum_witness(task,branch))
                self.assertTrue(target['feasible'])
                self.assertAlmostEqual(target['objective'],.7*(branch-s)**2,places=12)
                values.append(target['objective'])
            self.assertEqual(values.index(min(values)),s)
    def test_legacy_dominance_witness(self):
        for task in TASKS[:2]:
            for c in candidates(36,512):
                t=truth(task,c)
                if c['branch'] and t['feasible']:
                    basic=dict(branch=0,x=c['x']+.035*(c['branch']-2),y=c['y'],trim=.5)
                    b=truth(task,basic)
                    self.assertTrue(b['feasible'])
                    self.assertLessEqual(b['objective'],t['objective']+1e-12)
    def test_mask_hides_failed_magnitude_and_objective(self):
        c=dict(branch=0,x=0.,y=0.,trim=.5)
        t=truth(TASKS[0],c);r=observed(c,t)
        self.assertEqual(r['region'],dict(kind='interval',lower=0.,lower_inclusive=False))
        self.assertEqual(r['objective']['kind'],'missing')
        self.assertNotIn('value',r['region'])
    def test_complete_observation(self):
        c=dict(branch=0,x=.68,y=.25,trim=.5)
        r=observed(c,truth(TASKS[0],c))
        self.assertEqual(r['region']['kind'],'exact')
        self.assertEqual(r['objective']['kind'],'exact')
    def test_label_counts_no_imputation(self):
        rows=[observed(c,truth(TASKS[0],c)) for c in candidates(22,24)]
        counts=label_counts(rows)
        self.assertEqual(sum(counts['region'][k] for k in ('exact','interval','missing')),24)
        self.assertEqual(counts['objective']['exact'],counts['region']['exact'])
    def test_ordinary_models_refuse_privileged_argument(self):
        c=candidates(1,1)[0];r=observed(c,truth(TASKS[0],c))
        for method in MODELS[:-1]:
            with self.assertRaises(ValueError): training_payload([r],method,[truth(TASKS[0],c)])
    def test_privileged_requires_explicit_data(self):
        with self.assertRaises(ValueError): training_payload([],MODELS[-1])
    def test_payload_drops_objective(self):
        c=candidates(1,1)[0];r=observed(c,truth(TASKS[0],c))
        payload=training_payload([r],'native_censored')
        self.assertNotIn('objective',payload[0]);self.assertNotIn('truth',payload[0])
    def test_binary_sign_drops_margin(self):
        c=dict(branch=0,x=.68,y=.25,trim=.5);r=observed(c,truth(TASKS[0],c))
        payload=training_payload([r],'native_sign')
        self.assertEqual(payload[0]['region'],dict(kind='interval',upper=0.))
    def test_training_only_baseline(self):
        c=dict(branch=0,x=0.,y=0.,trim=.5);r=observed(c,truth(TASKS[0],c))
        bill=new_bill();p,_=predict([r],[c],'joint_prevalence',0,bill)
        self.assertAlmostEqual(p['probabilities']['joint'][0],1/3)
        self.assertFalse(any(bill.values()))
    def test_empty_metrics(self):
        self.assertIsNone(binary_metrics([],[])['brier'])
    def test_one_class_auc_is_missing(self):
        self.assertIsNone(binary_metrics([True,True],[.7,.9])['auroc'])
    def test_probability_rejected(self):
        for p in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): binary_metrics([True],[p])
    def test_tied_auc(self):
        self.assertEqual(binary_metrics([True,False],[.5,.5])['auroc'],.5)
    def test_known_brier(self):
        self.assertAlmostEqual(binary_metrics([True,False],[.8,.3])['brier'],.065)
    def test_prediction_id_mismatch(self):
        with self.assertRaises(ValueError): score([],dict(candidate_ids=['foreign']))
    def test_reject_output_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): run(tmp,'ci',dict(code_sha='a'*40))
    def test_manifest_detects_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'a.json',{'a':1});seal(p);write(p/'extra.json',{})
            with self.assertRaises(ValueError): check_manifest(p)
    def test_manifest_detects_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'a.json',{'a':1});seal(p);(p/'a.json').write_text('{}')
            with self.assertRaises(ValueError): check_manifest(p)
    def test_failure_sealed_no_silent_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            def broken(*a,**k): raise RuntimeError('injected failure')
            with self.assertRaises(RuntimeError): run(root,'ci',dict(code_sha='a'*40),broken)
            f=read(root/'failure.json')
            self.assertEqual(f['bill']['math_started'],88)
            self.assertEqual(f['bill']['math_settled'],88)
            check_manifest(root)
            with self.assertRaises(ValueError): verify(root)
    def test_stdlib_imports_only(self):
        import subprocess,sys
        script='import risk_learning.study, risk_learning.native; import sys; assert "numpy" not in sys.modules; assert "torch" not in sys.modules'
        subprocess.run([sys.executable,'-S','-c',script],check=True)


if __name__=='__main__': unittest.main()
