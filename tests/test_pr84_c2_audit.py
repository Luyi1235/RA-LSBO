"""Synthetic contracts only: no model/evaluator import and no new science calls."""
import ast
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import pr84_c2_audit as a


def candidate(i):
    return dict(branch=0,x=(i+1)/20.,y=.2,trim=.5)


def observation(i,region=-.2,resource=-.2,y=1.):
    c=candidate(i)
    return dict(candidate=c,candidate_id=a.digest(c),
        region=dict(kind='exact',value=region) if region<=0 else dict(kind='interval',lower=0.,lower_inclusive=False),
        resource=dict(kind='exact',value=resource),
        objective=dict(kind='exact',value=y) if region<=0 else dict(kind='missing',reason='controlled_region_abort'))


def unit_fixture(root,cold=False):
    u=dict(task='tradeoff_s0',repeat=0,seed=960000);spec=dict(initial=2,steps=1,pool=3)
    prefix='units/tradeoff_s0/r0';rows=[]
    def save(ref,value,kind):rows.append(dict(ref=prefix+'/'+ref,value=value,event_type=kind))
    initial=[observation(0,y=2.),observation(1,region=1.)] if not cold else [observation(0,region=1.),observation(1,region=1.)]
    raw=[dict(call_id='tradeoff_s0/r0/initial/%03d'%i,candidate=x['candidate'],candidate_id=x['candidate_id'],observed=x,truth={'opaque':'NOT USED'}) for i,x in enumerate(initial)]
    save('initial.json',raw,'initial_sealed')
    ids={'region':'r','resource':'s'}
    if not cold:ids['objective']='o'
    mid=a.digest(dict(training_sha256=a.digest(initial),seed=u['seed'],risk_only=False,heads=ids))
    ir=dict(status='OK',training_sha256=a.digest(initial),heads={k:dict(identity=v) for k,v in ids.items()},model_id=mid)
    save('models/initial.json',dict(receipt=ir,bill_delta={k:0 for k in a.BILL_KEYS}),'initial_composite_sealed')
    pool=[candidate(i) for i in (2,3,4)];cids=[a.digest(c) for c in pool]
    save('pools/t000.json',pool,'pool_sealed');trials=[]
    for arm in a.ARMS:
        ur=arm in (a.ARMS[1],a.ARMS[3],a.ARMS[4]);uo=arm in (a.ARMS[2],a.ARMS[3])
        model=mid;modes={k:dict(mode='frozen',identity=v) for k,v in ids.items()}
        if cold:modes['objective']=dict(mode='frozen_unavailable')
        if arm==a.ARMS[4]:
            model=a.digest({k:ids[k] for k in ('region','resource')});modes={k:modes[k] for k in ('region','resource')}
        if arm==a.ARMS[5]:model=None;modes=None
        pred=None if arm==a.ARMS[5] else dict(model_id=model,candidate_ids=cids,
            probabilities=dict(region=[.5]*3,resource=[.5]*3,joint=[.25]*3),
            objective=None if cold or arm==a.ARMS[4] else dict(mean=[1.]*3,variance=[.5]*3))
        pref=prefix+'/predictions/'+arm+'/t000.json'
        save('predictions/'+arm+'/t000.json',dict(model_id=model,prediction=pred,query_sha256=a.digest(pool),bill_delta={k:0 for k in a.BILL_KEYS}),'prediction_sealed')
        d=a.selection(arm,cids,list(range(3)),pred,a.best(initial),['tradeoff_s0/r0',0])
        prop=dict(history_sha256=a.digest(initial),history_information=a.information(initial),
            pool_ref=prefix+'/pools/t000.json',pool_sha256=a.digest(pool),available_indices=list(range(3)),
            prediction_ref=pref,model_id=model,head_modes=modes,update_factors=[ur,uo],decision=d)
        save('trials/'+arm+'/proposal_000.json',prop,'proposal_sealed')
        obs=observation(d['selected_index']+2,resource=.2,y=.1)
        raw=dict(call_id='tradeoff_s0/r0/'+arm+'/000',candidate=obs['candidate'],candidate_id=obs['candidate_id'],observed=obs,truth={'opaque':'NEVER EVALUATED'})
        save('trials/'+arm+'/observation_000.json',raw,'observation_sealed')
        trials.append(a.trial_replay(u['task'],0,arm,initial,[obs],[prop]))
    for arm,t in zip(a.ARMS,trials):save('trials/'+arm+'/result.json',t,'trial_sealed')
    p=root/prefix/'records.jsonl';p.parent.mkdir(parents=True)
    p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    return spec,u,p,rows


class Metrics(unittest.TestCase):
    def test_brier(self):self.assertAlmostEqual(a.binary([True,False],[.8,.2])['brier'],.04)
    def test_auc_tie(self):self.assertEqual(a.binary([True,False],[.5,.5])['auroc'],.5)
    def test_auc_perfect(self):self.assertEqual(a.auc([False,True],[.1,.9]),1.)
    def test_auc_reverse(self):self.assertEqual(a.auc([False,True],[.9,.1]),0.)
    def test_auc_single(self):self.assertIsNone(a.auc([True,True],[.1,.9]))
    def test_empty(self):self.assertIsNone(a.binary([],[])['brier'])
    def test_invalid(self):
        for p in (float('nan'),float('inf'),-.1,1.1):
            with self.assertRaises(ValueError):a.binary([True],[p])
    def test_misaligned(self):
        with self.assertRaises(ValueError):a.binary([True],[])
    def test_boolean_labels(self):
        with self.assertRaises(ValueError):a.binary([1],[.4])
    def test_logclip(self):
        r=a.binary([True,False],[0.,1.]);self.assertEqual(r['clipped_events'],2);self.assertTrue(math.isfinite(r['log_loss']))
    def test_factorial(self):self.assertEqual(a.effect([10.,8.,7.,4.]),dict(complete=True,risk=-2.5,objective=-3.5,interaction=-1.))
    def test_missing_factorial(self):self.assertFalse(a.effect([None,1.,2.,3.])['complete'])
    def test_sign_and_holm(self):
        self.assertEqual(a.sign_p(11,1),.00634765625)
        p={str(i):(.00634765625 if i<2 else 1.) for i in range(12)}
        self.assertEqual(a.holm(p)['0']['holm_p'],.076171875)
    def test_tolerance(self):self.assertTrue(a.same(1.,1.+1e-12));self.assertFalse(a.same(True,1))
    def test_ei_variance_zero(self):self.assertEqual(a.ei(1.,0.,2.),1.)


class Feedback(unittest.TestCase):
    def test_resource_violation_not_abort(self):
        x=observation(1,resource=.2);r,s,f=a.observed_check(x)
        self.assertTrue(r);self.assertFalse(s);self.assertFalse(f);self.assertEqual(x['objective']['kind'],'exact')
    def test_incumbent_not_online_success(self):
        init=[observation(0,y=.2)];obs=[observation(1,region=2.)]
        r=a.trial_replay('t',0,'a',init,obs,[dict(decision=dict(mode='risk_ei'))])
        self.assertEqual(r['final_best'],.2);self.assertEqual(r['first_feasible_call'],0)
    def test_zero_objective_is_feasible(self):self.assertEqual(a.best([observation(0,y=0.)]),0.)
    def test_first_online_feasible(self):
        r=a.trial_replay('t',0,'a',[observation(0,region=2.)],[observation(1)], [dict(decision=dict(mode='risk_ei'))])
        self.assertEqual(r['first_feasible_call'],1)
    def test_missing_objective_not_imputed(self):
        self.assertIsNone(a.best([observation(0,region=2.)]))
    def test_observation_identity(self):
        x=observation(0);x['candidate_id']='bad'
        with self.assertRaises(ValueError):a.observed_check(x)
    def test_unit_end_to_end_resource_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,u,p,rows=unit_fixture(root);z=a.audit_unit(root,spec,u)
            self.assertEqual(len(z['trials']),6);self.assertEqual(z['bill']['math_settled'],8)
            self.assertTrue(all(not x['joint_label'] and x['region_label'] for x in z['selected']))
            self.assertTrue(all(r['final_best']==2. for r in z['trials']))
    def test_unit_cold_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,u,p,rows=unit_fixture(root,True);z=a.audit_unit(root,spec,u)
            self.assertFalse(z['initial_warm']);self.assertTrue(all(r['final_best'] is None for r in z['trials']))
    def test_forged_result_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,u,p,rows=unit_fixture(root);rows[-1]['value']['final_best']=-9.
            p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Trial not supported'):a.audit_unit(root,spec,u)
    def test_duplicate_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,u,p,rows=unit_fixture(root);rows.append(rows[-1])
            p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Duplicate'):a.audit_unit(root,spec,u)
    def test_wrong_selection_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,u,p,rows=unit_fixture(root)
            row=next(x for x in rows if '/proposal_' in x['ref']);row['value']['decision']['selected_index']=(row['value']['decision']['selected_index']+1)%3
            p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Selected action'):a.audit_unit(root,spec,u)


class Files(unittest.TestCase):
    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'x').write_text('a');a.seal(root);self.assertEqual(a.verify_manifest(root),1)
            (root/'x').write_text('b')
            with self.assertRaises(ValueError):a.verify_manifest(root)
    def test_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'x').write_text('a');a.seal(root);(root/'y').write_text('b')
            with self.assertRaises(ValueError):a.verify_manifest(root)
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for path in ('../outside','/tmp/x','a\\b','a/../b','C:/x'):
                with self.assertRaises(ValueError):a.safe_path(Path(td),path)
    def test_write_once(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x';a.write_json(p,{'a':1})
            with self.assertRaises(FileExistsError):a.write_json(p,{'b':2})
    def test_no_model_evaluator_import(self):
        tree=ast.parse(Path(a.__file__).read_text())
        allowed={'csv','hashlib','io','json','math','pathlib','re','statistics','subprocess'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertTrue(all(n.name in allowed for n in node.names))
            if isinstance(node,ast.ImportFrom):self.assertIn(node.module,allowed)


if __name__=='__main__':unittest.main()
