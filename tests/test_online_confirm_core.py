import fnmatch
import inspect
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import unittest
from online_confirm_stats import outcome,sign_p,holm,paired,selected_forecast,analyze_units


class StatsTests(unittest.TestCase):
    def test_sign_empty(self):self.assertEqual(sign_p(0,0),1.)
    def test_sign_symmetric(self):self.assertEqual(sign_p(13,3),sign_p(3,13))
    def test_sign_exact(self):self.assertEqual(sign_p(5,0),.0625)
    def test_sign_cap(self):self.assertEqual(sign_p(4,4),1.)
    def test_sign_invalid(self):
        for x in (-1,True,1.5):
            with self.assertRaises(ValueError):sign_p(x,2)
    def test_holm(self):self.assertEqual(holm({'a':.01,'b':.04,'c':.03,'d':.2}),{'a':.04,'c':.09,'b':.09,'d':.2})
    def test_holm_invalid(self):
        with self.assertRaises(ValueError):holm({'a':float('nan')})
    def test_holm_empty(self):self.assertEqual(holm({}),{})
    def test_tie_uses_original_scale(self):self.assertEqual(outcome(1.,1.+1e-12),0)
    def test_missing_feasible_first(self):self.assertEqual(outcome(None,3.),-1)
    def test_nonfinite(self):
        with self.assertRaises(ValueError):outcome(float('nan'),1.)
    def test_paired_no_imputation(self):
        s=paired([dict(repeat=0,a=None,b=None)])
        self.assertIsNone(s['mean_delta']);self.assertIsNone(s['paired'][0]['delta'])
    def test_paired_duplicate(self):
        with self.assertRaises(ValueError):paired([dict(repeat=0,a=1,b=2)]*2)
    def test_paired_values(self):
        s=paired([dict(repeat=0,a=1.,b=2.),dict(repeat=1,a=3.,b=1.)])
        self.assertEqual(s['wins_losses_ties'],[1,1,0]);self.assertEqual(s['mean_delta'],.5)
    def test_forecast_empty(self):self.assertIsNone(selected_forecast([])['brier'])
    def test_forecast(self):self.assertAlmostEqual(selected_forecast([dict(p=.8,feasible=True)])['brier'],.04)
    def test_forecast_invalid(self):
        with self.assertRaises(ValueError):selected_forecast([dict(p=1.1,feasible=False)])
    def test_sign_single(self):self.assertEqual(sign_p(1,0),1.)


class PlanTests(unittest.TestCase):
    def test_full_budget(self):
        from online_confirm import plan
        p=plan('full');self.assertEqual((len(p['units']),p['trials'],p['total_math_calls']),(64,384,12288))
        self.assertEqual((p['initial_calls'],p['online_calls']),(3072,9216))
    def test_smoke_and_ci(self):
        from online_confirm import plan
        self.assertEqual(plan('smoke')['total_math_calls'],264);self.assertEqual(plan('ci')['total_math_calls'],120)
    def test_maxima(self):
        from online_confirm import plan
        p=plan('full');self.assertEqual((p['max_gp_heads'],p['max_gp_kernels']),(11968,95744))
    def test_models_and_budgets_unchanged(self):
        from online_confirm import plan
        from c1_protocol import plan as old
        a,b=plan('full'),old('full')
        for k in ('arms','initial','pool','steps','contrasts','checkpoints','primary','frozen','cold'):
            self.assertEqual(a[k],b[k])
    def test_seed_streams_disjoint(self):
        from online_confirm import plan
        from c1_protocol import plan as old
        used=set()
        for fn in (old,plan):
            for profile in ('ci','smoke','full'):
                p=fn(profile)
                seeds={u['seed']+t for u in p['units'] for t in range(p['steps']+1)}
                self.assertFalse(seeds & used);used |= seeds
    def test_plan_scope_restored(self):
        import c1_study,c1_verify
        from online_confirm import plan_scope
        a,b=c1_study.plan,c1_verify.plan
        with plan_scope():self.assertEqual(len(c1_study.plan('full')['units']),64)
        self.assertIs(c1_study.plan,a);self.assertIs(c1_verify.plan,b)
    def test_no_model_library_import_on_plan(self):
        code="from online_confirm import plan; plan('full'); import sys; assert not any(k in sys.modules for k in ['numpy','scipy','torch','gpytorch'])"
        subprocess.run([sys.executable,'-S','-c',code],check=True)
    def test_old_cli_untouched(self):
        root=Path(__file__).resolve().parents[1]
        names=['c1_protocol.py','c1_native.py','c1_study.py','c1_verify.py','check_c1_evidence.py','run_c1_online.py','risk_learning','ra_lsbo']
        changed=subprocess.check_output(['git','diff','--name-only','62b46f7a30ab5c3475cd59b86c7f3ad51974a1da','HEAD','--']+names,cwd=root,text=True)
        self.assertFalse(changed.strip())
    def test_unknown_profile(self):
        from online_confirm import plan
        with self.assertRaises(ValueError):plan('surprise')
    def test_primary_four_task_family(self):
        from online_confirm import plan
        p=plan('full');self.assertEqual(p['primary_inference']['family'],['tradeoff_s0','tradeoff_s1','tradeoff_s2','tradeoff_s3'])
        self.assertTrue(p['primary_inference']['no_pooling_with_pilot'])
    def test_reserved_not_evaluated(self):
        from online_confirm import plan
        self.assertFalse(plan('full')['retained_task_s4_s7'])
    def test_new_paths_do_not_trigger_old_pr_workflows(self):
        root=Path(__file__).resolve().parents[1]
        names=['online_confirm.py','online_confirm_stats.py','run_online_confirmation.py',
               'tests/test_online_confirm_core.py','docs/C1R_REPLICATION_PROTOCOL.txt',
               '.github/workflows/online-confirmation.yml']
        for f in (root/'.github/workflows').glob('*.yml'):
            if f.name=='online-confirmation.yml':continue
            text=f.read_text(encoding='utf-8')
            event=re.search(r'(?ms)^  pull_request:[ \t]*\n(.*?)(?=^  \S|^\S|\Z)',text)
            if not event:continue
            paths=re.search(r'(?m)^    paths:[ \t]*\n((?:      .*\n)+)',event.group(1))
            patterns=([x.strip().strip("'\"") for x in re.findall(r'^      - (.+)$',paths.group(1),re.M)] if paths else [])
            for name in names:
                self.assertFalse(any(fnmatch.fnmatchcase(name,p) for p in patterns),str(f)+': '+name)


class AnalysisTests(unittest.TestCase):
    def data(self):
        units=[]
        for r,warm in ((0,True),(1,False)):
            trials={}
            for a in ('a','b'):
                initial=10. if warm else None
                final=(2. if a=='a' else 4.) if warm else None
                trials[a]=dict(initial_warm=warm,initial_best=initial,best_curve=[initial,final],
                    information_curve=[dict(feasible=int(warm),exact_objectives=int(warm)),
                                       dict(feasible=2*int(warm),exact_objectives=2*int(warm))],
                    modes=['ei'],candidate_path=[a+str(r)])
            units.append(dict(task='x',repeat=r,trials=trials,
                online_candidates={a:[dict(branch=2)] for a in ('a','b')},
                selected_forecasts={a:[dict(p=.5,feasible=warm)] for a in ('a','b')}))
        spec=dict(arms=['a','b'],units=[dict(task='x',repeat=r) for r in (0,1)],
                  checkpoints=[1],steps=1,contrasts=[['a','b']])
        return units,spec
    def test_warm_stratified(self):
        u,s=self.data();r=analyze_units(u,s)
        self.assertEqual(r['primary']['x']['wins_losses_ties'],[1,0,0])
    def test_cold_not_imputed(self):
        u,s=self.data();r=analyze_units(u,s)
        self.assertIsNone(r['by_task']['x/k1/initial_cold']['comparisons']['a_vs_b']['mean_delta'])
    def test_online_coverage_not_initial(self):
        u,s=self.data();r=analyze_units(u,s)
        self.assertEqual(r['by_task']['x/k1/all']['arms']['a']['online_only_branch_counts'],[1,1])
    def test_pilot_not_confirmatory(self):
        u,s=self.data();r=analyze_units(u,s)
        self.assertFalse(r['inference']['confirmatory'])
    def test_failure_delta(self):
        u,s=self.data();r=analyze_units(u,s)
        self.assertEqual(r['primary']['x']['paired'][0]['failure_delta'],0)
    def test_duplicate_units(self):
        u,s=self.data()
        with self.assertRaises(ValueError):analyze_units(u+u[:1],s)
    def test_no_pooled_mean(self):
        u,s=self.data();r=analyze_units(u,s)
        self.assertNotIn('mean_delta',r);self.assertTrue(r['inference']['no_pooling_with_pilot'])
    def test_stats_deterministic(self):
        u,s=self.data();self.assertEqual(analyze_units(u,s),analyze_units(u,s))


if __name__=='__main__':unittest.main()
