"""Native model/coverage integration with synthetic observations, no true calls."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cold_coverage_core import CONTROL, TREATMENT, PERIOD
from cold_coverage_engine import ColdCoverageBO
from cold_coverage_verify import check
from ra_lsbo.experiments.information_value_stage1.runner import NativeAdapter
from ra_lsbo.experiments.information_value_stage3.native import FantasyValueBO
from ra_lsbo.experiments.discovery_stages.policies import coverage_pick, filter_near, COVERAGE_PERIOD
from ra_lsbo.experiments.core.problems import MathematicalCase
from ra_lsbo.partial_bo.models import EvidenceGP
from ra_lsbo.risk_core.observations import OutputObservation


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.api=NativeAdapter();self.case='branin_sparse';self.seed=999190
        self.p=self.api.problem(self.case)
        self.designs=self.api.initial(self.p,self.case,self.seed)
        self.guard=patch.object(MathematicalCase,'evaluate',side_effect=AssertionError('No test truth calls'))
        self.guard.start();self.addCleanup(self.guard.stop)
    def history(self,cold):
        rows=[]
        for i,c in enumerate(self.designs):
            objective=OutputObservation('missing','synthetic-f-'+str(i),reason='synthetic_abort') if cold else float(i+3)
            region=OutputObservation('interval','synthetic-g-'+str(i),lower=0.,lower_inclusive=False) if cold else -1.
            rows.append(self.p.record(c,objective,{'region':region,'resource':-.2},{'fixture':'synthetic_not_evaluated'}))
        return rows
    def engine(self,treatment,name):
        folder=Path(self.tmp.name)/name;folder.mkdir()
        if treatment:return ColdCoverageBO(self.p,self.api.base,self.seed,folder,horizon=2)
        return FantasyValueBO(self.p,self.api.base,self.seed,folder,arm=CONTROL,horizon=2)
    def cold_proposal(self):
        h=self.history(True);e=self.engine(True,'cold');pool=self.api.pool(self.p,self.case,self.seed,0,h)
        c,m=e.propose(h,pool,0);return h,e,pool,c,m
    def test_only_scope_setting_changed(self):
        a,b=self.engine(False,'a'),self.engine(True,'b')
        self.assertEqual(asdict(a.settings),asdict(b.settings));self.assertEqual(a.iv_arm,b.iv_arm)
        self.assertEqual((a.coverage_scope,b.coverage_scope),('none','cold'))
    def test_acquisition_method_is_inherited(self):
        self.assertIs(ColdCoverageBO._choose_acquisition,FantasyValueBO._choose_acquisition)
    def test_cadence_not_tuned(self):self.assertEqual(COVERAGE_PERIOD,PERIOD)
    def test_scheduled_choice_matches_original_rule(self):
        h,e,pool,c,m=self.cold_proposal();protected=filter_near(self.p,pool,h)[0]
        idx,rule=coverage_pick(self.p,protected,h,self.seed,0)
        self.assertEqual(c,protected[idx]);self.assertEqual(m['coverage_rule'],rule)
    def test_scheduled_call_has_no_gp_fit(self):
        with patch.object(EvidenceGP,'__init__',side_effect=AssertionError('GP must be skipped')):
            _,_,_,_,m=self.cold_proposal()
        self.assertTrue(m['posterior_fit_skipped'])
    def test_coverage_has_no_fake_forecast(self):
        h,e,pool,c,m=self.cold_proposal()
        self.assertNotIn('information_value_decision',m)
        d=check(m,c,[r.to_dict() for r in h],0,2,TREATMENT)
        self.assertFalse(d['forecast_available']);self.assertTrue(d['scheduled_coverage'])
    def test_warm_native_paths_and_forecasts_match(self):
        a,b=self.engine(False,'a'),self.engine(True,'b');h=self.history(False)
        for i in range(2):
            pool=self.api.pool(self.p,self.case,self.seed,i,h)
            ca,ma=a.propose(h,pool,i);cb,mb=b.propose(h,pool,i)
            self.assertEqual(ca,cb);self.assertEqual(ma['forecasts'],mb['forecasts'])
            self.assertEqual(ma['information_value_decision']['scores'],mb['information_value_decision']['scores'])
            self.assertIsNone(mb['coverage_reason'])
            check(mb,cb,[r.to_dict() for r in h],i,2,TREATMENT)
            rec=self.p.record(ca,2.-i,{'region':-1.,'resource':-.2},{'fixture':'synthetic'})
            a.accept(rec);b.accept(rec);h.append(rec)
    def test_discovery_disables_coverage(self):
        h,e,pool,c,m=self.cold_proposal()
        rec=self.p.record(c,2.,{'region':-1.,'resource':-.2},{'fixture':'synthetic_delivery'})
        e.accept(rec);h.append(rec);pool=self.api.pool(self.p,self.case,self.seed,1,h)
        c,m=e.propose(h,pool,1)
        self.assertFalse(m['coverage_active']);self.assertIsNone(m['coverage_reason'])
        self.assertEqual(m['information_value_decision']['depth'],1)
    def test_raw_checker_rejects_wrong_phase(self):
        h,e,pool,c,m=self.cold_proposal();m['cold_start']=False
        with self.assertRaises(ValueError):check(m,c,[r.to_dict() for r in h],0,2,TREATMENT)
    def test_raw_checker_rejects_scope_tamper(self):
        h,e,pool,c,m=self.cold_proposal();m['coverage_scope']='all'
        with self.assertRaises(ValueError):check(m,c,[r.to_dict() for r in h],0,2,TREATMENT)
    def test_raw_checker_rejects_fake_coverage_counts(self):
        h,e,pool,c,m=self.cold_proposal();m['coverage_rule']['selected_branch_visits']+=1
        with self.assertRaises(ValueError):check(m,c,[r.to_dict() for r in h],0,2,TREATMENT)
    def test_raw_checker_rejects_intervention_on_control(self):
        h,e,pool,c,m=self.cold_proposal()
        with self.assertRaises(ValueError):check(m,c,[r.to_dict() for r in h],0,2,CONTROL)


if __name__=='__main__':unittest.main()
