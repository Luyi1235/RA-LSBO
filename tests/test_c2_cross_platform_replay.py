import unittest
from c2_verify import same_decision_record
class CrossPlatformDecisionTests(unittest.TestCase):
 def setUp(self):
  self.a=dict(mode="risk_ei",selected_index=7,selected_candidate_id="abc",
   incumbent=1.25,scores=[.1,.2,.3])
 def test_tiny_float_drift_is_accepted(self):
  b=dict(self.a);b["scores"]=[.1+1e-12,.2-1e-12,.3];b["incumbent"]=1.25+1e-12
  self.assertTrue(same_decision_record(self.a,b))
 def test_selected_index_must_match_exactly(self):
  b=dict(self.a);b["selected_index"]=8
  self.assertFalse(same_decision_record(self.a,b))
 def test_selected_candidate_must_match(self):
  b=dict(self.a);b["selected_candidate_id"]="different"
  self.assertFalse(same_decision_record(self.a,b))
 def test_material_score_drift_rejected(self):
  b=dict(self.a);b["scores"]=[.1,.2,.30001]
  self.assertFalse(same_decision_record(self.a,b))
 def test_nonfinite_score_rejected(self):
  b=dict(self.a);b["scores"]=[.1,.2,float("nan")]
  self.assertFalse(same_decision_record(self.a,b))
if __name__=="__main__":unittest.main()
