import unittest
from unittest.mock import patch
from types import SimpleNamespace
from c2_native import fit_composite

class Head:
 def __init__(self,name): self.name=name;self.identity="id-"+name

class ComposerTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{"candidate":{"x":1},"candidate_id":"a"*64}]
  self.bill={}
  self.frozen=SimpleNamespace(heads={k:Head(k) for k in ("region","resource","objective")})
 def fake(self,rows,name,seed,bill):
  return Head(name),dict(training_sha256="prefix",head={"identity":"new-"+name})
 def test_risk_only_updates_risk_reuses_objective(self):
  with patch("c2_native._head",side_effect=self.fake) as fit:
   model,receipt=fit_composite(self.rows,10,self.bill,update_risk=True,update_objective=False,frozen=self.frozen)
  self.assertIs(model.heads["objective"],self.frozen.heads["objective"])
  self.assertEqual(fit.call_count,2)
  self.assertEqual(receipt["heads"]["objective"]["mode"],"frozen")
 def test_objective_only_reuses_risk(self):
  with patch("c2_native._head",side_effect=self.fake) as fit:
   model,_=fit_composite(self.rows,10,self.bill,update_risk=False,update_objective=True,frozen=self.frozen)
  self.assertIs(model.heads["region"],self.frozen.heads["region"])
  self.assertIs(model.heads["resource"],self.frozen.heads["resource"])
  self.assertEqual(fit.call_count,1)
 def test_unavailable_objective_not_fabricated(self):
  def no_objective(rows,name,seed,bill):
   if name=="objective":return None,dict(status="NO_EXACT_OBJECTIVE")
   return Head(name),dict(training_sha256="prefix",head={"identity":"new-"+name})
  with patch("c2_native._head",side_effect=no_objective):
   model,receipt=fit_composite(self.rows,10,self.bill,update_risk=True,update_objective=True,frozen=None)
  self.assertNotIn("objective",model.heads)
  self.assertEqual(receipt["heads"]["objective"]["mode"],"unavailable_no_exact_labels")
 def test_initial_requires_both_factors(self):
  with self.assertRaises(ValueError):
   fit_composite(self.rows,10,self.bill,update_risk=True,update_objective=False)
if __name__=="__main__":unittest.main()
