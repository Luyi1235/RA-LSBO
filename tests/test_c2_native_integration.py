import copy,unittest
from c1_native import fit as fit_all
from c1_protocol import bill
from c2_native import fit_composite
from risk_learning.core import observed

def rows(n=16):
 out=[]
 for i in range(n):
  x=.05+.9*i/(n-1);c=dict(branch=i%4,x=x,y=.3,trim=.5 if i%4==0 else .4)
  target=dict(region=x-.55,resource=-.2,objective=(x-.3)**2+.2,feasible=x<=.55)
  out.append(observed(c,target))
 return out

class C2NativeIntegration(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.data=rows();cls.b=bill()
  cls.frozen,cls.initial=fit_all(copy.deepcopy(cls.data),773,cls.b,risk_only=False)
  if cls.frozen is None: raise RuntimeError("Initial native fit failed")
 def test_initial_has_three_heads(self):
  self.assertEqual(set(self.frozen.heads),{"region","resource","objective"})
 def test_risk_only_factor_reuses_objective(self):
  b=bill();m,r=fit_composite(self.data,774,b,update_risk=True,update_objective=False,frozen=self.frozen)
  self.assertIs(m.heads["objective"],self.frozen.heads["objective"])
  self.assertNotEqual(m.heads["region"].identity,self.frozen.heads["region"].identity)
  self.assertEqual(r["heads"]["objective"]["mode"],"frozen")
 def test_objective_only_factor_reuses_risk(self):
  b=bill();m,r=fit_composite(self.data,775,b,update_risk=False,update_objective=True,frozen=self.frozen)
  for h in ("region","resource"):self.assertIs(m.heads[h],self.frozen.heads[h])
  self.assertEqual(r["heads"]["region"]["mode"],"frozen")
 def test_double_update_changes_all_heads(self):
  b=bill();m,r=fit_composite(self.data,776,b,update_risk=True,update_objective=True,frozen=self.frozen)
  for h in ("region","resource","objective"):
   self.assertNotEqual(m.heads[h].identity,self.frozen.heads[h].identity)
 def test_prediction_composes_joint_risk_and_objective(self):
  q=[dict(branch=0,x=.2,y=.3,trim=.5),dict(branch=1,x=.8,y=.7,trim=.4)]
  p,e=self.frozen.predict(q,bill());self.assertIsNone(e)
  self.assertEqual(len(p["probabilities"]["joint"]),2)
  self.assertIsNotNone(p["objective"])
 def test_objective_unavailable_is_not_fabricated(self):
  data=copy.deepcopy(self.data)
  for r in data:r["objective"]=dict(kind="missing",reason="controlled_abort")
  b=bill()
  model,receipt=fit_composite(data,778,b,update_risk=True,update_objective=True,frozen=None)
  self.assertNotIn("objective",model.heads)
  self.assertEqual(receipt["heads"]["objective"]["mode"],"unavailable_no_exact_labels")
if __name__=="__main__":unittest.main()
