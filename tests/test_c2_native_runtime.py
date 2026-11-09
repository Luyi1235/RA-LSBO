import copy,unittest
from c2_native import fit_composite
from c1_native import fit
from c1_protocol import bill
from risk_learning.core import observed

def rows(n=18):
 out=[]
 for i in range(n):
  x=.05+.9*i/(n-1);c=dict(branch=i%2,x=x,y=.3+.1*(i%3),trim=.4+.1*(i%2))
  target=dict(region=x-.52,resource=-.2 if i%3 else .1,objective=(x-.31)**2+.2,feasible=(x<=.52 and i%3!=0))
  out.append(observed(c,target))
 return out

class RuntimeComposerTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.data=rows();cls.b=bill();cls.frozen,cls.initial=fit(copy.deepcopy(cls.data),1701,cls.b)
  if cls.frozen is None:raise RuntimeError("Initial native C1 composite failed")
 def test_00_reuses_both_heads(self):
  m,r=fit_composite(self.data,1702,self.b,update_risk=False,update_objective=False,frozen=self.frozen)
  for k in ("region","resource","objective"):self.assertIs(m.heads[k],self.frozen.heads[k])
 def test_10_updates_risk_only(self):
  m,r=fit_composite(self.data+[self.data[0]],1703,self.b,update_risk=True,update_objective=False,frozen=self.frozen)
  self.assertIsNot(m.heads["region"],self.frozen.heads["region"])
  self.assertIsNot(m.heads["resource"],self.frozen.heads["resource"])
  self.assertIs(m.heads["objective"],self.frozen.heads["objective"])
 def test_01_updates_objective_only(self):
  m,r=fit_composite(self.data+[self.data[1]],1704,self.b,update_risk=False,update_objective=True,frozen=self.frozen)
  self.assertIs(m.heads["region"],self.frozen.heads["region"])
  self.assertIs(m.heads["resource"],self.frozen.heads["resource"])
  self.assertIsNot(m.heads["objective"],self.frozen.heads["objective"])
 def test_11_updates_all_heads(self):
  m,r=fit_composite(self.data+[self.data[2]],1705,self.b,update_risk=True,update_objective=True,frozen=self.frozen)
  for k in ("region","resource","objective"):self.assertIsNot(m.heads[k],self.frozen.heads[k])
 def test_prediction_composition(self):
  m,_=fit_composite(self.data,1706,self.b,update_risk=True,update_objective=True,frozen=self.frozen)
  p,e=m.predict([dict(branch=0,x=.2,y=.3,trim=.5),dict(branch=1,x=.8,y=.4,trim=.5)],self.b)
  self.assertIsNone(e);self.assertEqual(set(p["probabilities"]),{"region","resource","joint"})
  self.assertIsNotNone(p["objective"])
if __name__=="__main__":unittest.main()
