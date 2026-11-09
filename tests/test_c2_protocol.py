import unittest
from c2_protocol import plan,factorial,TASKS,ARMS,select_c2,digest
from c2_stats import summarize_factorial,holm_factorial
class PlanTests(unittest.TestCase):
 def test_full_budget(self):
  p=plan("full");self.assertEqual((len(p["units"]),p["trials"],p["total_math_calls"]),(64,384,12288))
  self.assertEqual((p["initial_calls"],p["online_calls"]),(3072,9216))
 def test_smoke_budget(self): self.assertEqual(plan("smoke")["total_math_calls"],264)
 def test_ci_budget(self): self.assertEqual(plan("ci")["total_math_calls"],120)
 def test_six_arms(self): self.assertEqual(len(ARMS),6)
 def test_full_gp_budget(self):
  p=plan("full");self.assertEqual((p["max_gp_heads"],p["max_gp_kernels"]),(11968,95744))
 def test_risk_ei_selection(self):
  ids=[digest({"i":i}) for i in range(3)]
  pred={"probabilities":{"joint":[.9,.5,.2]},"objective":{"mean":[1.,.5,.1],"variance":[0.,0.,0.]}}
  d=select_c2("risk_frozen_objective_frozen",ids,[0,1,2],pred,2.,["u",0])
  self.assertEqual(d["selected_index"],1)
 def test_cold_shared_hash(self):
  ids=[digest({"i":i}) for i in range(3)]
  pred={"probabilities":{"joint":[.9,.5,.2]},"objective":{"mean":[1.,.5,.1],"variance":[0.,0.,0.]}}
  a=select_c2("risk_frozen_objective_frozen",ids,[0,1,2],pred,None,["u",0])
  b=select_c2("risk_frozen_objective_updated",ids,[0,1,2],pred,None,["u",0])
  self.assertEqual(a["selected_index"],b["selected_index"])
  self.assertEqual(a["mode"],"cold_shared_hash")
 def test_seed_disjoint_from_c1r(self):
  from online_confirm import plan as old
  a={u["seed"]+t for p in ("ci","smoke","full") for u in plan(p)["units"] for t in range(plan(p)["steps"]+1)}
  b={u["seed"]+t for p in ("ci","smoke","full") for u in old(p)["units"] for t in range(old(p)["steps"]+1)}
  self.assertFalse(a&b)
 def test_factorial(self): self.assertEqual(factorial(10,8,7,4),dict(complete=True,risk=-2.5,objective=-3.5,interaction=-1))
 def test_missing_not_imputed(self): self.assertFalse(factorial(None,2,3,4)["complete"])
 def test_invalid_nan(self):
  with self.assertRaises(ValueError):factorial(1,float("nan"),2,3)
class StatsTests(unittest.TestCase):
 def test_summary(self):
  rows=[dict(repeat=i,effects=factorial(10,8,7,4)) for i in range(5)]
  s=summarize_factorial(rows);self.assertEqual(s["complete_four_cell"],5)
  self.assertEqual(s["effects"]["risk"]["wins_losses_ties"],[5,0,0])
 def test_duplicate(self):
  row=dict(repeat=0,effects=factorial(1,2,3,4))
  with self.assertRaises(ValueError):summarize_factorial([row,row])
 def test_holm_family(self):
  d={t:dict(effects={k:dict(two_sided_sign_p=.01) for k in ("risk","objective","interaction")}) for t in TASKS}
  self.assertEqual(len(holm_factorial(d)),12)
  with self.assertRaises(ValueError):holm_factorial({"x":d[TASKS[0]]})
if __name__=="__main__":unittest.main()
