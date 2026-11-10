import json,unittest
from pr86_protocol import plan,sample_orders,coverage
from pr86_metrics import metrics
from risk_learning.core import candidates,truth,observed
class ProtocolTests(unittest.TestCase):
 def test_budgets(self):
  self.assertEqual(plan("ci")["total_math_calls"],512)
  self.assertEqual(plan("smoke")["total_math_calls"],1920)
  self.assertEqual(plan("full")["total_math_calls"],12672)
  self.assertEqual(plan("full")["model_rows"],720)
  self.assertEqual(plan("full")["max_gp_heads"],1440)
  self.assertEqual(plan("full")["max_gp_kernels"],11520)
 def test_json_roundtrip(self):
  for p in ("ci","smoke","full"):self.assertEqual(json.loads(json.dumps(plan(p))),plan(p))
 def test_seed_blocks_disjoint_from_phase_a_and_c2(self):
  from risk_learning.core import plan as old
  from c2_protocol import plan as c2
  a={u["train_seed"]+k for p in ("ci","smoke","full") for u in plan(p)["units"] for k in (0,1)}
  b={u[s] for p in ("ci","smoke","full") for u in old(p)["units"] for s in ("train_seed","test_seed")}
  c={u["seed"] for p in ("ci","smoke","full") for u in c2(p)["units"]}
  self.assertFalse(a&b);self.assertFalse(a&c)
 def test_sampling_orders_are_permutations_and_nested(self):
  pool=candidates(1234,192);rows=[observed(x,truth("tradeoff_s0",x)) for x in pool]
  orders,_=sample_orders(pool,rows,99)
  for name,order in orders.items():
   self.assertEqual(sorted(order),list(range(192)))
   for a,b in zip((12,24,48,96),(24,48,96,192)):
    self.assertTrue(set(order[:a]).issubset(order[:b]))
 def test_class_balance_uses_only_train_observations(self):
  pool=candidates(1234,192);rows=[observed(x,truth("tradeoff_s0",x)) for x in pool]
  a,_=sample_orders(pool,rows,99)
  labels=[all(rows[i][h]["kind"]=="exact" and rows[i][h]["value"]<=0 for h in ("region","resource")) for i in a["class_balanced"]]
  self.assertLessEqual(abs(sum(labels[:48])-24),1)
 def test_coverage_design_deterministic_and_label_blind(self):
  pool=candidates(4321,192);rows=[observed(x,truth("tradeoff_s1",x)) for x in pool]
  a,_=sample_orders(pool,rows,7)
  altered=[dict(r,region=dict(kind="interval",lower=0.,lower_inclusive=False)) for r in rows]
  b,_=sample_orders(pool,altered,7)
  self.assertEqual(a["coverage_maximin"],b["coverage_maximin"])
  self.assertEqual(a["coverage_maximin"],sample_orders(pool,rows,7)[0]["coverage_maximin"])
 def test_metric_single_class_null(self):
  x=metrics([True,True],[.8,.3]);self.assertIsNone(x["auroc"]);self.assertIsNotNone(x["auprc"])
 def test_metric_perfect(self):
  x=metrics([True,False],[.9,.1]);self.assertEqual(x["auroc"],1.);self.assertAlmostEqual(x["brier"],.01)
if __name__=="__main__":unittest.main()
