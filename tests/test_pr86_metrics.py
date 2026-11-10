import unittest
from pr86_metrics import metrics
from pr86_study import _score
class MetricTests(unittest.TestCase):
 def test_single_class_auc_null(self):
  x=metrics([True,True],[.8,.4]);self.assertIsNone(x["auroc"]);self.assertIsNotNone(x["auprc"])
 def test_perfect_classification(self):
  x=metrics([True,False],[.9,.1]);self.assertEqual(x["auroc"],1.);self.assertAlmostEqual(x["brier"],.01)
 def test_ece(self):
  x=metrics([True,False],[.8,.2]);self.assertAlmostEqual(x["ece"],.2)
 def test_bad_probability(self):
  with self.assertRaises(ValueError):metrics([True],[1.1])
 def test_bad_alignment(self):
  with self.assertRaises(ValueError):metrics([True,False],[.2])
if __name__=="__main__":unittest.main()
