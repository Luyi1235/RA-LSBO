import math, tempfile, unittest
from pathlib import Path
from pr84_c2_audit import binary,auc,verify_manifest,require

class BinaryMetricTests(unittest.TestCase):
 def test_brier_and_logloss(self):
  r=binary([True,False],[.8,.2])
  self.assertAlmostEqual(r["brier"],.04)
  self.assertIsNotNone(r["log_loss"])
  self.assertEqual(r["auroc"],1.0)
 def test_single_class_auc_null(self):
  r=binary([True,True],[.9,.4])
  self.assertIsNone(r["auroc"])
 def test_empty_metrics_null(self):
  r=binary([],[])
  self.assertIsNone(r["brier"])
  self.assertIsNone(r["auroc"])
 def test_auc_tie_half(self):
  self.assertEqual(auc([True,False],[.5,.5]),.5)
 def test_invalid_probability(self):
  with self.assertRaises(ValueError):binary([True],[float("nan")])
 def test_manifest_detects_tamper(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"full").mkdir();(root/"full"/"x").write_text("a")
   import hashlib,json
   h=hashlib.sha256(b"a").hexdigest()
   (root/"full"/"manifest.json").write_text(json.dumps({"schema":"sha256_manifest_v1","files":{"x":h}}))
   self.assertEqual(verify_manifest(root)[0],1)
   (root/"full"/"x").write_text("b")
   with self.assertRaises(ValueError):verify_manifest(root)
if __name__=="__main__":unittest.main()
