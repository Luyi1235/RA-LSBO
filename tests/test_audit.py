import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ra_lsbo.audit import build_snapshot, export_snapshot


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.manifest = {
            "backend": "demo", "problem_config": {"version": "test_v1"},
            "required_constraints": ["soc", "span_min"], "analytical_constraints": ["span_min"],
            "design_variables": [
                {"name": "x", "kind": "continuous", "bounds": [0, 1]},
                {"name": "layout", "kind": "nominal", "values": ["a", "b"]},
            ],
            "source_sha256": {"problems/demo.py": "abc", "lsbo/optimizer.py": "def"},
        }
        self.sample = {
            "evaluation": 1, "candidate": {"continuous": {"x": 0.4}, "discrete": {"layout": "a"}},
            "result": {"objective": 1.0, "solver_success": True, "status": "ok",
                       "constraints": {"soc": -0.1, "span_min": -1}},
            "feasible": True, "phase": "initial",
        }
        self.write("manifest.json", self.manifest)
        self.write("summary.json", {"status": "complete", "evaluations": 1})
        self.write("evaluations.jsonl", self.sample)

    def write(self, name, data):
        (self.source / name).write_text(json.dumps(data) + "\n", encoding="utf-8")

    def test_export_keeps_all_source_bytes_and_has_separate_heads(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        destination, report = export_snapshot(self.source, self.root / "output")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual(report["valid_objectives"], 1)
        self.assertTrue(report["evaluator_hashes_present"])
        self.assertEqual(set(p.name for p in destination.iterdir()),
                         {"report.json", "training_rows.json", "observations.jsonl", "evaluator_contract.json"})
        tables = json.loads((destination / "training_rows.json").read_text())
        self.assertNotIn("span_min", tables["constraints"])
        self.assertEqual(tables["objective"][0]["source_journal_sha256"],
                         report["source_sha256"]["evaluations.jsonl"])

    def test_never_exports_inside_source(self):
        for path in (self.source, self.source / "child"):
            with self.assertRaises(ValueError):
                export_snapshot(self.source, path)

    def test_running_run_rejected(self):
        self.write("summary.json", {"status": "running", "evaluations": 1})
        with self.assertRaisesRegex(ValueError, "running"):
            build_snapshot(self.source)

    def test_stopped_run_can_be_audited_not_called_complete(self):
        self.write("summary.json", {"status": "stopped_with_error", "evaluations": 1})
        self.assertEqual(build_snapshot(self.source)[2]["source_status"], "stopped_with_error")

    def test_inconsistent_count_rejected(self):
        self.write("summary.json", {"status": "complete", "evaluations": 2})
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            build_snapshot(self.source)

    def test_duplicate_indices_rejected(self):
        with (self.source / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.sample) + "\n")
        with self.assertRaisesRegex(ValueError, "indices"):
            build_snapshot(self.source)

    def test_manifest_difference_changes_contract(self):
        before = build_snapshot(self.source)[2]["evaluator_contract_sha256"]
        self.manifest["problem_config"]["version"] = "test_v2"
        self.write("manifest.json", self.manifest)
        self.assertNotEqual(before, build_snapshot(self.source)[2]["evaluator_contract_sha256"])

    def test_candidate_schema_mismatch_rejected(self):
        self.sample["candidate"]["continuous"]["extra"] = 1
        self.write("evaluations.jsonl", self.sample)
        with self.assertRaisesRegex(ValueError, "schema"):
            build_snapshot(self.source)

    def test_nonstandard_json_rejected(self):
        self.sample["result"]["objective"] = float("nan")
        self.write("evaluations.jsonl", self.sample)
        with self.assertRaisesRegex(ValueError, "Nonstandard"):
            build_snapshot(self.source)

    def test_duplicate_json_fields_rejected(self):
        (self.source / "summary.json").write_text(
            '{"status":"complete","status":"running","evaluations":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            build_snapshot(self.source)

    def test_optimizer_hash_not_confused_with_evaluator_identity(self):
        before = build_snapshot(self.source)[2]["evaluator_contract_sha256"]
        self.manifest["source_sha256"]["lsbo/optimizer.py"] = "updated"
        self.write("manifest.json", self.manifest)
        self.assertEqual(before, build_snapshot(self.source)[2]["evaluator_contract_sha256"])
        self.manifest["source_sha256"]["problems/demo.py"] = "new_physics"
        self.write("manifest.json", self.manifest)
        self.assertNotEqual(before, build_snapshot(self.source)[2]["evaluator_contract_sha256"])

    def test_source_mutation_rejected_before_any_export(self):
        original = Path.read_bytes
        reads = {}
        def changing(path):
            reads[path.name] = reads.get(path.name, 0) + 1
            data = original(path)
            if path.name == "evaluations.jsonl" and reads[path.name] > 1:
                return data + b"\n"
            return data
        with patch.object(Path, "read_bytes", changing):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                export_snapshot(self.source, self.root / "output")
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
