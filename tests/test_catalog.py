import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ra_lsbo.catalog import catalog_runs, export_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        source = self.sources / "one"
        source.mkdir(parents=True)
        manifest = {
            "problem_config": {"version": "one"},
            "required_constraints": ["g"], "analytical_constraints": [],
            "design_variables": [{"name": "x", "kind": "continuous", "bounds": [0, 1]}],
            "source_sha256": {"problems/demo.py": "frozen"},
        }
        row = {"evaluation": 1, "candidate": {"continuous": {"x": .4}, "discrete": {}},
               "result": {"status": "ok", "solver_success": True, "objective": 1.,
                          "constraints": {"g": -1.}}}
        for name, value in (("manifest.json", manifest), ("evaluations.jsonl", row),
                            ("summary.json", {"status": "complete", "evaluations": 1})):
            (source / name).write_text(json.dumps(value), encoding="utf-8")

    def test_copies_not_counted_twice_and_sources_not_modified(self):
        shutil.copytree(self.sources / "one", self.sources / "two")
        before = {str(p): p.read_bytes() for p in self.sources.rglob("*") if p.is_file()}
        result = catalog_runs([self.sources, self.sources])
        self.assertEqual(result["unique_terminal_runs"], 1)
        self.assertEqual(result["unique_evaluations"], 1)
        self.assertEqual([e["status"] for e in result["entries"]], ["accepted", "duplicate"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.sources.rglob("*") if p.is_file()})

    def test_changed_physics_creates_a_separate_group(self):
        shutil.copytree(self.sources / "one", self.sources / "two")
        path = self.sources / "two" / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["problem_config"]["version"] = "two"
        path.write_text(json.dumps(manifest))
        self.assertEqual(len(catalog_runs([self.sources])["compatible_groups"]), 2)

    def test_running_copy_is_excluded_not_counted_as_terminal(self):
        (self.sources / "one" / "summary.json").write_text('{"status":"running","evaluations":1}')
        result = catalog_runs([self.sources])
        self.assertEqual(result["unique_terminal_runs"], 0)
        self.assertEqual(result["entries"][0]["status"], "excluded")

    def test_export_inside_source_is_rejected(self):
        with self.assertRaises(ValueError):
            export_catalog([self.sources], self.sources / "output")


if __name__ == "__main__":
    unittest.main()
