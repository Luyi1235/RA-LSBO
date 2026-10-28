from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import json

from ra_lsbo.preflight import differences, inspect_x57


class PreflightTests(unittest.TestCase):
    def test_nested_physical_changes_are_not_hidden_by_backend_name(self):
        before = {"backend": "same", "mission": {"rate": 500., "version": "v2"}}
        after = {"backend": "same", "mission": {"rate": 450., "version": "v3"}}
        result = differences(before, after)
        self.assertEqual({row["field"] for row in result}, {"mission.rate", "mission.version"})

    def test_missing_and_extra_config_fields(self):
        self.assertEqual(len(differences({"a": 1}, {"b": 1})), 2)
        self.assertEqual(differences({"a": 1}, {"a": 1}), [])

    def test_changed_evaluator_blocks_reuse_without_importing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base = root / "source", root / "base"
            source.mkdir()
            (base / "problems").mkdir(parents=True)
            (source / "manifest.json").write_bytes(b"original")
            (base / "problems" / "engine.py").write_text("raise AssertionError('must not import')")
            config = root / "config.json"
            config.write_text(json.dumps({"mission": {"version": "v3"}}))
            audit = {
                "source_sha256": {"manifest.json": hashlib.sha256(b"original").hexdigest()},
                "evaluator_contract_sha256": "test",
            }
            contract = {
                "problem_config": {"mission": {"version": "v2"}},
                "evaluator_source_sha256": {"problems/engine.py": "oldhash"},
            }
            with patch("ra_lsbo.preflight.build_snapshot", return_value=([], {}, audit, contract)):
                report = inspect_x57(source, base, config)
            self.assertFalse(report["historical_data_reusable_without_revalidation"])
            self.assertFalse(report["real_mda_enabled"])
            self.assertIn("evaluator_source_changed", report["readiness_issues"])
            self.assertIn("physical_config_changed", report["readiness_issues"])


if __name__ == "__main__":
    unittest.main()
