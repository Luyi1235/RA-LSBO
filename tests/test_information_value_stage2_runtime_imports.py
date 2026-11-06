"""Real import/CLI regressions: installed scientific dependencies are mandatory.

No sys.modules stand-ins or skip-on-missing-dependency behavior. These tests
perform no model fit or evaluator call and do not establish BO performance.
"""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Stage2RuntimeImportTests(unittest.TestCase):
    def child(self, source):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)],
                                cwd=str(ROOT), env=env, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        return result.stdout

    def test_engine_import_uses_real_shared_acquisition(self):
        output = self.child('''
            from ra_lsbo.experiments.information_value_stage2 import engine
            from ra_lsbo.soc_revision.acquisition import quantities
            from ra_lsbo.experiments.risk_budget63_local_value_engine import LocalCredibilityValueBO
            assert engine.quantities is quantities
            assert issubclass(engine.AllSupportSoftGPBO, LocalCredibilityValueBO)
            print("REAL_ENGINE_IMPORT_OK")
        ''')
        self.assertIn("REAL_ENGINE_IMPORT_OK", output)

    def test_runner_and_entry_import_without_running_experiments(self):
        output = self.child('''
            from contextlib import ExitStack
            from importlib import import_module
            from unittest.mock import patch
            from ra_lsbo.experiments.core.problems import MathematicalCase, AircraftCase
            from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
            from ra_lsbo.experiments.risk_budget63_local_value_engine import LocalCredibilityValueBO
            with ExitStack() as stack:
                guards = [stack.enter_context(patch.object(cls, "evaluate",
                    side_effect=AssertionError("No evaluation allowed in import test")))
                    for cls in (MathematicalCase, AircraftCase, SimplifiedElectricAircraft)]
                guards.append(stack.enter_context(patch.object(LocalCredibilityValueBO, "__init__",
                    side_effect=AssertionError("No optimizer construction allowed"))))
                runner = import_module("ra_lsbo.experiments.information_value_stage2.runner")
                entry = import_module("run_ra_lsbo_information_stage2_pycharm")
                assert entry.run is runner.run
                assert callable(entry.main)
                for guard in guards:
                    guard.assert_not_called()
            print("REAL_RUNNER_AND_ENTRY_IMPORT_OK_ZERO_EVALUATIONS")
        ''')
        self.assertIn("REAL_RUNNER_AND_ENTRY_IMPORT_OK_ZERO_EVALUATIONS", output)

    def test_cli_help_does_not_dispatch_a_run(self):
        output = self.child('''
            import runpy
            import sys
            from unittest.mock import patch
            from ra_lsbo.experiments.information_value_stage2 import runner
            with patch.object(runner, "run", side_effect=AssertionError("Help must not start a run")) as guard:
                sys.argv = ["run_ra_lsbo_information_stage2_pycharm.py", "--help"]
                try:
                    runpy.run_path(sys.argv[0], run_name="__main__")
                except SystemExit as exc:
                    assert exc.code == 0, exc.code
                else:
                    raise AssertionError("argparse help did not exit")
                guard.assert_not_called()
            print("CLI_HELP_OK_ZERO_EVALUATIONS")
        ''')
        self.assertIn("--profile", output)
        self.assertIn("--output", output)
        self.assertIn("CLI_HELP_OK_ZERO_EVALUATIONS", output)

    def test_previous_import_bug_is_detected_in_memory(self):
        output = self.child('''
            from pathlib import Path
            package = "ra_lsbo.experiments.information_value_stage2"
            path = Path("ra_lsbo/experiments/information_value_stage2/engine.py")
            source = path.read_text(encoding="utf-8")
            good = "from ...soc_revision.acquisition import quantities"
            bad = "from ..soc_revision.acquisition import quantities"
            assert source.count(good) == 1
            old_source = source.replace(good, bad, 1)
            namespace = {"__name__": package + ".regression_probe", "__package__": package}
            try:
                exec(compile(old_source, "<old-import-regression>", "exec"), namespace)
            except ModuleNotFoundError as exc:
                assert exc.name == "ra_lsbo.experiments.soc_revision", exc.name
            else:
                raise AssertionError("Original bad import unexpectedly succeeded")
            assert path.read_text(encoding="utf-8") == source
            print("OLD_IMPORT_REJECTED_NO_SOURCE_WRITES")
        ''')
        self.assertIn("OLD_IMPORT_REJECTED_NO_SOURCE_WRITES", output)


if __name__ == "__main__":
    unittest.main()
