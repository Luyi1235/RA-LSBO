"""Standard-library path checks; do not import or stub scientific modules."""
import ast
from importlib.util import resolve_name
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ra_lsbo.experiments.information_value_stage2"
FOLDER = ROOT / "ra_lsbo" / "experiments" / "information_value_stage2"


def module_exists(name):
    path = ROOT.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


class Stage2ImportPathTests(unittest.TestCase):
    def test_acquisition_import_resolves_to_real_top_level_package(self):
        tree = ast.parse((FOLDER / "engine.py").read_text(encoding="utf-8"))
        nodes = [node for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and
                 node.module == "soc_revision.acquisition"]
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(resolve_name("." * node.level + node.module, PACKAGE),
                         "ra_lsbo.soc_revision.acquisition")

    def test_stage2_relative_import_targets_exist(self):
        files = sorted(FOLDER.glob("*.py"))
        self.assertGreaterEqual(len(files), 4)
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    target = resolve_name("." * node.level + (node.module or ""), PACKAGE)
                    with self.subTest(source=path.name, line=node.lineno, target=target):
                        self.assertTrue(module_exists(target), "Missing repository module: " + target)

    def test_previous_two_dot_spelling_is_the_missing_module(self):
        target = resolve_name("..soc_revision.acquisition", PACKAGE)
        self.assertEqual(target, "ra_lsbo.experiments.soc_revision.acquisition")
        self.assertFalse(module_exists(target))


if __name__ == "__main__":
    unittest.main()
