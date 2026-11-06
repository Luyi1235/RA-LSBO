from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pr69_verify', ROOT/'pr69_evidence_verify.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def record(abort=False, resource=-.2):
    candidate = {'continuous': {'x': .5}, 'discrete': {'branch': 1}}
    region = {'kind': 'interval', 'lower': 0., 'lower_inclusive': False, 'upper': None, 'value': None} if abort else {'kind': 'exact', 'value': -.1}
    objective = {'kind': 'missing', 'value': None} if abort else {'kind': 'exact', 'value': 3.}
    return {'candidate': candidate, 'diagnostics': {}, 'observation': {
        'status': 'completed', 'design_id': v.native_digest(candidate), 'objective': objective,
        'constraints': [['region', region], ['resource', {'kind': 'exact', 'value': resource}]]}}


class ContractTests(unittest.TestCase):
    def test_censored_abort_not_imputed(self):
        self.assertFalse(v.feasible(record(True)))
        self.assertIsNone(v.best([record(True)]))

    def test_completed_resource_infeasible_not_deliverable(self):
        self.assertFalse(v.feasible(record(resource=.2)))
        self.assertTrue(v.feasible(record()))
        self.assertEqual(v.best([record(True), record()]), 3.)

    def test_wrong_design_identity_rejected(self):
        r = record()
        r['candidate']['continuous']['x'] = .6
        with self.assertRaises(ValueError):
            v.feasible(r)

    def test_abort_hidden_objective_rejected(self):
        r = record(True)
        r['observation']['objective']['value'] = 5.
        with self.assertRaises(ValueError):
            v.feasible(r)


if __name__ == '__main__':
    unittest.main()
