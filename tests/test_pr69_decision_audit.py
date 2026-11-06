from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pr69_audit', ROOT/'run_pr69_decision_audit.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def proposal(depth=2):
    pool = [{'x': 0.}, {'x': 1.}]
    base = [.6, .5]
    fs = []
    for i, value in ((0, .1), (1, .4)):
        for j in range(4):
            fs.append({'first_index': i, 'weight': .25,
                       'observation': {'aborted': True,
                           'region': {'kind': 'interval', 'lower': 0., 'lower_inclusive': False},
                           'resource': {'kind': 'exact', 'value': -.2},
                           'objective': {'kind': 'missing'}},
                       'second_index': 1-i, 'continuation': value})
    r = {'myopic_scores': base, 'scores': [.7, .9], 'selected_index': 1,
         'myopic_index': 0, 'first_action_indices': [0, 1], 'fantasies': fs,
         'criterion': 'first_feasible_probability', 'depth': depth, 'fallback': None}
    if depth == 1:
        r.update(scores=base, selected_index=0, first_action_indices=[], fantasies=[])
    return {'iteration': 0, 'candidate': pool[r['selected_index']], 'history_hash': 'abc',
            'metadata': {'pool': pool, 'forecasts': {'value': 1}, 'information_value_decision': r}}


class AuditTests(unittest.TestCase):
    def test_import_is_scientific_library_free(self):
        code = 'import runpy,sys; runpy.run_path(sys.argv[1]); assert not ({"torch", "numpy", "scipy"} & set(sys.modules))'
        proc = subprocess.run([sys.executable, '-S', '-c', code, str(ROOT/'run_pr69_decision_audit.py')], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_decomposition(self):
        r = a.decision_metrics(proposal(), [None, None, None])
        self.assertAlmostEqual(r['immediate_sacrifice'], .1)
        self.assertAlmostEqual(r['predicted_two_step_advantage'], .2)
        self.assertTrue(r['deviated_from_own_myopic'])

    def test_quadrature_deletion_constant_invariance(self):
        self.assertEqual(a.delete_one_node_choices(proposal()['metadata']['information_value_decision']), [1]*4)

    def test_quadrature_deletion_flags_fragility(self):
        p = proposal()
        r = p['metadata']['information_value_decision']
        for j, row in enumerate(r['fantasies'][4:]):
            row['continuation'] = 1.6 if j == 0 else 0.
        self.assertEqual(a.delete_one_node_choices(r), [0, 1, 1, 1])

    def test_broken_weight_rejected(self):
        p = proposal()
        p['metadata']['information_value_decision']['fantasies'][0]['weight'] = .4
        with self.assertRaises(ValueError):
            a.decision_metrics(p, [None]*3)

    def test_bad_score_rejected(self):
        p = proposal()
        p['metadata']['information_value_decision']['scores'][1] = .95
        with self.assertRaises(ValueError):
            a.decision_metrics(p, [None]*3)

    def test_hidden_abort_objective_rejected(self):
        p = proposal()
        p['metadata']['information_value_decision']['fantasies'][0]['observation']['objective'] = {'kind': 'exact', 'value': 2.}
        with self.assertRaises(ValueError):
            a.decision_metrics(p, [None]*3)

    def test_hidden_abort_region_rejected(self):
        p = proposal()
        p['metadata']['information_value_decision']['fantasies'][0]['observation']['region']['value'] = 1.
        with self.assertRaises(ValueError):
            a.decision_metrics(p, [None]*3)

    def test_repeated_second_action_rejected(self):
        p = proposal()
        p['metadata']['information_value_decision']['fantasies'][0]['second_index'] = 0
        with self.assertRaises(ValueError):
            a.decision_metrics(p, [None]*3)

    def test_last_step_has_no_fantasy(self):
        self.assertEqual(a.decision_metrics(proposal(1), [None]*3)['depth'], 1)

    def test_pool_retention_not_a_safety_measure(self):
        p, nxt = proposal(), proposal(1)
        r = a.decision_metrics(p, [None, None, 4.], nxt)
        self.assertEqual(r['next_pool_retention_mass'], 1.)
        self.assertEqual(r['realized_next_two_calls_delivery_or_gain'], 1)
        nxt['metadata']['pool'] = [{'x': .01}]
        self.assertEqual(a.decision_metrics(p, [None]*3, nxt)['next_pool_retention_mass'], 0.)

    def test_warm_window(self):
        p = proposal()
        p['metadata']['information_value_decision']['criterion'] = 'expected_feasible_objective_improvement'
        r = a.decision_metrics(p, [10., 10., 7.], proposal(1))
        self.assertEqual(r['realized_next_two_calls_delivery_or_gain'], 3.)

    def test_same_prefix_requires_forecast_match(self):
        p, q = proposal(), proposal()
        self.assertTrue(a.same_prefix(p, q))
        q['metadata']['forecasts']['value'] = 2
        self.assertFalse(a.same_prefix(p, q))

    def test_no_model_is_not_false_probability(self):
        self.assertIsNone(a.decision_metrics({'metadata': {}}, []))

    def test_risk_direction_is_explicit(self):
        rows = [{'forecast_available': True, 'predicted_failure': .9, 'nonfeasible': 1},
                {'forecast_available': True, 'predicted_failure': .1, 'nonfeasible': 0},
                {'forecast_available': False, 'predicted_failure': None, 'nonfeasible': 1}]
        r = a.risk_summary(rows)
        self.assertEqual(r['predicted_failure_ge_0_8']['failures'], 1)
        self.assertEqual(r['predicted_failure_le_0_2']['feasible'], 1)
        self.assertEqual(r['unpredicted_rows'], 1)

    def test_unknown_forecast_not_silently_imputed(self):
        with self.assertRaises(ValueError):
            a.risk_summary([{'forecast_available': True, 'predicted_failure': None, 'nonfeasible': 1}])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            a.number(float('nan'))

    def test_empty_summary_is_null(self):
        self.assertIsNone(a.describe([])['mean'])

    def test_first_action_myopic_must_exist(self):
        p = proposal()
        p['metadata']['information_value_decision']['first_action_indices'] = [1]
        with self.assertRaises(ValueError):
            a.decision_metrics(p, [None]*3)

    def test_input_not_mutated(self):
        p = proposal()
        before = deepcopy(p)
        a.decision_metrics(p, [None]*3)
        self.assertEqual(p, before)


if __name__ == '__main__':
    unittest.main()
