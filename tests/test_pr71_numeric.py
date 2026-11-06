from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from pr71_quadrature import node_rule, score, parity, compare_forecasts
from pr70_future_pool import choose_next
from ra_lsbo.experiments.information_value_stage3.planner import nodes, FantasyFitError
from pr71_restore import RestoredHead, GPLedger, no_new_truth_or_training
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableGroupedHead


def forecast():
    return {'constraints': {'region': {'mean': [-.4, .1, -.1], 'variance': [.2, .3, .2]},
                            'resource': {'mean': [-.4, -.3, -.2], 'variance': [.1, .1, .1]}},
            'objective': {'mean': [2., 1., 3.], 'variance': [1., 2., 1.]}}


class Factory:
    def __init__(self, error=None):
        self.error = error
    def support(self, i):
        return {'model_candidates': [{'x': .2}, {'x': .4}, {'x': .6}]}
    def condition(self, i, o, use_abort_interval=True):
        if self.error:
            raise self.error('deliberate test')
        return forecast()
    def diagnostics(self):
        return {'head_fits_started': 0, 'head_fits_completed': 0}


class NumericTests(unittest.TestCase):
    def test_original_four_nodes_exact(self):
        np.testing.assert_array_equal(node_rule(998901, 4, 4), nodes(998901, 4))

    def test_nested_sequence(self):
        np.testing.assert_array_equal(node_rule(5, 0, 4), node_rule(5, 0, 64)[:4])
        np.testing.assert_array_equal(node_rule(5, 0, 16), node_rule(5, 0, 64)[:16])

    def test_independent_scramble(self):
        self.assertFalse(np.array_equal(node_rule(5, 0, 64), node_rule(5, 0, 64, 1)))

    def test_reproducible(self):
        np.testing.assert_array_equal(node_rule(9, 1, 16), node_rule(9, 1, 16))

    def test_bad_nodes_rejected(self):
        for n in (0, 3, 17, True):
            with self.assertRaises(ValueError):
                node_rule(9, 1, n)

    def test_bad_replicate_rejected(self):
        with self.assertRaises(ValueError):
            node_rule(9, 1, 4, 2)

    def test_parent_same_rule_parity(self):
        f = forecast()
        _, original = choose_next(f, 4., 2, Factory(), seed=7, iteration=0)
        r = score(f, 4., Factory(), seed=7, iteration=0, n=4)
        self.assertTrue(parity(r, original))

    def test_cold_mass_and_missing_objective(self):
        f = forecast()
        f['objective'] = None
        r = score(f, None, Factory(), seed=7, iteration=0, n=16)
        self.assertEqual(r['status'], 'ok')
        for o in r['fantasies']:
            self.assertEqual(o['observation']['objective']['kind'], 'missing')

    def test_abort_never_discloses_hidden_values(self):
        r = score(forecast(), 4., Factory(), seed=7, iteration=0, n=4)
        for f in r['fantasies']:
            if f['observation']['aborted']:
                self.assertNotIn('value', f['observation']['region'])
                self.assertEqual(f['observation']['objective'], {'kind': 'missing'})

    def test_numerical_failure_has_no_winner(self):
        r = score(forecast(), 4., Factory(FantasyFitError), seed=7, iteration=0, n=4)
        self.assertEqual(r['status'], 'numerical_failure')
        self.assertIsNone(r['selected_index'])

    def test_programming_error_propagates(self):
        with self.assertRaises(ValueError):
            score(forecast(), 4., Factory(ValueError), seed=7, iteration=0, n=4)

    def test_input_unmutated(self):
        f = forecast()
        before = deepcopy(f)
        score(f, 4., Factory(), seed=7, iteration=0, n=4)
        self.assertEqual(f, before)

    def test_forecast_discrepancy_rejected(self):
        a, b = forecast(), forecast()
        b['objective']['mean'][0] += 1.
        with self.assertRaises(ValueError):
            compare_forecasts(a, b)

    def test_real_saved_selected_head_restore(self):
        x = np.array([[0., 1., 1.], [.25, 1., 1.], [.7, 1., 1.], [1., 1., 1.]])
        obs = [OutputObservation('exact', str(i), value=v) for i,v in enumerate([.7, .3, -.4, -.2])]
        original = StableGroupedHead(x, obs, continuous_count=1, seed=19, name='region')
        with TemporaryDirectory() as d:
            ledger = GPLedger(Path(d)/'fits.jsonl')
            h = RestoredHead(x, obs, original.diagnostics(), 1, ledger)
            np.testing.assert_allclose(h.predict(x)['mean'], original.predict(x)['mean'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(h.predict(x)['variance'], original.predict(x)['variance'], rtol=1e-10, atol=1e-10)
            self.assertEqual(ledger.summary(), {'started': 1, 'completed': 1, 'errors': 0})

    def test_evaluator_and_training_are_forbidden(self):
        from ra_lsbo.experiments.core.problems import MathematicalCase
        from ra_lsbo.structure_corpus import CorpusRepresentation
        with TemporaryDirectory() as d:
            with no_new_truth_or_training(GPLedger(Path(d)/'fits.jsonl')):
                with self.assertRaises(RuntimeError):
                    MathematicalCase.evaluate(None, {})
                with self.assertRaises(RuntimeError):
                    CorpusRepresentation.fit(None, [])


if __name__ == '__main__':
    unittest.main()
