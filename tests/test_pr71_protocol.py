import subprocess
import sys
import unittest
from pathlib import Path
from pr71_protocol import plan, summarize, rule_id, RULES, PARENT, EVIDENCE


class ProtocolTests(unittest.TestCase):
    def test_full_roster(self):
        p = plan('full')
        self.assertEqual(p['n_states'], 32)
        self.assertEqual({s['iteration'] for s in p['states']}, {0, 4})
        self.assertEqual({s['seed'] for s in p['states']}, set(range(998901, 998905)))

    def test_smoke_is_four_states(self):
        self.assertEqual(plan('smoke')['n_states'], 4)

    def test_ci_covers_two_end_seeds(self):
        self.assertEqual(plan('ci')['n_states'], 2)
        self.assertEqual([s['seed'] for s in plan('ci')['states']], [998901, 998904])

    def test_no_true_or_vae_work(self):
        p = plan('full')
        for k in ('new_mathematical_evaluations', 'new_VAE_fits', 'aircraft_calls', 'rcaide_calls'):
            self.assertEqual(p[k], 0)

    def test_exact_rule_roster(self):
        self.assertEqual(RULES, ((4, 0), (16, 0), (64, 0), (64, 1)))

    def test_bad_profile(self):
        with self.assertRaises(ValueError):
            plan('online')

    def test_two_distinct_fixed_shas(self):
        self.assertEqual(len(PARENT), 40)
        self.assertEqual(len(EVIDENCE), 40)
        self.assertNotEqual(PARENT, EVIDENCE)

    def test_rule_identity(self):
        self.assertEqual(rule_id(64, 1), 'n64_r1')

    def test_summary_keeps_unavailable(self):
        rules = {rule_id(*r): {'status': 'ok', 'selected_index': 1} for r in RULES}
        rules['n64_r1'] = {'status': 'numerical_failure', 'selected_index': None}
        out = summarize([{'rules': rules}])['n64_r0_vs_n64_r1']
        self.assertEqual(out['comparable_states'], 0)
        self.assertEqual(out['unavailable_states'], 1)

    def test_empty_summary(self):
        self.assertEqual(summarize([])['n4_r0_vs_n16_r0']['states_total'], 0)

    def test_summary_records_changed_winner(self):
        rules = {rule_id(*r): {'status': 'ok', 'selected_index': int(r[0] != 4)} for r in RULES}
        self.assertEqual(summarize([{'rules': rules}])['n4_r0_vs_n16_r0']['different_winner'], 1)

    def test_help_without_scientific_libraries(self):
        root = Path(__file__).resolve().parents[1]
        r = subprocess.run([sys.executable, '-S', str(root/'run_pr71_quadrature.py'), '--help'],
                           cwd=root, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('--evidence-root', r.stdout)


if __name__ == '__main__':
    unittest.main()
