"""Synthetic evidence tests; fixture observations are not scientific experiments."""
import copy
import importlib.util
import math
from pathlib import Path
import random
import tempfile
import unittest

import pr86e_diagnostics as d


def candidate(i):
    return dict(branch=i % 4, x=i / 1000.0, y=0.5, trim=0.5)


def record(c, value=100.0, resource_ok=True):
    cid = d.digest(c)
    obs = dict(candidate=c, candidate_id=cid,
               region=dict(kind='exact', value=-0.5) if value is not None else dict(kind='interval', lower=0.0, lower_inclusive=False),
               resource=dict(kind='exact', value=-0.2 if resource_ok else 0.2),
               objective=dict(kind='exact', value=value) if value is not None else dict(kind='missing'))
    return dict(candidate=c, candidate_id=cid, observed=obs)


def fixture(cold=False, feasible=True):
    anchors = [record(candidate(i), None if cold and i < 12 else 100.0+i, feasible) for i in range(24)]
    pool = [candidate(i) for i in range(100, 148)]
    trials = []
    for n0 in (12, 24):
        for method in d.METHODS:
            history = [r['observed'] for r in anchors[:n0]]
            available = list(pool)
            online, proposals = [], []
            values = {r['candidate_id']: r['observed']['objective']['value'] for r in anchors[:n0]
                      if r['observed']['objective']['kind'] == 'exact' and feasible}
            for step in range(48-n0):
                ids = [d.digest(c) for c in available]
                exact = [(o['candidate_id'], o['objective']) for o in history if o['objective']['kind'] == 'exact']
                incumbent = min(values.values()) if values else None
                ps = {'region': [0.2 + (i % 4)*0.15 for i in range(len(ids))], 'resource': [0.8]*len(ids)}
                ps['joint'] = [a*b for a, b in zip(ps['region'], ps['resource'])]
                obj = {'mean': [95.0 + i % 10 for i in range(len(ids))], 'variance': [2.0]*len(ids)} if len(exact) >= 2 else None
                vectors = d.rule_vectors(ps, obj, incumbent)
                rng = random.Random(77 + 104729*step)
                scores = [rng.random() for _ in ids] if method == 'iid_random' else vectors[method]
                idx = d.argmax(scores, ids)
                forecast = dict(probabilities=ps)
                rec = record(available[idx], 200.0 + available[idx]['x'], feasible)
                proposal = dict(step=step, method=method, anchor_size=n0,
                    history_observed_sha256=d.digest(history), available_sha256=d.digest(available),
                    incumbent=incumbent, candidate_forecast=forecast, forecast_sha256=d.digest(forecast),
                    objective_forecast=obj, method_fit_objective=dict(exact_training_rows=len(exact),
                        training_rows=len(history), training_input_sha256=d.digest(exact), query_ids=ids,
                        status='OK' if obj is not None else 'INSUFFICIENT_EXACT_OBJECTIVES'),
                    scores=scores, score_digest=d.digest(scores), selected_index=idx,
                    selected_candidate_sha256=ids[idx], selected_candidate=available[idx])
                proposals.append(proposal)
                online.append(dict(step=step, candidate_id=ids[idx], record=rec))
                history.append(rec['observed'])
                if feasible:
                    values[ids[idx]] = rec['observed']['objective']['value']
                available.pop(idx)
            trials.append(dict(task='synthetic_fixture', repeat=0, method=method, anchor_size=n0, seed=77,
                anchor_ids=[r['candidate_id'] for r in anchors[:n0]], proposals=proposals,
                online_observations=online, final_outcome=dict(best_exact_feasible=min(values.values()) if values else None)))
    return dict(task='synthetic_fixture', repeat=0, anchors=anchors, pool=pool, trials=trials)


class PureTests(unittest.TestCase):
    def test_numeric_rules(self):
        self.assertEqual(d.ei(4.0, 0.0, 5.0), 1.0)
        self.assertEqual(d.ei(6.0, 0.0, 5.0), 0.0)
        self.assertAlmostEqual(d.ei(0.0, 1.0, 0.0), 1/math.sqrt(2*math.pi))
        self.assertEqual(d.ranks([3, 3, 3]), [0.5]*3)
        self.assertEqual(d.argmax([1, 1], ['a', 'b']), 1)

    def test_zeros_and_invalid_probabilities(self):
        ps = dict(region=[0.0, 0.0], resource=[0.5, 0.5], joint=[0.0, 0.0])
        result = d.rule_vectors(ps, None, None)
        self.assertEqual(result[d.CEI], [0.0, 0.0])
        self.assertEqual(result[d.WEIGHTED], [0.5, 0.5])
        ps['joint'][0] = 0.1
        with self.assertRaisesRegex(ValueError, 'Joint product'):
            d.rule_vectors(ps, None, None)

    def test_censored_objective_is_not_a_label(self):
        r = record(candidate(2), None)
        r['observed']['objective']['value'] = -99999.0  # ignored because kind is missing
        self.assertIsNone(d.observable(r)[1])
        self.assertFalse(d.observable(r)[2])

    def test_resource_infeasible_still_has_exact_objective(self):
        _, value, feasible = d.observable(record(candidate(1), 8.0, False))
        self.assertEqual(value, 8.0)
        self.assertFalse(feasible)

    def test_full_unit_and_initial_ties(self):
        decisions, endpoints, pairs = d.diagnose_unit(fixture())
        self.assertEqual((len(decisions), len(endpoints), len(pairs)), (300, 10, 8))
        self.assertTrue(all(p['category'] == 'same_observed_candidate' for p in pairs))
        self.assertTrue(all(p['shared_initial_incumbent_ids'] for p in pairs))
        self.assertTrue(all(r['phase'] == 'objective_ready' for r in decisions))
        self.assertTrue(d.summarize(decisions))

    def test_cold_fallback_is_separate(self):
        rows, _, _ = d.diagnose_unit(fixture(cold=True))
        cold = [r for r in rows if r['anchor_size'] == 12 and r['step'] < 2]
        self.assertEqual(len(cold), 10)
        self.assertTrue(all(r['phase'] == 'no_objective_gp' and r['objective_error'] is None for r in cold))
        self.assertTrue(all(r['phase'] == 'objective_ready' for r in rows if r['anchor_size'] == 24))

    def test_gp_without_incumbent_is_separate(self):
        rows, endpoints, pairs = d.diagnose_unit(fixture(feasible=False))
        self.assertTrue(all(r['phase'] == 'no_feasible_incumbent' for r in rows))
        self.assertTrue(all(r['best_exact_feasible'] is None for r in endpoints))
        self.assertTrue(all(r['category'] == 'neither_feasible' for r in pairs))

    def test_no_hidden_truth_or_test_labels_used(self):
        a = fixture()
        expected = d.diagnose_unit(a)
        a['tests'] = object()  # must never be accessed
        for r in a['anchors']:
            r['truth'] = object()
        for t in a['trials']:
            for r in t['online_observations']:
                r['record']['truth'] = object()
        self.assertEqual(d.diagnose_unit(a), expected)

    def test_reject_bad_prefix(self):
        a = fixture()
        a['trials'][0]['proposals'][0]['history_observed_sha256'] = 'fake'
        with self.assertRaisesRegex(ValueError, 'prefix/pool'):
            d.diagnose_unit(a)

    def test_reject_score_tampering(self):
        a = fixture()
        a['trials'][0]['proposals'][0]['scores'][0] += 0.001
        with self.assertRaisesRegex(ValueError, 'Score evidence'):
            d.diagnose_unit(a)

    def test_reject_feedback_misbinding(self):
        a = fixture()
        a['trials'][0]['online_observations'][0]['candidate_id'] = 'fake'
        with self.assertRaisesRegex(ValueError, 'feedback binding'):
            d.diagnose_unit(a)

    def test_output_cannot_overlap_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'outside input'):
                d.diagnose(root, root/'new', 'a'*40)

    def test_bad_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'input'
            root.mkdir()
            d.write_json(root/'plan.json', {})
            d.write_json(root/'manifest.json', dict(schema='sha256_manifest_v1', files={'plan.json': '0'*64}))
            with self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
                d.diagnose(root, Path(tmp)/'output', 'a'*40)


class EndToEndTests(unittest.TestCase):
    def test_write_once_full_diagnostic_on_synthetic_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'input', Path(tmp)/'diagnostics'
            root.mkdir()
            unit = fixture()
            units = []
            files = {}
            for task in d.TASKS:
                for repeat in range(3):
                    unit['task'], unit['repeat'] = task, repeat
                    for trial in unit['trials']:
                        trial['task'], trial['repeat'] = task, repeat
                    rel = 'units/%s/r%d/unit.json' % (task, repeat)
                    path = root/rel
                    path.parent.mkdir(parents=True)
                    d.write_json(path, unit)
                    files[rel] = d.file_hash(path)
                    units.append(dict(task=task, repeat=repeat))
            metadata = {
                'plan.json': dict(profile='full', units=units),
                'report.json': dict(code_sha=d.SOURCE_SHA, trials=300, status='COMPLETE_PR86D_INITIAL_SAMPLE_ABLATION'),
                'source_receipt.json': dict(code_sha=d.SOURCE_SHA),
            }
            for rel, payload in metadata.items():
                d.write_json(root/rel, payload)
                files[rel] = d.file_hash(root/rel)
            d.write_json(root/'manifest.json', dict(schema='sha256_manifest_v1', files=files))
            result = d.diagnose(root, output, 'a'*40)
            self.assertEqual(result['decisions'], 9000)
            self.assertEqual(result['new_evaluator_calls'], 0)
            self.assertEqual(result['new_gp_fits'], 0)
            self.assertEqual(result['identity_pairs'], 240)
            summary = d.read_json(output/'diagnostic_summary.json')
            self.assertEqual(len(summary['verified_input_files']), 33)
            self.assertEqual(d.file_hash(root/'plan.json'), files['plan.json'])
            with self.assertRaises(ValueError):
                d.diagnose(root, output, 'a'*40)


class FrozenReferenceTests(unittest.TestCase):
    def test_against_actual_frozen_acquisition(self):
        if importlib.util.find_spec('pr86c_acquisition') is None:
            self.skipTest('Frozen repository module absent in isolated local test; required by GitHub CI')
        from pr86c_acquisition import score_candidates
        rng = random.Random(42)
        for incumbent in (None, 5.0):
            for obj_present in (False, True):
                ps = dict(region=[rng.random() for _ in range(17)], resource=[rng.random() for _ in range(17)])
                ps['joint'] = [a*b for a, b in zip(ps['region'], ps['resource'])]
                obj = dict(mean=[rng.random()*10 for _ in range(17)], variance=[rng.random() for _ in range(17)]) if obj_present else None
                got = d.rule_vectors(ps, obj, incumbent)
                for method in d.METHODS[1:]:
                    expected, _ = score_candidates(method, dict(probabilities=ps), obj, incumbent)
                    self.assertTrue(all(d.close(a, b) for a, b in zip(got[method], expected)))


if __name__ == '__main__':
    unittest.main()
