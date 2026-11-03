from unittest.mock import patch
import hashlib
import json
from pathlib import Path

import pytest

from tools import h2_p5_full_path as p5


def test_strict_identity_and_fixed_float_tolerance():
    assert not p5.differences({'x': [1., 2.]}, {'x': [1., 2. + 1e-11]})
    assert p5.differences({'x': [1.]}, {'x': [1. + 1e-5]})
    assert p5.differences(['candidate-a'], ['candidate-b'])
    assert p5.differences([1, 2], [2, 1])


def test_replay_rejects_candidate_and_local_path_mismatch():
    saved = {'candidate': {'id': 1}, 'metadata': {'pool': [{'id': 1}], 'local_reports': [{'status': 0}]}}
    meta = {'pool': [{'id': 1}], 'local_reports': [{'status': 1}]}
    assert not p5.verify_replay(saved, {'id': 2}, meta)['passed']


def test_aircraft_and_vae_calls_are_forbidden():
    with patch.object(p5.AircraftCase, 'evaluate', p5.forbidden), patch.object(p5.CorpusRepresentation, 'fit', p5.forbidden):
        with pytest.raises(RuntimeError, match='Forbidden'):
            p5.AircraftCase.evaluate(None, {})
        with pytest.raises(RuntimeError, match='Forbidden'):
            p5.CorpusRepresentation.fit(None, [])


def test_rcaide_not_needed_for_offline_analysis():
    with patch.dict('sys.modules', {'RCAIDE': None}):
        with pytest.raises(ModuleNotFoundError):
            __import__('RCAIDE')


def test_committed_results_are_hashed_and_near_ties_are_not_identified():
    root = Path(__file__).resolve().parents[1] / 'results/h2-p5-full-path-intervention-20261103-revised'
    hashes = json.loads((root/'SHA256.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    result = json.loads((root/'full_path_interventions.json').read_text())
    assert len(result['pairs']) == 16
    for pair in result['pairs']:
        assert len(pair['native_replays']) == 2
        assert all(len(replays) == 2 and all(x['passed'] for x in replays)
                   for replays in pair['native_replays'].values())
        if pair.get('reason') == 'numerical_near_tie':
            assert pair['category'] == 'not_identified'
    identified = [p for p in result['pairs'] if p['category'] == 'identified']
    assert result['identified_count'] == len(identified)
    assert result['confirmed_choice_flips'] == sum(p['weight_changes_choice'] for p in identified)
