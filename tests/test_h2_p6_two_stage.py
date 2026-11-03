from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
from pathlib import Path

import pytest
from tools import h2_p6_two_stage as p6


def test_switch_occurs_only_after_second_original_near_filter():
    posterior = SimpleNamespace(h2_weight=None)
    seen = []
    def original_filter(*args, **kwargs):
        seen.append(('filter', posterior.h2_weight))
        return ('original-result',)
    def original_decision(*args):
        posterior.h2_weight = args[-1]
        seen.append(('initial', posterior.h2_weight))
        assert p6.discovery.filter_near()[0] == 'original-result'
        seen.append(('generation', posterior.h2_weight))
        assert p6.discovery.filter_near()[0] == 'original-result'
        seen.append(('final', posterior.h2_weight))
        return {'id': 1}, {'marker': True}
    with patch.object(p6.discovery, 'filter_near', original_filter), patch.object(p6.p5, 'replay', original_decision):
        result = p6.two_stage_replay(None, None, 1, 4, [], [], {}, posterior, .7, .5)
    assert result == ({'id': 1}, {'marker': True})
    assert seen == [('initial', .7), ('filter', .7), ('generation', .7), ('filter', .7), ('final', .5)]


def test_no_switch_does_not_bypass_coverage():
    with patch.object(p6.p5, 'replay', return_value=({}, {})):
        with pytest.raises(ValueError, match='near filters'):
            p6.two_stage_replay(None, None, 1, 4, [], [], {}, object(), .7, .5)


def test_ties_and_coverage_are_never_identified():
    cells = {name: {'score_gap': 1e-8, 'coverage_reason': None} for name in ('DD','FF','DF','FD')}
    assert p6.classify(cells, True, True) == ('identified', None)
    cells['DF']['score_gap'] = 1e-10
    assert p6.classify(cells, True, True)[0] == 'not_identified'
    cells['DF']['score_gap'] = 1e-8
    cells['DD']['coverage_reason'] = 'eligible_at_most_one'
    assert p6.classify(cells, True, True)[0] == 'not_identified'
    assert p6.classify(cells, False, True)[0] == 'not_identified'
    assert p6.classify(cells, True, False)[0] == 'not_identified'


def test_forbidden_aircraft_vae_and_rcaide():
    with patch.object(p6.AircraftCase, 'evaluate', p6.p5.forbidden), \
         patch.object(p6.SimplifiedElectricAircraft, 'evaluate', p6.p5.forbidden), \
         patch.object(p6.CorpusRepresentation, 'fit', p6.p5.forbidden), \
         patch.dict('sys.modules', {'RCAIDE': None}):
        with pytest.raises(RuntimeError, match='Forbidden'):
            p6.AircraftCase.evaluate(None, {})
        with pytest.raises(RuntimeError, match='Forbidden'):
            p6.SimplifiedElectricAircraft.evaluate(None, {})
        with pytest.raises(RuntimeError, match='Forbidden'):
            p6.CorpusRepresentation.fit(None, [])
        with pytest.raises(ModuleNotFoundError):
            __import__('RCAIDE')


def test_committed_evidence_and_preregistered_sensitivity():
    root = Path(__file__).resolve().parents[1] / 'results/h2-p6-two-stage-attribution-20261103'
    hashes = json.loads((root/'SHA256.json').read_text())
    for name, sha in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha
    data = json.loads((root/'two_stage_cells.json').read_text())
    ties = json.loads((root/'tie_sensitivity.json').read_text())
    assert len(data['pairs']) == 16
    assert all(data[key] == 0 for key in ('aircraft_calls', 'rcaide_calls', 'vae_fits'))
    assert [r['threshold'] for r in ties['sensitivity']] == list(p6.GRID)
    for row in data['pairs']:
        assert set(row['cells']) == {'DD', 'FF', 'DF', 'FD'}
        if row['category'] == 'identified':
            assert all(c['score_gap'] is not None and c['score_gap'] > p6.PRIMARY
                       and c['coverage_reason'] is None and c['repeat_identity']
                       for c in row['cells'].values())
