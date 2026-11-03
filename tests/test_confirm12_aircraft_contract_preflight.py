from confirm12.aircraft_contract_preflight import inspect


def test_actual_aircraft_contracts_read_without_truth(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    def forbidden(*args, **kwargs):
        raise AssertionError('aircraft truth is forbidden')
    monkeypatch.setattr(AircraftCase, 'evaluate', forbidden)
    result = inspect()
    assert result['new_aircraft_calls'] == result['new_rcaide_calls'] == 0
    assert not result['task_qualified'] and not result['production_authorized']
    for case, distance in [('aircraft_ordinary', 100000), ('aircraft_sparse', 140000)]:
        row = result['cases'][case]
        assert row['nominal_distance_m'] == distance
        assert row['base_distance_m_before_evaluate'] == 100000
        assert row['nominal_step_s'] == 20 and row['verification_step_s'] == 5
        assert row['nominal_20s_contract'] != row['verification_5s_contract']
        assert row['nominal_matches_historical'] == (
            row['nominal_20s_contract'] == row['historical_nominal_contract'])
