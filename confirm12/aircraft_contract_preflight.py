"""Inspect existing aircraft problem contracts without invoking the evaluator."""
from ra_lsbo.experiments import iid_qualification
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for


def inspect():
    cases = {}
    for case in ('aircraft_ordinary', 'aircraft_sparse'):
        nominal = problem_for(case, 20.)
        refined = problem_for(case, 5.)
        expected = iid_qualification.HISTORICAL_CONTRACTS[case]
        cases[case] = {'nominal_20s_contract': nominal.contract_id,
                       'historical_nominal_contract': expected,
                       'nominal_matches_historical': nominal.contract_id == expected,
                       'verification_5s_contract': refined.contract_id,
                       'nominal_distance_m': nominal.settings['nominal_cruise_distance_m'],
                       'base_distance_m_before_evaluate': nominal.settings['cruise_distance_m'],
                       'nominal_step_s': nominal.settings['integration_step_s'],
                       'verification_step_s': refined.settings['integration_step_s']}
    return {'cases': cases, 'new_aircraft_calls': 0, 'new_rcaide_calls': 0,
            'task_qualified': False, 'production_authorized': False}
