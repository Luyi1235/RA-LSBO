"""Preregistered cold-only coverage ablation; no scientific imports."""
import math
import statistics

PARENT = '1949c3139d2256b35e1734af184248f551912ba0'
CONTROL = 'cei_myopic'
TREATMENT = 'cei_myopic_cold_coverage'
LEGACY = 'soft_gp_eligible'
ARMS = (CONTROL, TREATMENT, LEGACY)
CASES = ('branin_ordinary', 'branin_sparse', 'rosenbrock_ordinary', 'rosenbrock_sparse')
PERIOD = 5


def require(value, message):
    if not value:
        raise ValueError(message)


def scheduled(cold, iteration):
    require(type(cold) is bool and type(iteration) is int and iteration >= 0, 'Invalid phase/iteration')
    return cold and iteration % PERIOD == 0


def plan(profile):
    require(profile in ('ci', 'smoke', 'full'), 'Unregistered profile')
    cases = (CASES[0], CASES[3]) if profile == 'ci' else (CASES if profile == 'smoke' else (CASES[1], CASES[3]))
    seeds = (999190,) if profile == 'ci' else ((999191,) if profile == 'smoke' else tuple(range(999201, 999217)))
    steps = 6 if profile == 'full' else 2
    groups = len(cases)*len(seeds)
    return {'schema': 'cold_coverage_scope_isolation_v1', 'profile': profile,
            'cases': list(cases), 'seeds': list(seeds), 'arms': list(ARMS),
            'initial': 12, 'steps': steps, 'pool': 128,
            'initial_calls': groups*12, 'online_calls': groups*len(ARMS)*steps,
            'total_calls': groups*(12+len(ARMS)*steps),
            'primary_pair': [TREATMENT, CONTROL], 'primary_stratum': 'initial_cold',
            'primary_endpoint': 'first_feasible_discovery_by_fixed_call_budget',
            'only_change': 'inherited coverage_scope none -> cold on the same cei_myopic engine',
            'coverage_period': PERIOD, 'coverage_rule': 'least_visited_available_branch_then_uniform_pool',
            'coverage_degenerate_support_guard_retained': True,
            'warm_primary_pair_must_have_identical_paths': True,
            'all_initial_groups_retained_no_rejection_or_resampling': True,
            'same_true_calls_not_same_compute': True, 'fantasy_fits': 0,
            'legacy_internal_horizon': 12, 'exploratory_not_confirmatory': True,
            'aircraft_calls': 0, 'rcaide_calls': 0}


def first_feasible(curve):
    require(isinstance(curve, list) and len(curve) >= 2, 'Need initial and online curve')
    previous = None
    for value in curve:
        require(value is None or (type(value) in (int, float) and math.isfinite(value)), 'Invalid incumbent')
        if previous is not None:
            require(value is not None and value <= previous + 1e-10, 'Incumbent cannot disappear or increase')
        if value is not None:
            previous = value
    return next((i for i, v in enumerate(curve) if v is not None), None)


def compare(a, b):
    require(all(v is None or (type(v) in (int, float) and math.isfinite(v)) for v in (a, b)), 'Invalid score')
    if a is None or b is None:
        return 0 if a is b else (1 if a is not None else -1)
    require(all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)), 'Invalid score')
    return 1 if a < b-1e-10 else (-1 if a > b+1e-10 else 0)


def summarize(rows, spec):
    indexed = {(r['case'], r['seed'], r['arm']): r for r in rows}
    expected = {(c, s, a) for c in spec['cases'] for s in spec['seeds'] for a in ARMS}
    require(len(rows) == len(indexed) and set(indexed) == expected, 'Incomplete/duplicate roster')
    result = {}
    for case in spec['cases']:
        for seed in spec['seeds']:
            group = [indexed[case, seed, a] for a in ARMS]
            require(len({r['initial_hash'] for r in group}) == 1 and
                    len({r['initial_best'] for r in group}) == 1, 'Initial evidence differs')
            for r in group:
                require(len(r['curve']) == spec['steps']+1 and r['curve'][0] == r['initial_best']
                        and r['curve'][-1] == r['final_best'] and
                        r['first_feasible_call'] == first_feasible(r['curve']), 'Curve/first delivery differs')
            a, b = indexed[case, seed, CONTROL], indexed[case, seed, TREATMENT]
            if a['initial_best'] is not None:
                require(a['path_hash'] == b['path_hash'] and a['curve'] == b['curve'],
                        'Matched initially-warm paths differ: not a cold-only intervention')
        strata = {}
        for stratum in ('all', 'initial_cold', 'initial_warm'):
            seeds = [s for s in spec['seeds'] if stratum == 'all' or
                     ((indexed[case, s, CONTROL]['initial_best'] is None) == (stratum == 'initial_cold'))]
            item = {'groups': len(seeds), 'arms': {}, 'pairs': {}}
            for arm in ARMS:
                group = [indexed[case, s, arm] for s in seeds]
                delivered = [r for r in group if r['final_best'] is not None]
                item['arms'][arm] = {'trials': len(group), 'final_feasible': len(delivered),
                    'no_solution': len(group)-len(delivered),
                    'discoveries': sum(r['initial_best'] is None and r['final_best'] is not None for r in group),
                    'first_feasible_calls': {str(r['seed']): r['first_feasible_call'] for r in group},
                    **{k: sum(r[k] for r in group) for k in ('online_calls', 'nonfeasible', 'proposal_seconds',
                        'scheduled_coverage_calls', 'degenerate_coverage_calls', 'model_failure_calls',
                        'cold_decisions', 'model_free_calls')}}
            for left, right in ((TREATMENT, CONTROL), (TREATMENT, LEGACY), (CONTROL, LEGACY)):
                wlt = [0, 0, 0]; details = []; deltas = []
                for seed in seeds:
                    a, b = indexed[case, seed, left], indexed[case, seed, right]
                    outcome = compare(a['final_best'], b['final_best'])
                    wlt[0 if outcome == 1 else 1 if outcome == -1 else 2] += 1
                    delta = a['final_best']-b['final_best'] if a['final_best'] is not None and b['final_best'] is not None else None
                    if delta is not None:
                        deltas.append(delta)
                    details.append({'seed': seed, 'outcome': outcome, 'objective_delta': delta,
                        'left_first_feasible': a['first_feasible_call'], 'right_first_feasible': b['first_feasible_call'],
                        'left_feasible': a['final_best'] is not None, 'right_feasible': b['final_best'] is not None,
                        'path_changed': a['path_hash'] != b['path_hash']})
                item['pairs'][left+'_vs_'+right] = {'wins_losses_ties': wlt,
                    'both_feasible_objective_delta_mean': statistics.mean(deltas) if deltas else None,
                    'discovery_discordance_left_only_right_only': [
                        sum(d['left_feasible'] and not d['right_feasible'] for d in details),
                        sum(d['right_feasible'] and not d['left_feasible'] for d in details)],
                    'per_seed': details}
            strata[stratum] = item
        result[case] = strata
    return result
