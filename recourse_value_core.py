"""Finite-node recourse decomposition; no models, evaluators or file I/O.

For each first action, both criteria use IDENTICAL positive weighted scenarios
and candidate columns. E[max_y v(y,Y)] - max_y E[v(y,Y)] is nonnegative for
this finite scenario model, not a calibrated physical information-value bound.
"""
import math

ADAPTIVE = 'cei_two_step_adaptive'
FIXED = 'cei_two_step_fixed_second'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(x):
    require(type(x) in (int, float) and math.isfinite(x), 'Finite numeric value required')
    return float(x)


def winner(values):
    require(bool(values), 'Empty action values')
    require(all(type(k) is int and k >= 0 for k in values), 'Integer action identities required')
    return max(values, key=lambda k: (number(values[k]), -k))


def decompose(base, firsts, scenarios, counts):
    """Reconcile each root's full scenario vectors, preserving parent sum order."""
    require(isinstance(base, list) and base and all(number(v) >= 0 for v in base), 'Invalid Q1')
    require(firsts and len(set(firsts)) == len(firsts)
            and all(type(i) is int and 0 <= i < len(base) for i in firsts), 'Invalid root screen')
    require(set(counts) == set(firsts) and all(type(k) is int and k > 0 for k in counts.values()), 'Support dimensions differ')
    root_rows = {i: [] for i in firsts}
    for row in scenarios:
        require(type(row['first_index']) is int and row['first_index'] in root_rows, 'Unknown first action')
        root_rows[row['first_index']].append(row)
    roots = {}
    for i in firsts:
        expected = [0.] * counts[i]
        adaptive, mass = 0., 0.
        require(root_rows[i], 'Missing root outcomes')
        for row in root_rows[i]:
            w = number(row['weight'])
            values = row['future_values']
            require(w > 0 and isinstance(values, list) and len(values) == counts[i], 'Invalid scenario weight/support')
            require(all(number(v) >= 0 for v in values), 'Negative/nonfinite continuation')
            second = winner(dict(enumerate(values)))
            maximum = float(values[second])
            mass += w
            adaptive += w * maximum
            for j, v in enumerate(values):
                expected[j] += w * float(v)
        require(math.isclose(mass, 1., rel_tol=0., abs_tol=1e-10), 'Scenario mass not one')
        fixed_index = winner(dict(enumerate(expected)))
        fixed = expected[fixed_index]
        premium = adaptive - fixed
        require(premium >= -1e-12 * max(1., abs(adaptive), abs(fixed)), 'Finite-node Jensen inequality violated')
        roots[str(i)] = {
            'adaptive_continuation': adaptive, 'fixed_continuation': fixed,
            'expected_value_by_second_action': expected, 'fixed_second_index': fixed_index,
            'adaptive_score': float(base[i] + adaptive), 'fixed_score': float(base[i] + fixed),
            'recourse_premium': max(0., premium), 'scenario_mass': mass,
        }
    adaptive_index = winner({i: roots[str(i)]['adaptive_score'] for i in firsts})
    fixed_index = winner({i: roots[str(i)]['fixed_score'] for i in firsts})
    myopic = winner(dict(enumerate(base)))
    require(myopic in firsts, 'Root screen must include myopic winner')
    return {'roots': roots, 'adaptive_index': adaptive_index, 'fixed_index': fixed_index,
            'myopic_index': myopic, 'same_state_root_choice_differs': adaptive_index != fixed_index,
            'interpretation': 'finite-node adaptive continuation premium, not physical calibration or safety'}
