"""Keep electrical operating constraints distinct from numerical audit failures."""

import math

CONSTRAINT_POLICY = 'audited_operating_constraints_20261004_v1'
LEGACY_POLICY = 'strict_audit_error'
CONSTRAINT_NAMES = ('battery_current', 'battery_voltage')


def normalize_policy(value):
    if value not in (LEGACY_POLICY, CONSTRAINT_POLICY):
        raise ValueError('Unsupported battery operating-limit policy')
    return value


def constraints(peak_current_A, minimum_pack_voltage_V):
    from .nasa_battery import CELL_MAX_A, CELL_CUTOFF_V
    if (not all(math.isfinite(x) for x in (peak_current_A, minimum_pack_voltage_V))
            or peak_current_A < 0 or minimum_pack_voltage_V <= 0):
        raise ValueError('Invalid audited electrical operating values')
    return dict(battery_current=peak_current_A/CELL_MAX_A-1.,
                battery_voltage=1.-minimum_pack_voltage_V/(128*CELL_CUTOFF_V))
