"""20 C quasi-steady reduction of the published X-57 cell performance maps.

Chin et al., Battery Performance Modeling on Maxwell X-57,
author-hosted 2019 PDF, Appendix A, p. 13, 20 C row.
R = R_0 + R_Th is the steady-state reduction, NOT the full transient RC model.
No cooling power or thermal qualification is inferred from this reference case.
"""

import math
import numpy as np


CELL_AH = 3.0
CELL_NOMINAL_V = 3.6
CELL_MAX_V = 4.2
# Table 1 in the 2019 modeling paper: mission limit 9 A, pack floor 330 V.
# The cell hardware rating (15 A) is not the mission operating limit.
CELL_HARDWARE_MAX_A = 15.0
CELL_CUTOFF_V = 330. / 128
CELL_MAX_A = 9.0
REFERENCE_T_K = 293.15
PHYSICAL_PACKS = 2
PHYSICAL_MODULES_PER_PACK = 8
SERIES_PER_MODULE = 16
PARALLEL_PER_PHYSICAL_MODULE = 20
# Balanced single-bus equivalent: pair corresponding physical modules in parallel.
EQUIVALENT_MODULES = 8
PARALLEL_CELLS = 40
TOTAL_CELLS = 5120
USABLE_TERMINAL_WH = 47000.
SOC_BP = np.array([0., .03333333, .06666667, .1, .13333333, .16666667,
                   .2, .23333333, .26666667, .3, .33333333, .36666667,
                   .4, .43333333, .46666667, .5, .53333333, .56666667,
                   .6, .63333333, .66666667, .7, .73333333, .76666667,
                   .8, .83333333, .86666667, .9, .93333333, .96666667, 1.])
OCV = np.array([2.99293893, 3.05400763, 3.11507634, 3.17614504, 3.23506616,
                3.30371247, 3.37521374, 3.43605852, 3.48697455, 3.52002290,
                3.54251908, 3.58374046, 3.63293130, 3.67379644, 3.70287532,
                3.73784733, 3.76526463, 3.79174809, 3.81922901, 3.84108142,
                3.87212214, 3.90738931, 3.93615267, 3.98113995, 4.02093893,
                4.04504071, 4.07114758, 4.07583969, 4.08371501, 4.10560814, 4.161])
R0 = np.array([.08801527, .07274809, .05748092, .04221374, .03231552,
               .02722646] + [.025] * 25)
RTH = np.array([.08534351, .07516539, .06498728, .05480916, .04838931,
                .04589059, .045, .045, .04195929, .03937405, .03642494,
                .035, .035, .03848601, .05430025, .04534351, .03624682,
                .03115776, .03, .03, .03, .03839695, .04, .04, .04,
                .03089059, .03, .03, .02807125, .02505344, .02])
_SLOPE = np.diff(OCV) / np.diff(SOC_BP)
_INTEGRAL = np.r_[0., np.cumsum(.5 * (OCV[:-1] + OCV[1:]) * np.diff(SOC_BP))]


def ocv_integral(soc):
    """Exact integral of piecewise-linear OCV from empty to charge SOC."""
    z = np.asarray(soc, dtype=float)
    if not np.isfinite(z).all() or (z < -1e-10).any() or (z > 1 + 1e-10).any():
        raise ValueError("Charge SOC outside the NASA reference map")
    z = np.clip(z, 0., 1.)
    j = np.clip(np.searchsorted(SOC_BP, z, side='right') - 1, 0, len(SOC_BP) - 2)
    dz = z - SOC_BP[j]
    return _INTEGRAL[j] + OCV[j] * dz + .5 * _SLOPE[j] * dz * dz


def stored_energy_j(soc, cells=1):
    return cells * CELL_AH * 3600. * ocv_integral(soc)


def soc_from_energy(energy_j, cells=1):
    target = float(energy_j) / (cells * CELL_AH * 3600.)
    if not math.isfinite(target) or target < -1e-8 or target > _INTEGRAL[-1] + 1e-8:
        raise ValueError("Stored energy outside the NASA reference map")
    target = min(max(target, 0.), _INTEGRAL[-1])
    j = min(max(int(np.searchsorted(_INTEGRAL, target, side='right')) - 1, 0), len(_SLOPE) - 1)
    area = target - _INTEGRAL[j]
    dz = 2 * area / (OCV[j] + math.sqrt(OCV[j] ** 2 + 2 * _SLOPE[j] * area))
    return float(SOC_BP[j] + dz)


PACK_CAPACITY_WH = float(stored_energy_j(1., TOTAL_CELLS) / 3600.)
NOMINAL_ENERGY_WH = TOTAL_CELLS * CELL_AH * CELL_NOMINAL_V
MAX_VOLTAGE_AH_WH = TOTAL_CELLS * CELL_AH * CELL_MAX_V


class BatteryDomainError(ValueError):
    """Requested discharge has no admissible mathematical battery trajectory."""


def electrical_state(soc, cell_power, *, enforce_limits=True):
    """Low-current physical root of P = I (U_oc - I R)."""
    ocv_integral(soc)  # reject extrapolation
    if not math.isfinite(cell_power) or cell_power < 0:
        raise BatteryDomainError("Only finite nonnegative battery discharge power is supported")
    u = float(np.interp(soc, SOC_BP, OCV))
    r = float(np.interp(soc, SOC_BP, R0 + RTH))
    discriminant = u * u - 4 * r * cell_power
    if discriminant < 0:
        raise BatteryDomainError(f'Battery power exceeds reference-model voltage-collapse limit: '
                         f'SOC={soc:.8g}, cell_power_W={cell_power:.8g}, '
                         f'pack_power_W={cell_power * TOTAL_CELLS:.8g}, '
                         f'cell_max_power_W={u*u/(4*r):.8g}')
    current = 2 * cell_power / (u + math.sqrt(discriminant)) if cell_power else 0.
    voltage = u - current * r
    if enforce_limits and (current > CELL_MAX_A + 1e-8 or voltage < CELL_CUTOFF_V - 1e-8):
        raise ValueError(f'Battery current/voltage limit exceeded: SOC={soc:.8g}, '
                         f'cell_current_A={current:.8g} (limit {CELL_MAX_A:g}), '
                         f'pack_voltage_V={voltage*128:.8g} (minimum {CELL_CUTOFF_V*128:g}), '
                         f'pack_power_W={cell_power*TOTAL_CELLS:.8g}')
    return current, voltage, u, r


def advance_cell(soc, cell_power, duration, max_step=10., *, enforce_limits=True, envelope=None):
    """Coulomb counting with energy-conserving implicit interval current.

    Each substep uses OCV averaged over the actual charge removed, and resistance
    at midpoint SOC. This enforces dE = P_terminal*dt + I^2*R*dt without rescaling
    mission power or conflating energy SOC with charge SOC.
    """
    if not math.isfinite(duration) or duration < 0 or not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("Invalid battery integration interval")
    def record(current, voltage):
        if envelope is not None:
            envelope['peak_current_A'] = max(envelope.get('peak_current_A', 0.), current)
            envelope['minimum_voltage_V'] = min(envelope.get('minimum_voltage_V', float('inf')), voltage)

    initial_state = electrical_state(soc, cell_power, enforce_limits=enforce_limits)
    record(initial_state[0], initial_state[1])
    initial = float(soc)
    if duration == 0 or cell_power == 0:
        return float(soc), 0., 0.
    count = max(1, int(math.ceil(duration / max_step)))
    dt, q = duration / count, CELL_AH * 3600.
    heat = 0.
    for _ in range(count):
        start_integral = float(ocv_integral(soc))

        def at_current(current):
            next_soc = max(0., soc - current * dt / q)
            dz = soc - next_soc
            avg_u = ((start_integral - float(ocv_integral(next_soc))) / dz
                     if dz > 1e-12 else float(np.interp(soc, SOC_BP, OCV)))
            resistance = float(np.interp(.5 * (soc + next_soc), SOC_BP, R0 + RTH))
            return next_soc, avg_u - current * resistance, resistance

        u = float(np.interp(soc, SOC_BP, OCV))
        r = float(np.interp(soc, SOC_BP, R0 + RTH))
        # During nonlinear trial evaluations use the mathematical low-current
        # branch, not the final engineering envelope. Never clip requested power.
        hi = min(CELL_MAX_A if enforce_limits else u / (2 * r), soc * q / dt)
        # Restrict to the admissible voltage branch before solving the power equation.
        if enforce_limits and at_current(hi)[1] < CELL_CUTOFF_V:
            a, b = 0., hi
            for _ in range(35):
                mid = .5 * (a + b)
                if at_current(mid)[1] >= CELL_CUTOFF_V:
                    a = mid
                else:
                    b = mid
            hi = a
        if hi * at_current(hi)[1] < cell_power - 1e-8:
            raise BatteryDomainError(f'Battery depleted or requested interval power is infeasible: '
                             f'SOC={soc:.8g}, pack_power_W={cell_power*TOTAL_CELLS:.8g}, '
                             f'interval_s={dt:.8g}, current_bracket_upper_A={hi:.8g}')
        lo = 0.
        for _ in range(35):
            mid = .5 * (lo + hi)
            if mid * at_current(mid)[1] < cell_power:
                lo = mid
            else:
                hi = mid
        current = .5 * (lo + hi)
        soc, voltage, resistance = at_current(current)
        heat += current * current * resistance * dt
        record(current, voltage)
        end_state = electrical_state(soc, cell_power, enforce_limits=enforce_limits)
        record(end_state[0], end_state[1])
    coulombs = (initial - soc) * q
    return float(soc), heat, coulombs
