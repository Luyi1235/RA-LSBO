"""Numerical seed continuation only. Never an admissible recharge/mission model.

Negative-load trials extend the voltage equation and charge bookkeeping to
help a root finder cross a domain boundary. Any resulting seed MUST be
re-evaluated using the unmodified discharge model before acceptance.
"""

from contextlib import contextmanager
import math
import numpy as np
from scipy.optimize import brentq

from . import nasa_battery as model


@contextmanager
def signed_seed_extension():
    original_state, original_advance = model.electrical_state, model.advance_cell

    def state(soc, power, *, enforce_limits=True):
        if enforce_limits or power >= 0 or not math.isfinite(power):
            return original_state(soc, power, enforce_limits=enforce_limits)
        model.ocv_integral(soc)
        u = float(np.interp(soc, model.SOC_BP, model.OCV))
        r = float(np.interp(soc, model.SOC_BP, model.R0 + model.RTH))
        current = 2 * power / (u + math.sqrt(u*u - 4*r*power))
        return current, u - current*r, u, r

    def advance(soc, power, duration, max_step=10., *, enforce_limits=True, envelope=None):
        if enforce_limits or power >= 0 or not math.isfinite(power):
            return original_advance(soc, power, duration, max_step,
                                    enforce_limits=enforce_limits, envelope=envelope)
        if not math.isfinite(duration) or duration < 0 or not math.isfinite(max_step) or max_step <= 0:
            raise ValueError('Invalid seed integration interval')
        def record(current, voltage):
            if envelope is not None:
                envelope['peak_current_A'] = max(envelope.get('peak_current_A',0.),current)
                envelope['minimum_voltage_V'] = min(envelope.get('minimum_voltage_V',float('inf')),voltage)
        current, voltage, _, _ = state(soc, power, enforce_limits=False)
        record(current, voltage)
        if duration == 0:
            return float(soc), 0., 0.
        initial, heat, q = float(soc), 0., model.CELL_AH*3600.
        count = max(1,int(math.ceil(duration/max_step)))
        dt = duration/count
        for _ in range(count):
            start = float(model.ocv_integral(soc))
            def at_current(current):
                end = soc-current*dt/q
                dz = soc-end
                u = ((start-float(model.ocv_integral(end)))/dz if abs(dz)>1e-12
                     else float(np.interp(soc,model.SOC_BP,model.OCV)))
                r = float(np.interp(.5*(soc+end),model.SOC_BP,model.R0+model.RTH))
                return end,u-current*r,r
            lower = -(1.-soc)*q/dt
            if lower >= 0 or lower*at_current(lower)[1] > power:
                raise model.BatteryDomainError('Numerical seed extension exceeds reference SOC map')
            current = brentq(lambda i:i*at_current(i)[1]-power,lower,0.,xtol=1e-12)
            soc,voltage,r = at_current(current)
            heat += current*current*r*dt
            record(current,voltage)
        return float(soc),heat,(initial-soc)*q

    model.electrical_state, model.advance_cell = state, advance
    try:
        yield
    finally:
        model.electrical_state, model.advance_cell = original_state, original_advance
