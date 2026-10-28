"""RCAIDE adapter for the balanced NASA 30Q reference battery model."""

import numpy as np
from . import nasa_battery as model


def append_conditions(self, segment, bus):
    self._lsbo_native_append_conditions(segment, bus)
    if 'initial_battery_state_of_charge' in segment:
        soc = float(segment.initial_battery_state_of_charge)
        energy = float(model.stored_energy_j(soc, 640))
        bc = segment.state.conditions.energy[bus.tag]
        history = bc.battery_modules[self.tag]
        history.energy[:] = energy
        history.maximum_initial_energy = energy
        history.cell.energy[:] = energy / 640.
        bc.energy[:] = energy * model.EQUIVALENT_MODULES


def no_age_update(self, *args, **kwargs):
    # A fresh-pack, single-mission reference case; do not apply an unrelated NMC aging fit.
    return None


def compute_bus_conditions(self, state, index, dt):
    self._lsbo_native_compute_conditions(state, index, dt)
    bc = state.conditions.energy[self.tag]
    modules = list(self.battery_modules.values())
    histories = [bc.battery_modules[m.tag] for m in modules]
    bc.energy[index, 0] = sum(float(h.energy[index, 0]) for h in histories)
    bc.state_of_charge[index, 0] = min(float(h.state_of_charge[index, 0]) for h in histories)
    power = float(bc.power_draw[index, 0])
    loss = sum(float(h.heat_energy_generated[index, 0]) for h in histories)
    bc.efficiency[index, 0] = power / (power + loss) if power + loss else 1.
    # Balanced parallel strings have equal cell currents; series module currents match.
    bc.current_draw[index, 0] = histories[0].current[index, 0]


def energy_calc(self, state, bus, coolant_lines, index, delta_t):
    energy = state.conditions.energy
    if (not np.allclose(energy.hybrid_power_split_ratio, 1.) or
            not np.allclose(energy.battery_fuel_cell_power_split_ratio, 1.) or
            bool(energy.recharging)):
        raise ValueError('NASA battery adapter supports battery-only discharge, not hybrid/recharge')
    if len(coolant_lines):
        raise ValueError('NASA 20 C isothermal reference does not implement coolant coupling')
    if not np.isclose(self.maximum_energy, model.PACK_CAPACITY_WH * 3600 / 8):
        raise ValueError('NASA module energy contract was overwritten')
    if not np.isclose(bus.maximum_energy, model.PACK_CAPACITY_WH * 3600):
        raise ValueError('NASA bus energy contract was overwritten')
    bc = energy[bus.tag]
    history = bc.battery_modules[self.tag]
    for key in ('capacity_fade_factor', 'resistance_growth_factor'):
        if not np.allclose(getattr(history.cell, key), 1.):
            raise ValueError('NASA reference model supports fresh cells only')
    cells = self.electrical_configuration.series * self.electrical_configuration.parallel
    per_cell_power = float(bc.power_draw[index, 0]) / model.TOTAL_CELLS
    initial_energy = float(history.energy[index, 0])
    soc = model.soc_from_energy(initial_energy, cells)
    if not np.isclose(float(history.cell.state_of_charge[index, 0]), soc, atol=1e-7):
        raise ValueError('Initial charge SOC and OCV-integrated stored energy disagree')
    # RCAIDE calls this inside its nonlinear solver, before force balance has
    # converged. Engineering limits are enforced on the final mission histories
    # by energy_audit; impossible electrical roots/depletion still fail here.
    current, voltage, ocv, resistance = model.electrical_state(soc, per_cell_power, enforce_limits=False)
    interval = index < len(history.energy) - 1
    duration = float(np.asarray(delta_t).reshape(-1)[index]) if interval else 0.
    if interval and duration <= 0:
        raise ValueError('Invalid battery integration interval')
    envelope = {}
    next_soc, heat_j, coulombs = model.advance_cell(soc, per_cell_power, duration,
                                                 enforce_limits=False, envelope=envelope)
    for key, value in (('peak_current', envelope['peak_current_A']),
                       ('minimum_voltage', envelope['minimum_voltage_V'])):
        if not hasattr(history.cell, key):
            setattr(history.cell, key, np.zeros_like(history.cell.current))
        getattr(history.cell, key)[index, 0] = value
    final_energy = float(model.stored_energy_j(next_soc, cells))
    if duration > 0:
        loss_w = heat_j / duration
        current = coulombs / duration
        # Interval averages preserve both IV=P and the charge/energy integrals.
        if current > 0:
            voltage = per_cell_power / current
            ocv = (initial_energy - final_energy) / (cells * coulombs)
            resistance = loss_w / current ** 2
    else:
        loss_w = current * current * resistance
    history.cell.current[index, 0] = current
    history.current[index, 0] = current * self.electrical_configuration.parallel
    for target, scale in ((history, cells), (history.cell, 1)):
        target.power[index, 0] = scale * (per_cell_power + loss_w)
        target.heat_energy_generated[index, 0] = scale * loss_w
        target.temperature[index, 0] = model.REFERENCE_T_K
        target.state_of_charge[index, 0] = soc
    history.cell.energy[index, 0] = initial_energy / cells
    history.cell.depth_of_discharge[index, 0] = 1 - soc
    history.cell.internal_resistance[index, 0] = resistance
    history.internal_resistance[index, 0] = resistance * self.electrical_configuration.series / self.electrical_configuration.parallel
    history.cell.voltage_under_load[index, 0] = voltage
    history.cell.voltage_open_circuit[index, 0] = ocv
    history.voltage_under_load[index, 0] = voltage * self.electrical_configuration.series
    history.voltage_open_circuit[index, 0] = ocv * self.electrical_configuration.series
    if interval:
        history.energy[index + 1, 0] = final_energy
        history.cell.energy[index + 1, 0] = final_energy / cells
        for target in (history, history.cell):
            target.state_of_charge[index + 1, 0] = next_soc
            target.temperature[index + 1, 0] = model.REFERENCE_T_K
        history.cell.depth_of_discharge[index + 1, 0] = 1 - next_soc
        history.cell.charge_throughput[index + 1, 0] = history.cell.charge_throughput[index, 0] + coulombs / 3600.
    return True, self.tag
