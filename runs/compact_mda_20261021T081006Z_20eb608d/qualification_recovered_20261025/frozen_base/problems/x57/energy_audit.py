"""Audit stored-energy depletion against the solver's discharge power history."""

import math
import numpy as np

from .battery_energy import tagged, values, CAPACITY_SOURCE
from . import nasa_battery as nasa
from .operating_limits import LEGACY_POLICY, CONSTRAINT_POLICY, normalize_policy, constraints as operating_constraints


class EnergyAuditError(ValueError):
    pass


def series(value, count, name):
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) == 1:
        array = np.full(count, array[0])
    if len(array) != count or not np.isfinite(array).all():
        raise EnergyAuditError(f"Invalid {name} history")
    return array


def require_close(actual, expected, name):
    # Joule-scale absolute tolerance plus relative floating-point tolerance.
    if not np.allclose(actual, expected, rtol=1e-5, atol=1.):
        difference = float(np.max(np.abs(np.asarray(actual) - expected)))
        raise EnergyAuditError(f"{name}: maximum energy/power discrepancy {difference:.6g}")


def audit_mission(nexus, report, operating_limit_policy=LEGACY_POLICY):
    policy = normalize_policy(operating_limit_policy)
    busses = values(nexus.vehicle_configurations.base.networks.electric.busses)
    if len(busses) != 1:
        raise EnergyAuditError("Audit requires one electric bus")
    bus = busses[0]
    nasa_mode = bool(getattr(bus, 'nasa_battery_model', False))
    if policy == CONSTRAINT_POLICY and not nasa_mode:
        raise EnergyAuditError('Operating constraints require the audited NASA electrical model')
    modules = tagged(bus.battery_modules)
    if not modules:
        raise EnergyAuditError("No battery modules")
    capacities = {tag: float(module.maximum_energy) for tag, module in modules.items()}
    if any(not math.isfinite(c) or c <= 0 for c in capacities.values()):
        raise EnergyAuditError("Invalid module capacity")
    segments = values(nexus.results.base.segments)
    if not segments:
        raise EnergyAuditError("No mission energy histories")
    report.update(status="checking", objective_definition="sum_initial_minus_final_module_energy",
                  integration="left_endpoint_terminal_power_interval_average_battery_losses", segments=[],
                  operating_limit_policy=policy)
    previous = None
    first = None
    previous_time = None
    terminal_total = loss_total = segment_total = 0.
    final_soc = []
    min_soc = 1.
    mission_peak_current = 0.
    mission_minimum_voltage = float('inf')
    mission_maximum_power = 0.
    for segment in segments:
        time = np.asarray(segment.conditions.frames.inertial.time, dtype=float).reshape(-1)
        if len(time) < 2 or not np.isfinite(time).all() or (np.diff(time) <= 0).any():
            raise EnergyAuditError("Invalid mission time history")
        if previous_time is not None and not math.isclose(time[0], previous_time, abs_tol=1e-5):
            raise EnergyAuditError("Time discontinuity between mission segments")
        dt, n = np.diff(time), len(time)
        conditions = segment.conditions.energy[bus.tag]
        histories = tagged(conditions.battery_modules)
        if set(histories) != set(modules):
            raise EnergyAuditError("Missing or mismatched module histories")
        bus_power = series(conditions.power_draw, n, "bus power")
        if (bus_power < -1e-8).any():
            raise EnergyAuditError("Charging/regeneration is outside this mission definition")
        mission_maximum_power = max(mission_maximum_power, float(bus_power.max()))
        start, end, energy_sum = {}, {}, np.zeros(n)
        power_sum, heat_sum = np.zeros(n), np.zeros(n)
        block = {"tag": str(segment.tag), "time_s": time.tolist(), "modules": {}}
        report["segments"].append(block)
        final_soc = []
        for tag, history in histories.items():
            energy = series(history.energy, n, tag + " stored energy")
            power = series(history.power, n, tag + " depletion power")
            loss = series(history.heat_energy_generated, n, tag + " heat rate")
            if nasa_mode and (loss < -1e-8).any():
                raise EnergyAuditError('Negative resistive dissipation')
            loss = np.abs(loss)
            soc = series(history.state_of_charge, n, tag + " SOC")
            fade = series(history.cell.capacity_fade_factor, n, tag + " capacity fade")
            if (fade <= 0).any() or (soc < -1e-8).any() or (soc > 1 + 1e-8).any():
                raise EnergyAuditError("Invalid SOC/capacity fade history")
            if (energy < -1.).any() or (energy > capacities[tag] * fade + 1.).any():
                raise EnergyAuditError("Battery depleted or stored energy exceeds capacity")
            if nasa_mode:
                require_close(energy, nasa.stored_energy_j(soc, 640), tag + ' OCV energy/charge SOC')
                current = series(history.cell.current, n, tag + ' cell current')
                voltage = series(history.cell.voltage_under_load, n, tag + ' cell terminal voltage')
                peak_current = series(history.cell.peak_current, n, tag + ' substep peak current')
                minimum_voltage = series(history.cell.minimum_voltage, n, tag + ' substep minimum voltage')
                mission_peak_current = max(mission_peak_current, float(np.maximum(current, peak_current).max()))
                mission_minimum_voltage = min(mission_minimum_voltage, float(np.minimum(voltage, minimum_voltage).min()))
                # Discharge sign / nonpositive voltage are model-domain errors.
                # Positive current and low positive voltage may instead describe
                # a fully audited but electrically inadmissible mission.
                if ((current < -1e-8).any() or (peak_current < -1e-8).any()
                        or (np.minimum(voltage, minimum_voltage) <= 0).any()
                        or (voltage > nasa.CELL_MAX_V + 1e-8).any()):
                    raise EnergyAuditError('Invalid discharge current or cell voltage history')
                if policy == LEGACY_POLICY and (np.maximum(current, peak_current) > nasa.CELL_MAX_A + 1e-8).any():
                    index = int(np.argmax(np.maximum(current, peak_current)))
                    raise EnergyAuditError(f'Final mission current limit exceeded: segment={segment.tag}, '
                                           f'module={tag}, time_s={time[index]:.8g}, '
                                           f'peak_cell_current_A={max(current[index], peak_current[index]):.8g}, '
                                           f'limit_A={nasa.CELL_MAX_A:g}, pack_power_W={bus_power[index]:.8g}')
                if policy == LEGACY_POLICY and (np.minimum(voltage, minimum_voltage) < nasa.CELL_CUTOFF_V - 1e-8).any():
                    index = int(np.argmin(np.minimum(voltage, minimum_voltage)))
                    raise EnergyAuditError(f'Final mission voltage limit exceeded: segment={segment.tag}, '
                                           f'module={tag}, time_s={time[index]:.8g}, '
                                           f'minimum_pack_voltage_V={min(voltage[index], minimum_voltage[index])*128:.8g}, '
                                           f'minimum_allowed_V={nasa.CELL_CUTOFF_V*128:g}')
                if not np.allclose((soc[:-1] - soc[1:]) * nasa.CELL_AH * 3600., current[:-1] * dt,
                                   rtol=1e-6, atol=1e-5):
                    raise EnergyAuditError('Charge/SOC time-step balance')
                if not np.allclose(voltage * current * nasa.TOTAL_CELLS, bus_power, rtol=1e-6, atol=.01):
                    raise EnergyAuditError('Terminal voltage/current/power balance')
            else:
                require_close(energy, capacities[tag] * fade * soc, tag + " energy/SOC")
            require_close(energy[:-1] - energy[1:], power[:-1] * dt, tag + " time-step balance")
            require_close(power, bus_power / len(modules) + loss, tag + " discharge loss sign")
            start[tag], end[tag] = float(energy[0]), float(energy[-1])
            if previous is not None:
                require_close(start[tag], previous[tag], tag + " segment continuity")
            energy_sum += energy
            power_sum += power
            heat_sum += loss
            final_soc.append(float(soc[-1]))
            min_soc = min(min_soc, float(soc.min()))
            block["modules"][tag] = {"energy_J": energy.tolist(), "SOC": soc.tolist(),
                                     "depletion_power_W": power.tolist(), "loss_magnitude_W": loss.tolist()}
            if nasa_mode:
                block['modules'][tag].update(cell_current_A=current.tolist(), cell_voltage_V=voltage.tolist(),
                                            substep_peak_current_A=peak_current.tolist(),
                                            substep_minimum_voltage_V=minimum_voltage.tolist())
        require_close(series(conditions.energy, n, "bus stored energy"), energy_sum, "bus/module energy sum")
        terminal = float(np.dot(bus_power[:-1], dt))
        loss_energy = float(np.dot(heat_sum[:-1], dt))
        depletion = sum(start.values()) - sum(end.values())
        require_close(depletion, terminal + loss_energy, "segment electrical balance")
        block.update(bus_power_W=bus_power.tolist(), depletion_Wh=depletion / 3600.,
                     terminal_energy_Wh=terminal / 3600., loss_energy_Wh=loss_energy / 3600.)
        terminal_total += terminal
        loss_total += loss_energy
        segment_total += depletion
        first = start if first is None else first
        previous, previous_time = end, float(time[-1])
    depletion = sum(first.values()) - sum(previous.values())
    require_close(depletion, segment_total, "whole-mission segment sum")
    require_close(depletion, terminal_total + loss_total, "whole-mission electrical balance")
    metrics = {"energy_audit_status": "passed", "battery_capacity_Wh": sum(capacities.values()) / 3600.,
               "battery_module_count": len(modules), "capacity_source": CAPACITY_SOURCE if nasa_mode else "legacy_module_energy_sum",
               "objective_source": "sum_initial_minus_final_module_energy",
               "initial_stored_energy_Wh": sum(first.values()) / 3600.,
               "final_stored_energy_Wh": sum(previous.values()) / 3600.,
               "mission_depletion_Wh": depletion / 3600., "terminal_energy_Wh": terminal_total / 3600.,
               "battery_loss_energy_Wh": loss_total / 3600.,
               "energy_balance_residual_Wh": (depletion - terminal_total - loss_total) / 3600.,
               "final_soc": min(final_soc), "minimum_recorded_soc": min_soc,
               "final_soc_segment": str(segments[-1].tag)}
    if nasa_mode:
        electrical_constraints = operating_constraints(mission_peak_current, mission_minimum_voltage*128)
        metrics.update(usable_terminal_energy_Wh=nasa.USABLE_TERMINAL_WH,
                       operating_limit_policy=policy,
                       operating_limits_status='passed' if all(v <= 0 for v in electrical_constraints.values()) else 'violated',
                       battery_operating_constraints=electrical_constraints,
                       peak_cell_current_A=mission_peak_current,
                       minimum_loaded_pack_voltage_V=mission_minimum_voltage * 128,
                       maximum_terminal_power_W=mission_maximum_power,
                       usable_energy_constraint=terminal_total / 3600. / nasa.USABLE_TERMINAL_WH - 1.,
                       final_stored_energy_fraction=sum(previous.values()) / sum(capacities.values()),
                       soc_definition='charge_fraction_not_energy_fraction')
    report.update(status="passed", summary=metrics)
    return metrics
