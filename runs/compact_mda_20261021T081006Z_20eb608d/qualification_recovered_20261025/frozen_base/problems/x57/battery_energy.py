"""Versioned NASA X-57 Mod II battery calibration without editing the snapshot."""

import hashlib
import inspect
import math
from types import MethodType

import numpy as np
from . import nasa_battery as nasa
from . import nasa_battery_runtime as runtime


ENERGY_MODEL_VERSION = "nasa_modII_30Q_final_envelope_20260926"
# Eight balanced equivalent modules represent sixteen physical 16S20P modules.
PACK_MODULES = 8
CELL_SERIES = 16
CELL_PARALLEL = 40
CELL_CAPACITY_AH = nasa.CELL_AH
CELL_MAX_VOLTAGE = 4.2
PACK_CAPACITY_WH = nasa.PACK_CAPACITY_WH
CAPACITY_SOURCE = "nasa_2019_20C_OCV_integral_3Ah_5120cells"


def configure_bus(bus, initialize=True):
    modules = values(bus.battery_modules)
    if len(modules) != PACK_MODULES or bus.battery_module_electric_configuration != 'Series':
        raise ValueError('NASA equivalent model requires eight series 16S40P modules')
    if len(getattr(bus, 'fuel_cell_stacks', [])):
        raise ValueError('NASA battery-only equivalent cannot include fuel cells')
    for module in modules:
        module.electrical_configuration.series = CELL_SERIES
        module.electrical_configuration.parallel = CELL_PARALLEL
        module.cell.nominal_capacity = CELL_CAPACITY_AH
        module.cell.nominal_voltage = nasa.CELL_NOMINAL_V
        module.cell.maximum_voltage = CELL_MAX_VOLTAGE
        module.cell.mass = .048
        module.cell.specific_heat_capacity = 830.
    if initialize:
        bus.voltage = 0.
        bus._lsbo_native_initialize_bus_properties()
    for module in modules:
        module.maximum_energy = PACK_CAPACITY_WH * 3600. / PACK_MODULES
        module.initial_maximum_energy = module.maximum_energy
        # 350 kg is the 2019 report's pack-level mass, allocated across equivalent modules.
        module.mass_properties.mass = 350. / PACK_MODULES
        module.specific_energy = module.maximum_energy / module.mass_properties.mass
    bus.maximum_energy = PACK_CAPACITY_WH * 3600.
    bus.nasa_battery_model = ENERGY_MODEL_VERSION
    bus.nominal_voltage = 128 * nasa.CELL_NOMINAL_V
    bus.usable_terminal_energy_Wh = nasa.USABLE_TERMINAL_WH


def checked_bus_initialization(self, *args, **kwargs):
    # The native series initializer accumulates voltage without resetting it.
    self.voltage = 0.
    result = self._lsbo_native_initialize_bus_properties(*args, **kwargs)
    configure_bus(self, initialize=False)
    return result


def values(container):
    return list(container.values()) if hasattr(container, "values") else list(container)


def tagged(container):
    if hasattr(container, "items"):
        return {str(key): value for key, value in container.items()}
    return {str(value.tag): value for value in container}


def validate_vehicle_pack(vehicle):
    busses = values(vehicle.networks.electric.busses)
    if len(busses) != 1:
        raise ValueError("Energy contract requires one electric bus")
    bus = busses[0]
    modules = values(bus.battery_modules)
    if len(modules) != PACK_MODULES or bus.battery_module_electric_configuration != "Series":
        raise ValueError("Energy contract requires eight series-connected modules")
    total_wh = 0.
    for module in modules:
        expected = ((module.electrical_configuration.series, CELL_SERIES),
                    (module.electrical_configuration.parallel, CELL_PARALLEL),
                    (module.cell.nominal_capacity, CELL_CAPACITY_AH),
                    (module.cell.maximum_voltage, CELL_MAX_VOLTAGE))
        if any(not math.isclose(float(actual), target, rel_tol=1e-9) for actual, target in expected):
            raise ValueError("Battery cell/configuration differs from the declared energy contract")
        module_wh = PACK_CAPACITY_WH / PACK_MODULES
        for name in ("maximum_energy", "initial_maximum_energy"):
            if not math.isclose(float(getattr(module, name)) / 3600., module_wh, rel_tol=1e-7):
                raise ValueError(f"Battery {name} disagrees with cell-derived capacity")
        if not math.isclose(float(module.maximum_voltage), CELL_SERIES * CELL_MAX_VOLTAGE, rel_tol=1e-7):
            raise ValueError("Module voltage disagrees with series cell count")
        if not math.isclose(float(module.nominal_capacity), CELL_PARALLEL * CELL_CAPACITY_AH, rel_tol=1e-7):
            raise ValueError("Module Ah capacity disagrees with parallel cell count")
        total_wh += module_wh
    if not math.isclose(float(bus.maximum_energy) / 3600., total_wh, rel_tol=1e-7):
        raise ValueError("Bus capacity disagrees with module capacity sum")
    if not math.isclose(float(bus.voltage), PACK_MODULES * CELL_SERIES * CELL_MAX_VOLTAGE, rel_tol=1e-7):
        raise ValueError("Bus voltage disagrees with the series-connected pack")
    return bus, modules


def checked_energy_calc(self, state, bus, coolant_lines, t_idx, delta_t):
    # Reject later module-only reinitialization before it can change SOC dynamics.
    expected = PACK_CAPACITY_WH * 3600. / PACK_MODULES
    if not math.isclose(float(self.maximum_energy), expected, rel_tol=1e-7):
        raise ValueError("Battery capacity differs from the cell-derived energy contract")
    if not math.isclose(float(bus.maximum_energy), expected * PACK_MODULES, rel_tol=1e-7):
        raise ValueError("Bus capacity differs from the cell-derived energy contract")
    try:
        return runtime.energy_calc(self, state, bus, coolant_lines, t_idx, delta_t)
    except nasa.BatteryDomainError as error:
        from .battery_diagnostics import log_domain_trial
        log_domain_trial(error)
        raise
    except (ValueError, AttributeError, KeyError) as error:
        print(f'[NASA battery] {type(error).__name__}: {error}', flush=True)
        raise


def correct_discharge_step(module, state, bus, t_idx, delta_t):
    """Legacy sign-correction helper, retained for historical regression tests only.

    The native model supplies a heat-rate magnitude as its loss surrogate.
    The NASA runtime does not call this helper or use absolute heat as a loss proxy.
    """
    energy = state.conditions.energy
    bus_conditions = energy[bus.tag]
    conditions = bus_conditions.battery_modules[module.tag]
    phi = np.asarray(energy.hybrid_power_split_ratio).reshape(-1)
    psi = np.asarray(energy.battery_fuel_cell_power_split_ratio).reshape(-1)
    if not np.allclose(phi, 1.) or not np.allclose(psi, 1.):
        raise ValueError("Energy correction is restricted to the all-electric battery-only case")
    terminal = float(np.asarray(bus_conditions.power_draw)[t_idx, 0]) / len(bus.battery_modules)
    loss = abs(float(np.asarray(conditions.heat_energy_generated)[t_idx, 0]))
    if not math.isfinite(terminal + loss) or terminal < -1e-8:
        raise ValueError("Non-finite power or unsupported regenerative/charging segment")
    terminal = max(terminal, 0.)
    power = terminal + loss
    cells = module.electrical_configuration.series * module.electrical_configuration.parallel
    conditions.power[t_idx, 0] = power
    conditions.cell.power[t_idx, 0] = power / cells
    n = len(conditions.energy)
    fade = np.broadcast_to(np.asarray(conditions.cell.capacity_fade_factor).reshape(-1), (n,))
    if not np.isfinite(fade).all() or (fade <= 0).any():
        raise ValueError("Invalid battery capacity-fade factor")
    capacity = float(module.maximum_energy) * fade
    current = float(conditions.energy[t_idx, 0])
    conditions.state_of_charge[t_idx, 0] = np.clip(current / capacity[t_idx], 0., 1.)
    conditions.cell.state_of_charge[t_idx, 0] = conditions.state_of_charge[t_idx, 0]
    if t_idx < n - 1:
        dt = float(np.asarray(delta_t).reshape(-1)[t_idx])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid mission integration time step")
        following = current - power * dt
        # Do not hide depletion by clipping the stored energy to zero.
        conditions.energy[t_idx + 1, 0] = following
        conditions.cell.energy[t_idx + 1, 0] = following / cells
        soc = np.clip(following / capacity[t_idx + 1], 0., 1.)
        conditions.state_of_charge[t_idx + 1, 0] = soc
        conditions.cell.state_of_charge[t_idx + 1, 0] = soc
        conditions.cell.depth_of_discharge[t_idx + 1, 0] = 1. - soc


def install_energy_model(nexus):
    vehicles = values(nexus.vehicle_configurations)
    # Analyses normally share configurations; include distinct referenced copies too.
    for analysis in values(nexus.analyses):
        energy_analysis = getattr(analysis, "energy", None)
        if energy_analysis is not None:
            vehicles.append(energy_analysis.vehicle)
    seen = set()
    seen_busses = set()
    native_hashes = set()
    kernel_hashes = set()
    for vehicle in vehicles:
        busses = values(vehicle.networks.electric.busses)
        if len(busses) != 1:
            raise ValueError("Energy contract requires one electric bus")
        bus = busses[0]
        if id(bus) not in seen_busses:
            seen_busses.add(id(bus))
            if hasattr(bus, "_lsbo_native_initialize_bus_properties"):
                raise ValueError("Energy model was already installed")
            bus._lsbo_native_initialize_bus_properties = bus.initialize_bus_properties
            bus.initialize_bus_properties = MethodType(checked_bus_initialization, bus)
            bus._lsbo_native_compute_conditions = bus.compute_distributor_conditions
            bus.compute_distributor_conditions = MethodType(runtime.compute_bus_conditions, bus)
            configure_bus(bus)
        _, modules = validate_vehicle_pack(vehicle)
        for module in modules:
            if id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, "_lsbo_native_energy_calc"):
                raise ValueError("Energy model was already installed")
            native = module.energy_calc
            # Restrict this adapter rather than silently wrapping a different chemistry.
            if type(module).__name__ != "Lithium_Ion_NMC":
                raise ValueError("Energy correction requires Lithium_Ion_NMC modules")
            native_hashes.add(hashlib.sha256(inspect.getsource(native).encode()).hexdigest())
            kernel = native.__func__.__globals__.get("compute_nmc_cell_performance")
            if kernel is None:
                raise ValueError("Unsupported NMC energy kernel; inspect the installed RCAIDE version")
            kernel_hashes.add(hashlib.sha256(inspect.getsource(kernel).encode()).hexdigest())
            module._lsbo_native_energy_calc = native
            module.energy_calc = MethodType(checked_energy_calc, module)
            module._lsbo_native_append_conditions = module.append_operating_conditions
            module.append_operating_conditions = MethodType(runtime.append_conditions, module)
            module.update_battery_age = MethodType(runtime.no_age_update, module)
    return {"energy_model_version": ENERGY_MODEL_VERSION, "declared_pack_capacity_Wh": PACK_CAPACITY_WH,
            "capacity_source": CAPACITY_SOURCE,
            "soc_definition": "charge_fraction_3Ah_coulomb_counted",
            "physical_configuration": "two_128S20P_strings_16_physical_modules",
            "computational_equivalent": "balanced_128S40P_8_equivalent_modules",
            "nominal_energy_Wh": nasa.NOMINAL_ENERGY_WH,
            "maximum_voltage_times_Ah_Wh": nasa.MAX_VOLTAGE_AH_WH,
            "usable_terminal_energy_Wh": nasa.USABLE_TERMINAL_WH,
            "pack_nominal_voltage_V": 128 * nasa.CELL_NOMINAL_V,
            "pack_mass_kg": 350.,
            "mission_cell_current_limit_A": nasa.CELL_MAX_A,
            "operating_limit_check": "final_converged_mission_including_substep_extrema",
            "minimum_allowed_pack_voltage_V": 128 * nasa.CELL_CUTOFF_V,
            "aircraft_weight_scope": "existing_fixed_nonwing_mass_retained_no_battery_double_add",
            "thermal_assumption": "isothermal_20C_no_cooling_power_or_thermal_validation",
            "electrical_model": "NASA_2019_appendix_A_20C_OCV_R0_plus_Rth_quasisteady",
            "native_NMC_kernel_used": False,
            "cell_capacity_Ah": CELL_CAPACITY_AH, "cell_max_voltage_V": CELL_MAX_VOLTAGE,
            "module_series_cells": CELL_SERIES, "module_parallel_cells": CELL_PARALLEL,
            "energy_loss_convention": "I_squared_R0_plus_Rth_dissipation",
            "native_energy_method_sha256": sorted(native_hashes),
            "native_energy_kernel_sha256": sorted(kernel_hashes)}
