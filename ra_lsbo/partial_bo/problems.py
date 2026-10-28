"""Auditable analytic and reduced electric-aircraft development cases."""

from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from ..budget import digest
from ..features import load_base
from ..risk_core.observations import OutputObservation, RunObservation
from ..structured_encoder import hierarchy_space
from .records import DesignRecord


def actuator_disk_power(thrust, speed, density, disk_area):
    if (not all(np.isfinite(value) for value in (thrust, speed, density, disk_area))
            or thrust < 0 or speed <= 0 or density <= 0 or disk_area <= 0):
        raise ValueError("Invalid ideal actuator-disk operating point")
    induced = .5*(np.sqrt(speed**2+2*thrust/(density*disk_area))-speed)
    return thrust*(speed+induced)


class DevelopmentProblem:
    analytical_constraints = ()
    coverage_variables = ()
    solver_domain_metric = None

    def __init__(self, base, variables, settings):
        modules, _ = load_base(base)
        self.space = hierarchy_space(modules, variables)
        self.candidate_class = modules["space"].Candidate
        self.settings = settings
        self.contract_id = digest({"schema": variables, "settings": settings,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def canonical(self, raw):
        candidate = self.space.canonical(self.candidate_class(**raw))
        if (set(raw["continuous"]) != {variable.name for variable in self.space.continuous}
                or set(raw["discrete"]) != {variable.name for variable in self.space.discrete}):
            raise ValueError("Explicit complete candidate schema required")
        for variable in self.space.continuous:
            if self.space.active(variable, candidate.discrete) and not (
                    variable.bounds[0] <= raw["continuous"][variable.name] <= variable.bounds[1]):
                raise ValueError("Refusing active-coordinate clipping")
        return candidate.to_dict()

    def precheck(self, raw):
        self.canonical(raw)
        return {}

    def record(self, candidate, objective, constraints, diagnostics, status="completed"):
        evidence = digest([self.contract_id, candidate, diagnostics])
        observed = lambda name, value: OutputObservation("exact", evidence+":"+name, value=float(value))
        outputs = tuple((name, value if isinstance(value, OutputObservation) else observed(name, value))
                        for name, value in sorted(constraints.items()))
        objective = objective if isinstance(objective, OutputObservation) else observed("objective", objective)
        return DesignRecord(candidate, RunObservation(
            design_id=digest(candidate), contract_id=self.contract_id, evidence_id=evidence,
            status=status, objective=objective, constraints=outputs,
            failure_cause=diagnostics.get("failure_cause") if status == "hidden_failure" else None), diagnostics)

    def anchors(self):
        raise NotImplementedError


class HierarchicalBranin(DevelopmentProblem):
    """Declared hierarchical Branin extension, not an upstream benchmark replica."""

    name = "Hierarchical Branin with explicit early-stop constraint feedback"
    objective_reference_scale = 50.
    required_constraints = ("region", "resource")

    def __init__(self, base):
        variables = [
            {"name": "architecture", "kind": "nominal", "default": "basic",
             "values": ["basic", "augmented"]},
            {"name": "module", "kind": "ordinal", "default": 1, "values": [1, 2, 3],
             "active_when": {"architecture": ["augmented"]}},
            {"name": "x", "kind": "continuous", "default": .65, "bounds": [0., 1.]},
            {"name": "y", "kind": "continuous", "default": .2, "bounds": [0., 1.]},
            {"name": "trim", "kind": "continuous", "default": .5, "bounds": [0., 1.],
             "active_when": {"architecture": ["augmented"]}}]
        super().__init__(base, variables, {
            "family": "hierarchical_branin_extension_v1", "ellipse_radii": [.16, .2],
            "feedback": "region violation returns only a sign and skips the objective"})

    def evaluate(self, raw):
        candidate = self.canonical(raw)
        continuous, discrete = candidate["continuous"], candidate["discrete"]
        shift = .035*(discrete["module"]-2) if discrete["architecture"] == "augmented" else 0.
        position_x, position_y = continuous["x"]+shift, continuous["y"]
        region = ((position_x-.68)/.16)**2+((position_y-.25)/.2)**2-1
        resource = (continuous["trim"]+.12*discrete["module"]-1.05
                    if discrete["architecture"] == "augmented" else -.2)
        if region > 0:
            identity = digest([self.contract_id, candidate, "region_threshold"])
            return self.record(candidate,
                OutputObservation("missing", identity+":objective", reason="declared_constraint_abort"),
                {"region": OutputObservation("interval", identity+":region", lower=0., lower_inclusive=False),
                 "resource": resource},
                {"termination_reason": "controlled_region_abort", "hidden_failure": False,
                 "model_family": "analytic_extension", "new_physical_mda": False})
        position_x, position_y = 15*position_x-5, 15*position_y
        objective = ((position_y-5.1/(4*np.pi**2)*position_x**2+5/np.pi*position_x-6)**2
                     +10*(1-1/(8*np.pi))*np.cos(position_x)+10)
        if discrete["architecture"] == "augmented":
            objective += 2*(continuous["trim"]-.25*discrete["module"])**2
        return self.record(candidate, float(objective), {"region": region, "resource": resource},
                           {"termination_reason": "complete", "model_family": "analytic_extension",
                            "new_physical_mda": False})

    def anchors(self):
        first = self.space.default().to_dict()
        second = self.space.canonical(self.candidate_class(
            {"x": .72, "y": .2, "trim": .5}, {"architecture": "augmented", "module": 2})).to_dict()
        return [first, second]


class SimplifiedElectricAircraft(DevelopmentProblem):
    """Point-mass polar, ideal actuator disks, and frozen NASA cell maps."""

    name = "Reduced electric fixed-wing sizing and mission case"
    objective_reference_scale = 40000.
    required_constraints = ("span", "packing", "stall", "root_bending", "soc", "current", "voltage", "power")
    analytical_constraints = ("span", "packing")
    coverage_variables = ("prop_layout",)

    def __init__(self, base, integration_step=20.):
        variables = [
            {"name": "prop_layout", "kind": "nominal", "default": "wingtip",
             "values": ["wingtip", "distributed"]},
            {"name": "propeller_count", "kind": "ordinal", "default": 2, "values": [2, 4, 6],
             "active_when": {"prop_layout": ["distributed"]}},
            {"name": "wing_area", "kind": "continuous", "default": 14.75, "bounds": [10., 18.]},
            {"name": "aspect_ratio", "kind": "continuous", "default": 8.8, "bounds": [7.5, 13.5]},
            {"name": "tip_diameter", "kind": "continuous", "default": 1.72, "bounds": [1.2, 1.9],
             "active_when": {"prop_layout": ["wingtip"]}},
            {"name": "dep_diameter", "kind": "continuous", "default": .8, "bounds": [.45, 1.1],
             "active_when": {"prop_layout": ["distributed"]}},
            {"name": "prop_spacing", "kind": "continuous", "default": .7, "bounds": [.5, .9],
             "active_when": {"prop_layout": ["distributed"]}}]
        battery_path = Path(base)/"problems/x57/nasa_battery.py"
        specification = importlib.util.spec_from_file_location("_ra_reduced_nasa_cell", battery_path)
        self.battery = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(self.battery)
        self.battery_source = battery_path
        settings = {"family": "reduced_electric_aircraft_v1", "cruise_distance_m": 100000.,
                    "cruise_speed_m_s": 50., "climb_height_m": 500., "reserve_duration_s": 600.,
                    "initial_soc": .95, "minimum_soc": .2, "rho_kg_m3": 1.225,
                    "profile_cd": .025, "oswald_e": .8, "cl_max": 1.8,
                    "electric_efficiency": .92, "stall_limit_m_s": 30.,
                    "root_moment_limit_Nm": 70000., "ultimate_load_factor": 3.8,
                    "fixed_mass_kg": 1050., "wing_areal_density_kg_m2": 5.,
                    "integration_step_s": float(integration_step),
                    "battery_source_sha256": hashlib.sha256(battery_path.read_bytes()).hexdigest(),
                    "assumptions": "research screening assumptions, not validated X57 or certification"}
        if integration_step <= 0 or not np.isfinite(integration_step):
            raise ValueError("Positive finite integration step required")
        super().__init__(base, variables, settings)

    def geometry(self, candidate):
        continuous, discrete = candidate["continuous"], candidate["discrete"]
        count = discrete["propeller_count"] if discrete["prop_layout"] == "distributed" else 2
        diameter = continuous["dep_diameter"] if discrete["prop_layout"] == "distributed" else continuous["tip_diameter"]
        span = np.sqrt(continuous["wing_area"]*continuous["aspect_ratio"])
        packing = (count/2*diameter+.05*(count/2-1)+.6-continuous["prop_spacing"]*span/2
                   if discrete["prop_layout"] == "distributed" else -1.)
        return count, diameter, span, {"span": float(span/14.5-1), "packing": float(packing)}

    def precheck(self, raw):
        return self.geometry(self.canonical(raw))[3]

    def evaluate(self, raw):
        candidate = self.canonical(raw)
        count, diameter, span, constraints = self.geometry(candidate)
        if any(value > 0 for value in constraints.values()):
            raise ValueError("Apply the common exact geometry screen before evaluating")
        settings, battery = self.settings, self.battery
        area, aspect_ratio = candidate["continuous"]["wing_area"], candidate["continuous"]["aspect_ratio"]
        mass = (settings["fixed_mass_kg"]+settings["wing_areal_density_kg_m2"]*area
                +2*count+4*count*(diameter/1.5)**3)
        weight, rho = mass*9.80665, settings["rho_kg_m3"]
        constraints.update(stall=float(np.sqrt(2*weight/(rho*area*settings["cl_max"]))
                                       /settings["stall_limit_m_s"]-1),
                           root_bending=float(settings["ultimate_load_factor"]*weight*span
                                              /(8*settings["root_moment_limit_Nm"])-1))
        identity = digest([self.contract_id, candidate])
        for name in ("soc", "current", "voltage", "power"):
            constraints[name] = OutputObservation("missing", identity+":"+name, reason="not_executed")
        soc, energy_wh, elapsed = settings["initial_soc"], 0., 0.
        peak_current, minimum_voltage, max_power_margin = 0., np.inf, -np.inf
        stages = (("climb", 40., 2., settings["climb_height_m"]/2.),
                  ("cruise", settings["cruise_speed_m_s"], 0., settings["cruise_distance_m"]/settings["cruise_speed_m_s"]),
                  ("descent", 40., -1.5, settings["climb_height_m"]/1.5),
                  ("reserve", 40., 0., settings["reserve_duration_s"]))
        termination, trace = "complete", []
        for stage, speed, climb_rate, duration in stages:
            lift_coefficient = weight/(.5*rho*speed**2*area)
            drag_coefficient = settings["profile_cd"]+lift_coefficient**2/(np.pi*settings["oswald_e"]*aspect_ratio)
            drag = .5*rho*speed**2*area*drag_coefficient
            thrust = max(0., drag+weight*climb_rate/speed)
            disk_area = count*np.pi*(diameter/2)**2
            power_w = actuator_disk_power(thrust, speed, rho, disk_area)/settings["electric_efficiency"]+500.
            steps = max(1, int(np.ceil(duration/settings["integration_step_s"])))
            step_duration = duration/steps
            for step in range(steps):
                cell_ocv = float(np.interp(soc, battery.SOC_BP, battery.OCV))
                resistance = float(np.interp(soc, battery.SOC_BP, battery.R0+battery.RTH))
                max_power = battery.TOTAL_CELLS*cell_ocv**2/(4*resistance)
                max_power_margin = max(max_power_margin, power_w/max_power-1)
                constraints["power"] = OutputObservation(
                    "interval", identity+":power_prefix", lower=float(max_power_margin))
                if power_w > max_power:
                    termination = "physical_battery_power_domain"
                    break
                envelope = {}
                try:
                    next_soc, _, _ = battery.advance_cell(
                        soc, power_w/battery.TOTAL_CELLS, step_duration,
                        max_step=settings["integration_step_s"], enforce_limits=False, envelope=envelope)
                except battery.BatteryDomainError as error:
                    return self.record(candidate,
                        OutputObservation("interval", identity+":energy_prefix", lower=float(energy_wh)),
                        constraints, {"termination_reason": "physical_discharge_domain",
                            "failure_cause": "battery_discharge_domain", "numerical_failure": False,
                            "model_domain_failure": True, "domain_message": str(error),
                            "model_family": "reduced_aircraft", "new_physical_mda": False,
                            "trace": trace}, status="hidden_failure")
                except (ValueError, RuntimeError, FloatingPointError) as error:
                    return self.record(candidate,
                        OutputObservation("interval", identity+":energy_prefix", lower=float(energy_wh)),
                        constraints, {"termination_reason": "numerical_integration_failure",
                            "failure_cause": type(error).__name__, "numerical_failure": True,
                            "model_family": "reduced_aircraft",
                            "new_physical_mda": False, "trace": trace}, status="hidden_failure")
                soc, energy_wh = next_soc, energy_wh+power_w*step_duration/3600.
                elapsed += step_duration
                peak_current = max(peak_current, envelope["peak_current_A"])
                minimum_voltage = min(minimum_voltage, envelope["minimum_voltage_V"])
                if soc < settings["minimum_soc"]:
                    constraints["soc"] = OutputObservation(
                        "interval", identity+":soc_monotone_bound", lower=settings["minimum_soc"]-soc,
                        lower_inclusive=True)
                    termination = "physical_soc_reserve_abort"
                    break
            trace.append({"stage": stage, "energy_prefix_Wh": float(energy_wh),
                          "soc": float(soc), "elapsed_s": float(elapsed), "power_W": float(power_w)})
            if termination != "complete":
                break
        objective = float(energy_wh) if termination == "complete" else OutputObservation(
            "interval", identity+":energy_prefix", lower=float(energy_wh))
        if termination == "complete":
            constraints["power"] = float(max_power_margin)
            constraints["soc"] = float(settings["minimum_soc"]-soc)
            constraints["current"] = float(peak_current/battery.CELL_MAX_A-1)
            constraints["voltage"] = float(1-minimum_voltage/battery.CELL_CUTOFF_V)
        else:
            if peak_current > battery.CELL_MAX_A:
                constraints["current"] = OutputObservation(
                    "interval", identity+":current_prefix", lower=float(peak_current/battery.CELL_MAX_A-1))
            if minimum_voltage < battery.CELL_CUTOFF_V:
                constraints["voltage"] = OutputObservation(
                    "interval", identity+":voltage_prefix", lower=float(1-minimum_voltage/battery.CELL_CUTOFF_V))
        return self.record(candidate, objective, constraints, {
            "termination_reason": termination, "model_family": "reduced_aircraft", "new_physical_mda": False,
            "mission_completed": termination == "complete", "trace": trace,
            "mass_kg": float(mass), "propeller_count": int(count), "propeller_diameter_m": float(diameter),
            "limitations": "ideal actuator disks; no wake-wing interaction, thermal, flutter or detailed stress"})

    def anchors(self):
        first = self.space.default().to_dict()
        second = self.space.canonical(self.candidate_class(
            {"wing_area": 12.5, "aspect_ratio": 10.5, "dep_diameter": .9, "prop_spacing": .85},
            {"prop_layout": "distributed", "propeller_count": 6})).to_dict()
        return [first, second]


def candidate_pool(problem, seed, count, seen=()):
    random = np.random.default_rng(seed)
    configurations = problem.space.configurations()
    result, keys, rejections = [], set(seen), 0
    for _ in range(count*1000):
        discrete = configurations[int(random.integers(len(configurations)))]
        candidate = problem.space.sample(random, discrete).to_dict()
        key = digest(candidate)
        if key in keys or any(value > 0 for value in problem.precheck(candidate).values()):
            rejections += 1
            continue
        keys.add(key)
        result.append(candidate)
        if len(result) == count:
            return result, rejections
    raise RuntimeError("Exact-screened candidate domain exhausted")
