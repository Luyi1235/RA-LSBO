"""Declared benchmark extensions and a reduced electric-aircraft design case."""

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import numpy as np

from ...budget import digest
from ...partial_bo.problems import DevelopmentProblem, HierarchicalBranin, SimplifiedElectricAircraft
from ...partial_bo.records import DesignRecord
from ...risk_core.observations import OutputObservation


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class MathematicalCase(HierarchicalBranin):
    """Our constrained, partially observed extensions, not published test replicas."""

    def __init__(self, base, family="branin", difficulty="sparse"):
        if family not in {"branin", "rosenbrock"} or difficulty not in {"ordinary", "sparse"}:
            raise ValueError("Unknown mathematical case")
        super().__init__(base)
        self.family, self.difficulty = family, difficulty
        self.name = "Declared hierarchical %s extension (%s)" % (family, difficulty)
        self.settings = {**self.settings, "family": "paper_%s_extension_v1" % family,
                         "difficulty": difficulty,
                         "ellipse_radii": [.42, .40] if difficulty == "ordinary" else [.16, .20],
                         "center": [.68, .25] if family == "branin" else [.72, .70],
                         "source": source_identity(), "upstream_replica": False}
        self.contract_id = digest({"parent": self.contract_id, "settings": self.settings})

    def evaluate(self, raw):
        candidate = self.canonical(raw)
        continuous, discrete = candidate["continuous"], candidate["discrete"]
        augmented = discrete["architecture"] == "augmented"
        shift = .035*(discrete["module"]-2) if augmented else 0.
        position = np.array([continuous["x"]+shift, continuous["y"]])
        region = float(np.sum(((position-self.settings["center"])/self.settings["ellipse_radii"])**2)-1)
        resource = float(continuous["trim"]+.12*discrete["module"]-1.05) if augmented else -.2
        identity = digest([self.contract_id, candidate, "region_threshold"])
        diagnostics = {"termination_reason": "complete" if region <= 0 else "controlled_region_abort",
                       "model_family": self.settings["family"], "new_physical_mda": False}
        if region > 0:
            return self.record(candidate,
                OutputObservation("missing", identity+":objective", reason="region_stage_aborted"),
                {"region": OutputObservation("interval", identity+":region", lower=0., lower_inclusive=False),
                 "resource": resource}, diagnostics)
        if self.family == "branin":
            x, y = 15*position[0]-5, 15*position[1]
            objective = (y-5.1/(4*np.pi**2)*x*x+5/np.pi*x-6)**2+10*(1-1/(8*np.pi))*np.cos(x)+10
            if augmented:
                objective += 2*(continuous["trim"]-.25*discrete["module"])**2
        else:
            values = np.r_[position, continuous["trim"]] if augmented else position
            values = 4*values-2
            objective = np.sum(100*(values[1:]-values[:-1]**2)**2+(1-values[:-1])**2)
        return self.record(candidate, float(objective), {"region": region, "resource": resource}, diagnostics)

    def anchors(self):
        raw = self.space.default().to_dict()
        raw["continuous"].update(x=self.settings["center"][0], y=self.settings["center"][1])
        other = {"continuous": {**raw["continuous"], "trim": .75},
                 "discrete": {"architecture": "augmented", "module": 2}}
        return [self.canonical(raw), self.canonical(other)]


@dataclass(frozen=True)
class FlightScenario:
    headwind_m_s: float = 0.
    payload_delta_kg: float = 0.
    capacity_retention: float = 1.
    initial_soc: float = .95

    def __post_init__(self):
        values = (self.headwind_m_s, self.payload_delta_kg, self.capacity_retention, self.initial_soc)
        if (not all(np.isfinite(values)) or not -4 <= self.headwind_m_s <= 8
                or not -40 <= self.payload_delta_kg <= 60
                or not .9 <= self.capacity_retention <= 1. or not .9 <= self.initial_soc <= .98):
            raise ValueError("Scenario outside the declared screening uncertainty support")


def flight_scenarios(seed, count):
    if type(count) is not int or count < 1:
        raise ValueError("Positive scenario count required")
    random = np.random.default_rng(seed)
    draws = random.uniform(size=(count, 4))
    return [FlightScenario(-4+12*row[0], -40+100*row[1], .9+.1*row[2], .9+.08*row[3])
            for row in draws]


class AircraftCase(SimplifiedElectricAircraft):
    """Extended point-mass mission model; assumptions are not certification data.

    Headwind changes cruise duration at fixed airspeed. Battery parallel strings
    change both mass and cell loading. Capacity retention changes cell Ah only;
    resistance/thermal ageing and wind during other flight phases are omitted.
    """

    def __init__(self, base, difficulty="sparse", scenario=None, full_trajectory=False,
                 integration_step=20.):
        if difficulty not in {"ordinary", "sparse"}:
            raise ValueError("Unknown aircraft mission")
        super().__init__(base, integration_step)
        self.difficulty, self.scenario = difficulty, scenario or FlightScenario()
        self.full_trajectory = full_trajectory
        if type(full_trajectory) is not bool:
            raise ValueError("Boolean full-trajectory switch required")
        variables = [dict(name=variable.name, kind=variable.kind, default=variable.default,
                          **({"bounds": variable.bounds} if variable.kind == "continuous" else
                             {"values": variable.values}),
                          **({"active_when": variable.active_when} if variable.active_when else {}))
                     for variable in self.space.variables]
        variables += [{"name": "cruise_airspeed", "kind": "continuous", "default": 50., "bounds": [44., 58.]},
                      {"name": "parallel_strings", "kind": "ordinal", "default": 40, "values": [32, 40, 48]}]
        settings = {**self.settings, "family": "paper_reduced_aircraft_v1", "difficulty": difficulty,
                    "nominal_cruise_distance_m": 100000. if difficulty == "ordinary" else 140000.,
                    "soc_reserve_requirement": .2, "cell_mass_kg": .048, "pack_cell_mass_fraction": .75,
                    "scenario": vars(self.scenario), "full_trajectory": full_trajectory,
                    "uncertainty_status": "declared screening assumptions, not empirical distributions",
                    "source": source_identity()}
        # A fresh battery module belongs to this instance; no shared globals change.
        DevelopmentProblem.__init__(self, base, variables, settings)
        self.name = "Extended reduced electric-aircraft (%s)" % difficulty
        self._nominal_settings = dict(self.settings)
        self._base_cell_ah = self.battery.CELL_AH

    def evaluate(self, raw):
        candidate = self.canonical(raw)
        scenario, nominal = self.scenario, self._nominal_settings
        airspeed, parallel = candidate["continuous"]["cruise_airspeed"], candidate["discrete"]["parallel_strings"]
        cells = 128*parallel
        battery_mass_delta = (cells-5120)*nominal["cell_mass_kg"]/nominal["pack_cell_mass_fraction"]
        self.battery.TOTAL_CELLS = cells
        self.battery.CELL_AH = self._base_cell_ah*scenario.capacity_retention
        self.settings = {**nominal, "fixed_mass_kg": nominal["fixed_mass_kg"]+battery_mass_delta+scenario.payload_delta_kg,
                         "cruise_speed_m_s": airspeed,
                         "cruise_distance_m": nominal["nominal_cruise_distance_m"]*airspeed/(airspeed-scenario.headwind_m_s),
                         "initial_soc": scenario.initial_soc,
                         "minimum_soc": 0. if self.full_trajectory else nominal["soc_reserve_requirement"]}
        try:
            record = super().evaluate(candidate)
        finally:
            self.settings = nominal
            self.battery.TOTAL_CELLS = 5120
            self.battery.CELL_AH = self._base_cell_ah
        constraints = dict(record.observation.constraints)
        final_soc = record.diagnostics["trace"][-1]["soc"] if record.diagnostics.get("trace") else None
        if self.full_trajectory:
            if record.observation.objective.kind == "exact":
                constraints["soc"] = OutputObservation("exact", record.observation.evidence_id+":validated_soc",
                    value=float(nominal["soc_reserve_requirement"]-final_soc))
            else:
                # Stopping before mission end supplies no exact final SOC.
                constraints["soc"] = OutputObservation("missing", record.observation.evidence_id+":censored_soc",
                                                       reason="mission_did_not_complete")
        diagnostics = {**record.diagnostics, "scenario": vars(scenario), "final_soc": final_soc,
                       "final_soc_exact": record.observation.objective.kind == "exact",
                       "parallel_strings": parallel, "total_cells": cells,
                       "cruise_airspeed_m_s": airspeed, "nominal_cruise_distance_m": nominal["nominal_cruise_distance_m"],
                       "model_limitations": "no aeroelastic/thermal/wake model; capacity-only ageing; cruise-only steady wind"}
        packet = replace(record.observation, constraints=tuple(sorted(constraints.items())))
        return DesignRecord(record.candidate, packet, diagnostics)

    def anchors(self):
        return [self.canonical(raw) for raw in super().anchors()]


CASES = ("branin_ordinary", "branin_sparse", "rosenbrock_ordinary", "rosenbrock_sparse",
         "aircraft_ordinary", "aircraft_sparse")


def make_problem(base, case, **kwargs):
    if case not in CASES:
        raise ValueError("Unregistered experiment case")
    family, difficulty = case.split("_")
    return (AircraftCase(base, difficulty, **kwargs) if family == "aircraft"
            else MathematicalCase(base, family, difficulty))
