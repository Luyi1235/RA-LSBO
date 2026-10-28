"""Opt-in full-system strategy with explicit mechanism scores and censored labels."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from .budget import digest
from .calibration import CalibratedRiskHeads
from .constraint_risk import ConstraintRiskHeads
from .features import history_identity, load_base
from .legacy import normalize_evaluation
from .records import HIDDEN_MODES
from .structured_encoder import EncoderSettings, StructuredFeatures
from .structured_models import StructuredResponseHeads, StructuredRiskHeads


METHODS = ("ra_full", "ra_no_mechanisms", "ra_onehot_risk", "ra_no_calibration",
           "ra_no_boundary", "cbo", "lsbo", "random")


class MechanismRiskHeads(ConstraintRiskHeads):
    """Conservative raw minimum, calibrated separately; never a joint probability."""

    def __init__(self, risks, responses, required_constraints, use_mechanisms=True):
        super().__init__(risks, responses, required_constraints)
        self.use_mechanisms = use_mechanisms
        self.active_mechanisms = tuple(
            name for name in HIDDEN_MODES
            if use_mechanisms and risks.diagnostics[name]["status"] == "fitted_raw")
        self.hidden_success_available = (
            use_mechanisms and risks.diagnostics["hidden_success"]["status"] == "fitted_raw")
        self.missing_mechanisms = tuple(name for name in HIDDEN_MODES if name not in self.active_mechanisms)
        self.model_id = digest({
            "constraint_model_id": self.model_id, "rule": "mechanism_minimum_v1",
            "use_mechanisms": use_mechanisms, "active_mechanisms": self.active_mechanisms,
            "hidden_success_available": self.hidden_success_available})
        self.diagnostics["final_feasibility"].update(
            raw_score_rule="minimum_RF_constraints_and_available_mechanism_survivals",
            mechanism_decomposition_enabled=use_mechanisms,
            active_mechanisms=list(self.active_mechanisms),
            missing_mechanisms=list(self.missing_mechanisms),
            mechanism_data_complete=len(self.active_mechanisms) == len(HIDDEN_MODES))

    def predict(self, candidates):
        raw = super().predict(candidates)
        components = [np.asarray(raw["final_feasibility"])]
        if self.hidden_success_available:
            components.append(np.asarray(raw["hidden_success"]))
        components.extend(1.-np.asarray(raw[name]) for name in self.active_mechanisms)
        raw["final_feasibility"] = np.minimum.reduce(components).tolist()
        return raw

    def mechanism_uncertainty(self, raw):
        n = len(raw["final_feasibility"])
        return np.maximum.reduce(
            [np.zeros(n)]+[4*np.asarray(raw[name])*(1.-np.asarray(raw[name]))
                           for name in self.active_mechanisms])


@dataclass(frozen=True)
class SystemSettings:
    calls: int = 12
    calibration_window: int = 120
    update_every: int = 4
    pool_size: int = 512
    gp_steps: int = 45
    encoder_epochs: int = 600
    latent_dim: int = 8
    incremental_epochs: int = 20
    discovery_cap: int = 4
    boundary_cap: int = 4
    failure_cap: int = 6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError("System settings must be positive integers: "+name)
        if self.calls > 500 or self.failure_cap > self.calls:
            raise ValueError("Invalid complete-run budget")

    def encoder(self):
        return EncoderSettings(latent_dim=self.latent_dim, initial_epochs=self.encoder_epochs,
                               update_epochs=self.incremental_epochs)


class FullSystemLearning:
    def __init__(self, method, settings, seed=0):
        if method not in METHODS or method in ("lsbo", "random"):
            raise ValueError("A model-based system method is required")
        self.method, self.settings, self.seed = method, settings, seed
        self.features = None

    def fit(self, rows, variables, base_project, expensive_constraints):
        holds_out = self.method != "cbo"
        training = rows[:-self.settings.calibration_window] if holds_out else rows
        calibration = (rows[-self.settings.calibration_window:]
                       if self.method not in ("cbo", "ra_no_calibration") else [])
        if len(training) < 3:
            raise ValueError("Insufficient chronological training prefix")
        if self.features is None:
            self.features = StructuredFeatures(base_project, variables, self.seed, self.settings.encoder()).fit(training)
            update = "reset"
        else:
            self.features.update(training)
            update = "append_only_incremental"
        risks = StructuredRiskHeads(
            training, self.features, self.seed,
            view="hierarchical_onehot" if self.method == "ra_onehot_risk" else "latent")
        responses = StructuredResponseHeads(training, self.features, self.seed, self.settings.gp_steps)
        combined = MechanismRiskHeads(
            risks, responses, expensive_constraints, use_mechanisms=self.method != "ra_no_mechanisms")
        calibrated = CalibratedRiskHeads(combined, calibration) if calibration else None
        metadata = {
            "method": self.method, "training_indices": self.features.training_indices,
            "training_history_id": self.features.history_id, "encoder_update": update,
            "objective_indices": self.features.objective_indices,
            "representation_snapshot": self.features.snapshot_id,
            "encoder_settings": asdict(self.features.settings),
            "risk_feature_layout": risks.features.layout(), "response_feature_layout": self.features.layout(),
            "model_id": combined.model_id, "active_mechanisms": list(combined.active_mechanisms),
            "risk_diagnostics": combined.diagnostics, "response_diagnostics": responses.diagnostics,
            "calibration_indices": [] if calibrated is None else calibrated.calibration_indices,
            "unused_holdout_indices": ([r["evaluation"] for r in rows[-self.settings.calibration_window:]]
                                       if self.method == "ra_no_calibration" else []),
            "calibration_id": None if calibrated is None else calibrated.calibration_id,
            "calibration_records": [] if calibrated is None else calibrated.calibration_records,
            "calibration": None if calibrated is None else calibrated.diagnostics}
        return (combined, responses, calibrated), metadata


def mechanism_support(rows):
    return {
        name: {"positive": sum(r["risk"]["hidden_labels"][name] == 1 for r in rows),
               "negative": sum(r["risk"]["hidden_labels"][name] == 0 for r in rows),
               "unknown": sum(r["risk"]["hidden_labels"][name] is None for r in rows)}
        for name in HIDDEN_MODES}


class FiveModeProblem:
    """Controlled serial process; its labels are never aircraft observations."""

    required_constraints = ("packing", "envelope", "load")
    analytical_constraints = ("packing",)
    scenario = "five_mode_controlled"
    coverage_variables = ("layout",)
    solver_domain_metric = None

    def __init__(self, base_project):
        from .synthetic import VARIABLES
        modules, _ = load_base(base_project)
        self.variables = deepcopy(VARIABLES)
        self.space = modules["space"].DesignSpace.from_dict({"variables": self.variables})
        self.candidate_class = modules["space"].Candidate
        self.contract_id = digest({
            "schema": self.variables, "equations": "five_serial_modes_v1",
            "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        self.name = "Controlled five-mode hierarchy"

    def canonical(self, raw):
        value = self.space.canonical(self.candidate_class(**raw))
        for variable in self.space.continuous:
            if self.space.active(variable, value.discrete):
                if not variable.bounds[0] <= raw["continuous"][variable.name] <= variable.bounds[1]:
                    raise ValueError("Active coordinate outside controlled domain")
        return value.to_dict()

    def precheck(self, raw):
        value = self.canonical(raw)
        c, d = value["continuous"], value["discrete"]
        return {"packing": c["spacing"]*(d["count"]-1)-.9 if d["layout"] == "distributed" else -1.}

    def evaluate(self, raw):
        value = self.canonical(raw)
        c, d = value["continuous"], value["discrete"]
        constraints = self.precheck(value)
        if constraints["packing"] > 0:
            return normalize_evaluation({
                "status": "analytical_constraint_rejected", "constraints": constraints,
                "solver_success": False, "objective": None}, self.required_constraints,
                self.analytical_constraints).to_dict()
        predicates = (c["x"] < .08,
                      d["layout"] == "distributed" and c["y"] < .12,
                      c["x"] > .88 and d["layout"] == "distributed",
                      c["x"]+c["y"] < .62,
                      c["y"] > .86)
        failed = next((i for i, value in enumerate(predicates) if value), None)
        objective = None
        if failed is None:
            constraints.update(
                envelope=((c["x"]-.69)/.20)**2+((c["y"]-.63)/.25)**2-1.,
                load=c["x"]+c["y"]-1.48)
            objective = (100+40*(c["x"]-.81)**2+25*(c["y"]-.72)**2
                         +(.1*(d["count"]-2)+2*(c["spacing"]-.18)**2
                           if d["layout"] == "distributed" else 1.))
        status = (HIDDEN_MODES[failed] if failed is not None else
                  "constraint_violation" if any(v > 0 for v in constraints.values()) else "ok")
        risk = normalize_evaluation({
            "status": status, "constraints": constraints, "objective": objective,
            "solver_success": objective is not None, "metrics": {"mda_evaluated": True}},
            self.required_constraints, self.analytical_constraints).to_dict()
        # Only this process exposes an explicit, verified serial-stage execution trace.
        if failed is not None:
            for name in HIDDEN_MODES[:failed]:
                risk["hidden_labels"][name] = 0
        risk["evidence"]["controlled_stage_trace"] = [
            {"stage": name, "executed": failed is None or i <= failed,
             "failed": i == failed if failed is not None else False}
            for i, name in enumerate(HIDDEN_MODES)]
        return risk
