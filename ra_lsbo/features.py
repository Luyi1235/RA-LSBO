"""Load existing LSBO math modules under an isolated package name, without MDA."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .records import RiskEvaluation, finite_number


BASE_FILES = ("__init__.py", "space.py", "records.py", "representation.py", "surrogate.py")


def load_base(project):
    folder = Path(project).resolve() / "lsbo"
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in BASE_FILES}
    identity = hashlib.sha256((str(folder) + json.dumps(hashes, sort_keys=True)).encode()).hexdigest()
    package = "_ra_base_" + identity[:16]
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, folder / "__init__.py", submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    modules = {name: importlib.import_module(package + "." + name)
               for name in ("space", "records", "representation", "surrogate")}
    if any(hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha for name, sha in hashes.items()):
        raise RuntimeError("Base source changed while importing")
    return modules, hashes


def validate_history(rows):
    indices = [row["evaluation"] for row in rows]
    if any(type(i) is not int or i <= 0 for i in indices) or indices != sorted(set(indices)):
        raise ValueError("History must have increasing unique positive evaluation indices")
    for key in ("evaluator_contract_sha256", "source_journal_sha256"):
        if len({row.get(key) for row in rows}) > 1:
            raise ValueError("Mixed history identity: " + key)
    for row in rows:
        RiskEvaluation(**row["risk"])
        if any(issue in row["risk"]["issues"] for issue in (
                "legacy_feasibility_disagrees", "status_constraint_mismatch")):
            raise ValueError("Resolve inconsistent source labels before training")


def history_identity(rows):
    return hashlib.sha256(json.dumps(
        [{key: row.get(key) for key in ("evaluation", "candidate", "risk",
                                       "evaluator_contract_sha256", "source_journal_sha256")}
         for row in rows], sort_keys=True, allow_nan=False).encode()).hexdigest()


class LatentFeatures:
    def __init__(self, base_project, variables, seed=0, latent_dim=4):
        self.modules, self.base_hashes = load_base(base_project)
        self.space = self.modules["space"].DesignSpace.from_dict({"variables": variables})
        self.representation = self.modules["representation"].Representation(
            self.space, seed=seed, latent_dim=latent_dim)
        self.training_indices = None
        self.history_id = None
        self.snapshot_id = None
        self.variables = variables

    def candidate(self, raw):
        for kind, specs in (("continuous", self.space.continuous), ("discrete", self.space.discrete)):
            if set(raw[kind]) != {v.name for v in specs}:
                raise ValueError("Candidate variables differ from the frozen schema")
        for variable in self.space.continuous:
            value = raw["continuous"][variable.name]
            if not finite_number(value):
                raise ValueError("Candidate contains non-finite continuous values")
        candidate = self.modules["space"].Candidate(**raw)
        canonical = self.space.canonical(candidate)
        for variable in self.space.continuous:
            value = raw["continuous"][variable.name]
            if self.space.active(variable, canonical.discrete) and not variable.bounds[0] <= value <= variable.bounds[1]:
                raise ValueError("Active candidate coordinate outside bounds; refusing silent clipping")
        return canonical

    def fit(self, rows, epochs=60):
        validate_history(rows)
        if type(epochs) is not int or epochs < 1:
            raise ValueError("Positive integer representation epochs required")
        observations = []
        for row in rows:
            risk = row["risk"]
            result = self.modules["records"].Evaluation(
                objective=risk["objective"], solver_success=risk["objective_observed"],
                constraints=risk["constraints"], status=risk["raw_status"])
            observations.append(self.modules["records"].Observation(
                row["evaluation"], self.candidate(row["candidate"]), result, row.get("phase", "offline")))
        self.representation.fit(observations, epochs=epochs)
        self.training_indices = [row["evaluation"] for row in rows]
        self.history_id = history_identity(rows)
        self.representation_version = self.representation.version
        state = hashlib.sha256(json.dumps(
            {"schema": self.variables, "indices": self.training_indices,
             "history_id": self.history_id,
             "base_hashes": self.base_hashes, "version": self.representation.version,
             "y_mean": self.representation.y_mean, "y_std": self.representation.y_std},
            sort_keys=True).encode())
        for key, value in self.representation.model.state_dict().items():
            state.update(key.encode())
            state.update(value.detach().cpu().numpy().tobytes())
        self.snapshot_id = state.hexdigest()
        return self

    def transform(self, candidates):
        if self.snapshot_id is None:
            raise RuntimeError("Representation must be fitted before prediction")
        if self.representation.version != self.representation_version:
            raise RuntimeError("Base representation changed outside adapter; refit features and heads")
        if not candidates:
            return torch.empty((0, 2 * len(self.space.continuous) +
                                self.representation.model.latent_dim), dtype=torch.double)
        features = self.representation.features([self.candidate(c) for c in candidates])
        if not np.isfinite(features.numpy()).all():
            raise FloatingPointError("Nonfinite representation features")
        return features
