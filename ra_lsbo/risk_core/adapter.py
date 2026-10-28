"""Explicit evidence adapters; original truth and observable records stay separate."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from ..budget import digest
from ..structured_encoder import EncoderSettings, StructuredFeatures
from .observations import OutputObservation, RunObservation


def observable_row(row, censor_positive=False):
    """A controlled sign-only sensor, applied to every method including LSBO."""
    result = deepcopy(row)
    risk = result["risk"]
    bounds = deepcopy(risk["evidence"].get("verified_constraint_bounds", {}))
    if censor_positive:
        for name, observed in risk["constraint_observed"].items():
            if (observed and name not in risk["analytical_constraints"]
                    and risk["constraints"][name] > 0):
                bounds[name] = {"lower": 0., "upper": None, "lower_inclusive": False,
                                "upper_inclusive": True,
                                "evidence_id": "controlled_sign_sensor:"+digest([row["evaluation"], name])}
                risk["constraints"][name] = None
                risk["constraint_observed"][name] = False
        if bounds and risk["objective_observed"]:
            # Legacy consumers cannot train an interval. They retain the finite
            # objective and known other outputs, without a numeric replacement.
            risk["state"] = "incomplete"
            risk["feasible_label"] = None
            risk["raw_status"] = "controlled_sign_only_constraint"
    risk["evidence"]["verified_constraint_bounds"] = bounds
    return result


def packet(row):
    risk = row["risk"]
    source = digest(row)
    bounds = risk["evidence"].get("verified_constraint_bounds", {})
    constraints = []
    for name in sorted(risk["constraints"]):
        if risk["constraint_observed"][name]:
            value = OutputObservation("exact", source+":"+name, value=risk["constraints"][name])
        elif name in bounds:
            value = OutputObservation("interval", **bounds[name])
        else:
            value = OutputObservation("missing", source+":"+name, reason="unobserved_output")
        constraints.append((name, value))
    objective = (OutputObservation("exact", source+":objective", value=risk["objective"])
                 if risk["objective_observed"] else
                 OutputObservation("missing", source+":objective", reason="unobserved_objective"))
    status = ("completed" if risk["hidden_success"] == 1 else
              "hidden_failure" if risk["hidden_success"] == 0 else risk["state"])
    if status not in {"completed", "hidden_failure", "infrastructure_error", "unknown"}:
        raise ValueError("Unsupported record needs an explicitly declared adapter")
    trace = risk["evidence"].get("controlled_stage_trace", [])
    passed = tuple(t["stage"] for t in trace if t["executed"] and not t["failed"])
    failed = [t["stage"] for t in trace if t["executed"] and t["failed"]]
    return RunObservation(
        design_id=digest(row["candidate"]), contract_id=row["evaluator_contract_sha256"],
        evidence_id=source, status=status, objective=objective, constraints=tuple(constraints),
        passed_stages=passed, failed_stage=failed[0] if failed else None,
        failure_cause=(risk["failure_modes"][0] if status == "hidden_failure" else None))


class CoreFeatures:
    """Replace only representation, with all preprocessing fit to the current prefix."""

    def __init__(self, problem, base, rows, seed=0, latent=False):
        self.problem, self.encoder = problem, None
        candidates = [r["candidate"] for r in rows]
        if latent:
            self.encoder = StructuredFeatures(
                base, [asdict(v) for v in problem.space.variables], seed,
                EncoderSettings(latent_dim=4, initial_epochs=120, update_epochs=20)).fit(rows)
        x = self._raw(candidates)
        self.offset, self.scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
        if latent:
            self.offset[-4:] = x[:, -4:].min(0)-.5
            self.scale[-4:] = x[:, -4:].max(0)+.5-self.offset[-4:]
        self.identity = digest({
            "history": digest(rows), "latent": latent,
            "snapshot": None if self.encoder is None else self.encoder.snapshot_id,
            "offset": self.offset.tolist(), "scale": self.scale.tolist()})

    def _raw(self, candidates):
        if self.encoder is not None:
            return self.encoder.transform_view(candidates, "latent").numpy()
        space = self.problem.space
        values = [self.problem.candidate_class(**c) for c in candidates]
        units = np.asarray([space.unit(c) for c in values])
        masks = np.asarray([space.mask(c.discrete) for c in values])
        parts = [units*masks, masks]
        for variable in space.discrete:
            active = np.asarray([space.active(variable, c.discrete) for c in values], float)
            indicator = np.zeros((len(values), len(variable.values)))
            indicator[np.arange(len(values)),
                      [variable.values.index(c.discrete[variable.name]) for c in values]] = 1.
            parts.extend((indicator*active[:, None], active[:, None]))
        return np.hstack(parts)

    def transform(self, candidates):
        x = (self._raw(candidates)-self.offset)/self.scale
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite core features")
        return x


def global_pool(problem, seed, count, seen):
    rng, result, keys = np.random.default_rng(seed), [], set(seen)
    for _ in range(count*200):
        candidate = problem.canonical(problem.space.sample(rng).to_dict())
        key = digest(candidate)
        if key in keys or any(v > 0 for v in problem.precheck(candidate).values()):
            continue
        result.append(candidate)
        keys.add(key)
        if len(result) == count:
            return result
    raise RuntimeError("Independent global candidate pool exhausted")
