"""Frozen current RA model, matched ablations, and a declared EIC adapter."""

from dataclasses import asdict

import numpy as np

from ...budget import digest
from ...partial_bo.records import without_intervals
from ...soc_revision.acquisition import quantities
from ...soc_revision.optimizer import RevisionSettings, RiskRefiningBO
from ...soc_revision.posterior import RiskPosterior


ARMS = ("ra_lsbo", "original_lsbo", "onehot_local", "no_intervals", "no_local", "conventional_eic", "random")
POSTERIOR_ARMS = ("ra_lsbo", "onehot_local", "no_intervals", "no_local", "conventional_eic")


def settings_for(method, profile):
    if method not in POSTERIOR_ARMS:
        raise ValueError("This method does not use the experiment posterior")
    smoke = profile["name"] == "smoke"
    return RevisionSettings(horizon=profile["horizon"], average_kernel=False,
        probability_exploitation=True, latent=method not in {"onehot_local", "conventional_eic"},
        use_intervals=method not in {"no_intervals", "conventional_eic"},
        local=method not in {"no_local", "conventional_eic"}, encoder_epochs=profile["encoder_epochs"],
        continuous_grid=(.15, .4) if smoke else (.05, .15, .4, 1.),
        discrete_grid=(1.5,) if smoke else (.5, 1.5),
        local_branches=1 if smoke else 4, local_starts=1 if smoke else 2,
        local_points=3 if smoke else 12, local_maxiter=4 if smoke else 25)


def posterior_for(problem, base, records, seed, method, profile):
    settings = settings_for(method, profile)
    training = records if settings.use_intervals else [without_intervals(row) for row in records]
    return RiskPosterior(problem, base, training, seed, settings)


class ConventionalEIC:
    """One-hot, numeric-output-only EI*PoF with a common completion classifier.

    Before finding a feasible incumbent, maximize joint predicted feasibility.
    This is an explicitly adapted baseline, not an official CBOB/SCBO solver.
    """

    def __init__(self, problem, base, seed, profile):
        self.problem, self.base, self.seed, self.profile = problem, base, seed, profile
        self.settings = settings_for("conventional_eic", profile)
        self.pending, self.iteration, self.prefix = None, 0, None

    def propose(self, records, pool, iteration):
        prefix = digest([row.to_dict() for row in records])
        if self.pending is not None or iteration != self.iteration or (self.prefix and prefix != self.prefix):
            raise ValueError("Baseline state differs from the settled prefix")
        seen = {row.observation.design_id for row in records}
        if (not pool or len({digest(raw) for raw in pool}) != len(pool)
                or any(digest(raw) in seen or raw != self.problem.canonical(raw)
                       or any(value > 0 for value in self.problem.precheck(raw).values()) for raw in pool)):
            raise ValueError("A unique, canonical, unseen, screened pool is required")
        posterior = posterior_for(self.problem, self.base, records, self.seed, "conventional_eic", self.profile)
        values = quantities(posterior.predict(pool), posterior.best, 1., .5)
        scores = values["joint"] if values["ei"] is None else values["joint"]*values["ei"]
        index = int(np.argmax(scores))
        self.pending = (pool[index], list(records), prefix)
        return pool[index], {"mode": "joint_feasibility" if values["ei"] is None else "EI_times_joint_PoF",
            "settings": asdict(self.settings), "scores": scores.tolist(), "joint": values["joint"].tolist(),
            **posterior.diagnostics()}

    def accept(self, record):
        if (self.pending is None or record.candidate != self.pending[0]
                or record.observation.contract_id != self.problem.contract_id
                or digest([row.to_dict() for row in self.pending[1]]) != self.pending[2]):
            raise ValueError("Feedback does not match the pending EIC proposal")
        self.prefix = digest([row.to_dict() for row in self.pending[1]+[record]])
        self.pending, self.iteration = None, self.iteration+1


def optimizer_for(problem, base, seed, method, profile):
    if method == "conventional_eic":
        return ConventionalEIC(problem, base, seed, profile)
    if method not in POSTERIOR_ARMS:
        raise ValueError("Original LSBO and random search have separate runners")
    return RiskRefiningBO(problem, base, seed, settings_for(method, profile))
