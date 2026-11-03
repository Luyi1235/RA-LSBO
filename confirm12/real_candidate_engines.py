"""Six genuinely different BO policy *candidates* backed by the original H1 code.

Not a confirmatory authorization. Four methods reuse the frozen H1 code,
and two implement explicit experimental changes. CBOB-adapted remains
unavailable until a traceable reference port is independently accepted.
Nothing in this module imports an evaluator at module import or calls
AircraftCase.evaluate.

For science: the name "ra_lsbo" here is a PROPOSED comparison identity for
the existing H1 hybrid_onehot source, NOT a claim that the final candidate
method has been frozen or qualified.
"""
from dataclasses import dataclass
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.partial_bo.records import without_intervals
from ra_lsbo.structure_corpus import feature_snapshot
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import (
    HeterogeneousRiskPosterior, StableCorpusDiscoveryBO,
)
from .reverse_candidate import ReverseRepresentationCandidate


# A prospective label is NEVER silently substituted for the source H1 method.
H1_SOURCE = {
    "ra_lsbo": "hybrid_onehot",
    "full_latent": "corpus_stable",
    "full_onehot": "onehot_stable",
    "random": "random",
}
CANDIDATE_ARMS = (
    "ra_lsbo", "full_latent", "full_onehot",
    "reverse_representation", "interval_as_missing", "random",
)
EXTERNAL_UNQUALIFIED = ("cbob_adapted",)
ALL_REQUESTED = CANDIDATE_ARMS + EXTERNAL_UNQUALIFIED


class IntervalMissingHybridCandidate(StableCorpusDiscoveryBO):
    """H1 hybrid with interval outputs suppressed in GP fits only.

    Physical evidence and the incumbent/coverage eligibility remain RAW.  The
    already observed completion status is retained (and its derived binary
    completion likelihood remains active).  Only originally supplied objective
    or constraint interval *outputs* are suppressed, never converted to exact.
    Corpus training uses the identical raw structural feature schedule as H1.
    """
    def __init__(self, problem, base, seed, settings, output):
        super().__init__(problem, base, seed, settings, output,
                         allocation="onehot_risk")

    def _build_posterior(self, records):
        # Keep the exact original encoder/input design and feature identities.
        objective_features = feature_snapshot(
            self.problem, self.base, records, self.session)
        risk_features = PrefixFeatures(
            self.problem, self.base, records, self.seed, False, 30)
        training_only = [without_intervals(record) for record in records]
        posterior = HeterogeneousRiskPosterior(
            self.problem, training_only, self.seed, self.settings,
            objective_features, risk_features, dual_soc=False,
            view_policy="onehot_risk_gp_intervals_missing_v1")
        # The acquisition's real, settled feasible incumbent cannot disappear
        # just because one *surrogate* response was censored for this ablation.
        posterior.best = min(
            (r.observation.objective.value for r in records
             if r.usable_feasible and r.observation.objective.kind == "exact"),
            default=None)
        # Audit the two distinct causal prefixes, no output imputation.
        posterior.raw_history_digest = digest([r.to_dict() for r in records])
        posterior.masked_history_digest = digest(
            [r.to_dict() for r in training_only])
        posterior.history_digest = posterior.raw_history_digest
        return posterior

    def propose(self, records, global_pool, iteration):
        candidate, meta = super().propose(records, global_pool, iteration)
        model_records = [without_intervals(row) for row in records]
        masked_count = sum(
            1 for r in records for value in
            (r.observation.objective,) +
            tuple(x for _, x in r.observation.constraints)
            if value.kind == "interval")
        meta["interval_training_ablation"] = {
            "scope": "gp_objective_and_constraint_outputs_only",
            "completion_status_kept": True,
            "physical_history_and_incumbent_raw": True,
            "raw_prefix_digest": digest([r.to_dict() for r in records]),
            "gp_training_view_digest": digest(
                [r.to_dict() for r in model_records]),
            "masked_output_intervals": masked_count,
            "no_exact_imputation": True,
        }
        return candidate, meta


@dataclass(frozen=True)
class RealCandidateReceipt:
    case: str
    seed: int
    method: str
    origin_h1_method: str
    engine_class: str
    initial: int
    horizon: int
    pool: int
    nonphysical_instantiation: bool = True
    confirmed_for_experiment: bool = False
    source_mapping_reviewed: bool = False
    authorization_to_evaluate: bool = False


def create_engine(case, seed, method, output):
    """Construct genuine BO policy, but never grant production qualification."""
    if method not in ALL_REQUESTED:
        raise ValueError("Unregistered confirmatory BO method: " + str(method))
    if method == "cbob_adapted":
        raise RuntimeError(
            "CBOB-adapted is not implemented: upstream license, likelihood/EP, "
            "EICB, mixed hierarchy and feedback-contract parity must be reviewed")
    if case not in ("aircraft_ordinary", "aircraft_sparse"):
        raise ValueError("Unregistered aircraft task")
    if type(seed) is not int or seed < 0:
        raise ValueError("Unregistered seed")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refuse an existing method work directory")

    # Lazy import: no physical calls, VAE fits or GP fits at instantiation.
    from ra_lsbo.experiments.simplified_aircraft import dualview_pilot as h1
    from ra_lsbo.experiments.discovery_stages.methods import Labelled

    problem = h1.problem_for(case)
    profile = h1.profile_for("pilot")
    source = H1_SOURCE.get(method)
    if source is not None:
        engine = h1.engine_for(problem, source, seed, profile, output)
    else:
        settings = h1.model_settings("hybrid_onehot", profile)
        factory = (ReverseRepresentationCandidate if method == "reverse_representation"
                   else IntervalMissingHybridCandidate)
        engine = Labelled(factory(problem, h1.BASE, seed, settings, output), method)
        source = "H1_hybrid_onehot_modified_" + method
    if (profile["initial"], profile["horizon"], profile["pool"]) != (12, 18, 128):
        raise RuntimeError("Expected the frozen H1 12/18/128 pilot profile")

    receipt = RealCandidateReceipt(
        case=case, seed=seed, method=method,
        origin_h1_method=source,
        engine_class=type(getattr(engine, "engine", engine)).__name__,
        initial=profile["initial"], horizon=profile["horizon"],
        pool=profile["pool"])
    return problem, engine, receipt


def qualification_report():
    """Source-backed *implementation* inventory, not production readiness."""
    return {
        method: {
            "engine_construction": ("real_H1_or_candidate"
                                    if method in CANDIDATE_ARMS
                                    else "not_implemented"),
            "source_method": H1_SOURCE.get(method),
            "confirmatory_qualified": False,
            "physical_evaluation_enabled": False,
        }
        for method in ALL_REQUESTED
    }
