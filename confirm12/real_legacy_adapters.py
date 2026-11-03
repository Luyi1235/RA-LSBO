"""Real historical dual-view optimizer bridge for offline interface qualification.

This module instantiates four actual, existing frozen H1 policy classes.
It does NOT call any aircraft evaluator, start a campaign, or qualify any
of the seven proposed 151,040-call confirmatory treatments. In particular,
'hybrid_onehot' MUST NOT be silently relabeled as a newly qualified 'ra_lsbo'.
"""
from dataclasses import dataclass
from pathlib import Path

LEGACY_METHODS = frozenset((
    "hybrid_onehot",
    "corpus_stable",
    "onehot_stable",
    "random",
))
# The actual confirmatory seven-arm protocol has distinct identifiers.
CONFIRMATORY_ARM_NAMES = frozenset((
    "ra_lsbo",
    "full_latent",
    "full_onehot",
    "reverse_representation",
    "interval_as_missing",
    "random",
    "cbob_adapted",
))


@dataclass(frozen=True)
class HistoricalAdapterProbe:
    case: str
    seed: int
    legacy_method: str
    engine_type: str
    horizon: int
    initial: int
    pool: int
    record_kind: str = "offline_interface_probe_no_evaluation"
    experiment_qualified: bool = False


def build_historical_engine(case, seed, legacy_method, output):
    """Construct a real H1 optimizer with exactly its historical pilot settings.

    No candidate proposals, GP/VAE fitting, file writes or truth evaluations
    occur here. These are only nonphysical interface-qualification objects.
    Confirmatory labels deliberately cannot be passed to this function.
    """
    if legacy_method not in LEGACY_METHODS:
        raise ValueError(
            "Not a frozen historical H1 method; no automatic confirmatory remapping: "
            + str(legacy_method)
        )
    if case not in ("aircraft_ordinary", "aircraft_sparse"):
        raise ValueError("Unknown historical reduced-aircraft task")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new historical adapter output path")

    # Lazy import prevents import-time optimization, fits and aircraft calls.
    from ra_lsbo.experiments.simplified_aircraft import dualview_pilot as h1

    # Reuse the exact historical implementation and settings; don't create
    # a mathematical surrogate for one of the confirmation methods.
    problem = h1.problem_for(case)
    profile = h1.profile_for("pilot")
    engine = h1.engine_for(problem, legacy_method, seed, profile, output)
    info = HistoricalAdapterProbe(
        case=case,
        seed=seed,
        legacy_method=legacy_method,
        engine_type=type(engine.engine).__name__
            if hasattr(engine, "engine") else type(engine).__name__,
        horizon=profile["horizon"],
        initial=profile["initial"],
        pool=profile["pool"],
    )
    if (info.horizon, info.initial, info.pool) != (18, 12, 128):
        raise RuntimeError("Unexpected historical pilot optimizer parameters")
    return problem, engine, info


def confirmation_readiness():
    """Give explicit, non-fabricated status of the seven requested methods."""
    return {
        "ra_lsbo": "BLOCKED: main candidate math/adapter not independently frozen",
        "full_latent": "BLOCKED: historical corpus_stable exists; confirm adapter not qualified",
        "full_onehot": "BLOCKED: historical onehot_stable exists; confirm adapter not qualified",
        "reverse_representation": "BLOCKED: candidate feature routing exists; end-to-end and metadata not qualified",
        "interval_as_missing": "BLOCKED: offline evidence transformation exists; policy isolation not qualified",
        "random": "BLOCKED: historical random exists; registered q_A/feedback not qualified",
        "cbob_adapted": "BLOCKED: upstream/license/EP/EICB and mixed-domain port not qualified",
    }
