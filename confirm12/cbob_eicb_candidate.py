"""Independent, mixed-hierarchy CBOB-adapted *candidate*, not author-code port.

Reference: Wang & Li, AAAI 2024, CBOB; public mathematical EICB balance
formula and partial-observation GP framework. This fresh implementation uses
this project's auditable exact/interval EP + explicit one-hot grouped kernels
instead of the authors' GPy / heteroscedastic training. It samples the common
registered *legal* candidate pool instead of Sobol points in an unconstrained
continuous domain. Those substantial differences require an independent
methodology review before anyone can call this a qualified external baseline.

No aircraft or RCAIDE evaluator is called anywhere in this module.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.special import ndtr

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.experiments.discovery_stages.policies import filter_near
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import (
    HeterogeneousRiskPosterior,
)

FORMULA = "independent_EICB_balanced_feasibility_v1"
BETA = 2.0
JITTER = 0.01
EPS = 1e-12
NORMAL_FACTOR = 1.0 / math.sqrt(2.0*math.pi)


def _normal_pdf(z):
    return NORMAL_FACTOR*np.exp(-0.5*np.asarray(z, float)**2)


def balanced_probability(mean, variance, *, beta=BETA, threshold=0.):
    """Compute the explicit per-constraint EICB balanced probability on a grid.

    Balance is the capped max nonnegative 'gain' on the *same fixed candidate
    grid*. This adaptation is NOT the original 1,000 Sobol reference design.
    Treat nonfinite heads as a model fault; never turn them into a feasible point.
    """
    m = np.asarray(mean, dtype=float)
    v = np.asarray(variance, dtype=float)
    if (m.ndim != 1 or v.shape != m.shape or len(m) < 1 or
            not np.isfinite(m).all() or not np.isfinite(v).all() or
            np.any(v < 0) or not math.isfinite(beta) or beta < 0):
        raise ValueError("Invalid finite Gaussian constraint head")
    sigma = np.sqrt(np.maximum(v, EPS))
    center = (threshold-m)/sigma
    low = center-beta
    high = center+beta
    c0, cl, ch = ndtr(center), ndtr(low), ndtr(high)
    # Original published expression evaluates *densities* at all three bounds.
    p0, pl, ph = (_normal_pdf(center)/sigma, _normal_pdf(low)/sigma,
                  _normal_pdf(high)/sigma)
    gain = ((m-threshold)*(2*c0-cl-ch)
            - sigma*(2*p0-pl-ph)
            + beta*sigma*(ch-cl))*c0
    if not np.isfinite(gain).all():
        raise FloatingPointError("Nonfinite EICB balance value")
    balance = max(1e-3, min(float(np.max(np.maximum(gain, 0.))), 1e5))
    p = np.clip(c0 + np.clip(gain/balance, 0., 1.), 0., 1.)
    return p, {"balance": balance,
               "cdf_standard": c0.tolist(),
               "enhancement_max": float(np.max(np.maximum(gain, 0.))),
               "beta": beta}


def expected_improvement(mean, variance, best, *, jitter=JITTER):
    m = np.asarray(mean, dtype=float)
    v = np.asarray(variance, dtype=float)
    if (m.ndim != 1 or v.shape != m.shape or
            not np.isfinite(m).all() or not np.isfinite(v).all() or
            np.any(v < 0) or not math.isfinite(float(best))):
        raise ValueError("Invalid objective GP or incumbent")
    sigma = np.sqrt(np.maximum(v, EPS))
    z = (float(best)-m-jitter)/sigma
    ei = sigma*(z*ndtr(z)+_normal_pdf(z))
    return np.maximum(ei, 0.)


def candidate_eicb(prediction, feasible_best):
    """Combine separate component heads; never approximate a probability by a sum."""
    rows = []
    for name, values in (
        ("completion", prediction["completion"]),
        *[(name, value) for name, value in sorted(prediction["constraints"].items())],
    ):
        p, diagnostics = balanced_probability(values["mean"], values["variance"])
        rows.append((name, p, diagnostics))
    if not rows:
        raise ValueError("At least one completion/constraint head required")
    count = len(rows[0][1])
    if any(len(row[1]) != count for row in rows):
        raise ValueError("Unequal posterior candidate grids")
    joint = np.ones(count)
    for _, p, _ in rows:
        joint *= p
    obj = prediction["objective"]
    if feasible_best is None or obj is None:
        score, mode = joint, "cold_balanced_feasibility"
    else:
        score = joint * expected_improvement(
            obj["mean"], obj["variance"], feasible_best)
        mode = "EICB_expected_improvement_times_balanced_feasibility"
    if not np.isfinite(score).all() or np.any(score < 0):
        raise FloatingPointError("Untrusted EICB acquisition scores")
    return score, {"mode": mode,
                   "feasibility_joint": joint.tolist(),
                   "component_balance": {name: diag for name, _, diag in rows},
                   "jitter": JITTER,
                   "formula": FORMULA,
                   "not_upstream_code": True,
                   "independent_constraint_heads_assumed": True}


class CBOBAdaptedResearchCandidate:
    """Stateful EICB grid selector with actual mixed-likelihood EP head fitting.

    Uses one true 12-initial/18-online history, all caller-supplied; it never
    evaluates the candidate or conceals missing/unknown observations.
    """
    def __init__(self, problem, base, seed, settings, output):
        self.problem, self.base, self.seed = problem, base, seed
        self.settings = settings
        self.output = output
        self.pending = None
        self.next_iteration = 0
        self.expected_prefix = None
        self.first_count = None
        self.failed = False

    def propose(self, records, global_pool, iteration):
        prefix = digest([r.to_dict() for r in records])
        if self.failed or self.pending is not None or iteration != self.next_iteration:
            raise ValueError("CBOB partial-record proposal state changed")
        if self.first_count is None:
            if iteration != 0 or len(records) != 12:
                raise ValueError("Twelve initial records required for first BO proposal")
            self.first_count = len(records)
        if (len(records) != self.first_count+iteration or
                (self.expected_prefix is not None and self.expected_prefix != prefix) or
                any(r.observation.contract_id != self.problem.contract_id or
                    r.observation.status in ("unknown", "infrastructure_error") or
                    r.diagnostics.get("numerical_failure")
                    for r in records)):
            raise ValueError("Untrusted or foreign CBOB training history")
        try:
            legal, _, rejected = filter_near(self.problem, global_pool, records)
            if not legal:
                raise RuntimeError("No legal unobserved candidates")
            for x in legal:
                if (self.problem.canonical(x) != x or any(
                        not math.isfinite(float(v)) or v > 0
                        for v in self.problem.precheck(x).values())):
                    raise ValueError("Illegal mixed-hierarchy BO candidate")
            onehot = PrefixFeatures(
                self.problem, self.base, records, self.seed, False, 30)
            posterior = HeterogeneousRiskPosterior(
                self.problem, records, self.seed, self.settings,
                onehot, onehot, view_policy="CBOB_adapted_explicit_heads")
            forecasts = posterior.predict(legal)
            scores, diag = candidate_eicb(forecasts, posterior.best)
            best_index = int(np.argmax(scores))
            candidate = deepcopy(legal[best_index])
            self.pending = (candidate, prefix, [r.to_dict() for r in records])
            return candidate, {
                **diag,
                "method_provenance": "independent adapted CBOB formula, NOT official code",
                "GP_likelihood": "RA_LSBO_stable_exact_plus_interval_EP",
                "candidate_generator": "same_legal_global_pool_not_Sobol",
                "source_constraints": list(forecasts["constraints"]),
                "candidate_index": best_index,
                "scores": scores.tolist(),
                "pool": legal,
                "rejected_near": rejected,
                "history_digest": prefix,
                "posterior": posterior.diagnostics(),
                "official_implementation": False,
                "mathematics_qualified": False,
            }
        except Exception:
            self.failed = True
            raise

    def accept(self, record):
        if self.failed or self.pending is None:
            raise RuntimeError("No CBOB proposal pending")
        candidate, _, history = self.pending
        if (record.candidate != candidate or
                record.observation.contract_id != self.problem.contract_id or
                record.observation.status in ("unknown","infrastructure_error") or
                record.diagnostics.get("numerical_failure")):
            self.failed = True
            raise ValueError("Foreign or untrusted CBOB feedback")
        self.expected_prefix = digest(history+[record.to_dict()])
        self.next_iteration += 1
        self.pending = None
