"""New independent 6-arm H2-M prospective research experiment registration.

Not a resumption of either 2026-10-31 interrupted pilot or complete H1.
The H2-B hard-head controller remains a separate synthetic-only ablation,
not silently substituted for H2-M in any flight experiment.
"""
from copy import deepcopy
from dataclasses import asdict

from . import dualview_protocol as h1
from .h2_risk_controller import H2Settings

SCHEMA="ra_lsbo_aircraft_h2_causal_pilot_v1"
DOCUMENT="docs/RA_LSBO_H2_PROSPECTIVE_AIRCRAFT_PROTOCOL_20261102.md"
CASES=("aircraft_ordinary","aircraft_sparse")
ARMS=("random","onehot_stable","corpus_stable",
      "hybrid_onehot","hybrid_guard","h2_causal")
CORPUS_ARMS=("corpus_stable","hybrid_onehot","hybrid_guard","h2_causal")
REFERENCE="hybrid_onehot"
PROFILES={
    "smoke":{"name":"smoke","seeds":[11061],"initial":6,"horizon":2,
             "pool":32,"encoder_epochs":30,"native_gp_steps":2},
    "pilot":{"name":"pilot","seeds":[11062,11063,11064,11065],
             "initial":12,"horizon":18,
             "pool":128,"encoder_epochs":30,"native_gp_steps":15},
}


def profile_for(name):
    if name not in PROFILES:
        raise ValueError("Use only independently frozen H2 smoke or development pilot profile")
    return deepcopy(PROFILES[name])


def budget_for(profile):
    n_trials=len(CASES)*len(profile["seeds"])*len(ARMS)
    roles={
        "shared_initial":len(CASES)*len(profile["seeds"])*profile["initial"],
        "online":n_trials*profile["horizon"],
        "incumbent_step_verification":n_trials,
    }
    return {**roles,"maximum_model_calls":sum(roles.values()),
            "maximum_rcaide_mda_calls":0}


def model_settings(method,profile):
    if method not in ARMS or method=="random":
        raise ValueError("H2 method has no registered GP settings")
    reference="hybrid_guard" if method=="h2_causal" else method
    return h1.model_settings(reference,profile)


def method_spec(method,profile):
    if method not in ARMS:
        raise ValueError("Unregistered H2 comparison arm")
    if method=="h2_causal":
        result=h1.method_spec("hybrid_guard",profile)
        result["engine"]="ra_lsbo_h2_causal_risk_v1"
        result["representation_allocation"]="past_score_Gaussian_SOC_mixture"
        result["response_policy"]["soc_min_feasibility"]=None
        result["response_policy"]["soc_mixture"]="fixed_two_view_prior_components"
        result["controller"]={
            **asdict(H2Settings()),
            "update_after_settled_only":True,
            "exact_SOC_score":"Gaussian_density_NLL",
            "interval_SOC_score":"interval_probability_mass_NLL",
            "paired_delta_cap":True,
            "unknown_or_unfitted_no_update":True,
        }
        result["prequential_scoring"]="exact_two_component_mixture_log_likelihood"
        return result
    return h1.method_spec(method,profile)


def method_order(profile,case,seed):
    index=profile["seeds"].index(seed)
    start=(index+CASES.index(case))%len(ARMS)
    return ARMS[start:]+ARMS[:start]
