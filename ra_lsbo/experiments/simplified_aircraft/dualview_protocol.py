"""Prospective RA-LSBO risk-decoupled dual-view aircraft experiment.

A new, independent five-arm study; it never resumes or overwrites the
interrupted corpus pilot. Pilot results are exploratory, not confirmation.
"""
from copy import deepcopy
from dataclasses import asdict, replace

from ..core.methods import settings_for
from ..discovery_stages.policies import COVERAGE_PERIOD, NEAR_TOLERANCE

SCHEMA="ra_lsbo_aircraft_risk_dualview_v1"
DOCUMENT="docs/AIRCRAFT_DUALVIEW_EXPERIMENT_PROTOCOL_20261101.md"
CASES=("aircraft_ordinary","aircraft_sparse")
ARMS=("random","onehot_stable","corpus_stable","hybrid_onehot","hybrid_guard")
CORPUS_ARMS=("corpus_stable","hybrid_onehot","hybrid_guard")
REFERENCE="hybrid_onehot"
PROFILES={
    "smoke":{"name":"smoke","seeds":[9960],"initial":6,"horizon":2,
             "pool":32,"encoder_epochs":30,"native_gp_steps":2},
    "pilot":{"name":"pilot","seeds":[9961,9962,9963,9964],"initial":12,
             "horizon":18,"pool":128,"encoder_epochs":30,"native_gp_steps":15},
}


def profile_for(name):
    if name not in PROFILES:
        raise ValueError("No unspecified seeds, horizon or confirmation profile")
    return deepcopy(PROFILES[name])


def budget_for(profile):
    trials=len(CASES)*len(profile["seeds"])*len(ARMS)
    roles={
        "shared_initial":len(CASES)*len(profile["seeds"])*profile["initial"],
        "online":trials*profile["horizon"],
        "incumbent_step_verification":trials,
    }
    return {**roles,"maximum_model_calls":sum(roles.values()),
            "maximum_rcaide_mda_calls":0}


def model_settings(method,profile):
    if method not in ARMS or method=="random":
        raise ValueError("No GP settings for uniform random control")
    # The existing generic RA smoke reduces its GP grid to 2x1. This new
    # protocol intentionally holds the full 4x2 kernel grid even in smoke,
    # so numerical behavior is the same as in the prospective pilot.
    return replace(settings_for("ra_lsbo",profile),
        latent=method in CORPUS_ARMS,encoder_epochs=30,
        continuous_grid=(.05,.15,.4,1.),discrete_grid=(.5,1.5))


def method_spec(method,profile):
    if method not in ARMS:
        raise ValueError("Unregistered method")
    if method=="random":
        return {"engine":"unchanged_external_v2",
                "description":"Uniform random exact-screened pool", "adaptation":None}
    allocation={
        "onehot_stable":"onehot_objective_and_risk",
        "corpus_stable":"latent_all",
        "hybrid_onehot":"onehot_risk",
        "hybrid_guard":"soc_dual_guard",
    }[method]
    return {
        "engine":"risk_decoupled_dualview_v1",
        "settings":asdict(model_settings(method,profile)),
        "representation_allocation":allocation,
        "cold_global":True,"protect":True,
        "feedback":"interval","coverage_scope":"cold",
        "near_tolerance":NEAR_TOLERANCE,"coverage_period":COVERAGE_PERIOD,
        "gp_grid_continuous":[.05,.15,.4,1.],
        "gp_grid_discrete":[.5,1.5],
        "gp_stable_tail_moments":True,
        "gp_max_sweeps":60,"gp_damping":.7,"gp_tolerance":1e-8,
        "failure_policy":"on_no_valid_response_head_deterministic_coverage_no_prior_invention",
        "response_policy":{
            "completion":"onehot" if method!="corpus_stable" else "corpus_latent",
            "objective":"corpus_latent" if method in CORPUS_ARMS else "onehot",
            "constraints":"onehot" if method not in ("corpus_stable",) else "corpus_latent",
            "soc_min_feasibility":"both_latent_and_onehot" if method=="hybrid_guard" else None,
        },
        "corpus_training": {
            "draws":2000,"initial_epochs":30,"update_epochs":20,
            "fit_before_online_indices":[0,8,16],
            "keep_duplicate_uniform_canonical_draws":True,
            "no_new_truth_to_train_unless_previously_evaluated":True,
        } if method in CORPUS_ARMS else None,
    }


def method_order(profile,case,seed):
    i=profile["seeds"].index(seed)
    offset=(i+CASES.index(case))%len(ARMS)
    return ARMS[offset:]+ARMS[:offset]
