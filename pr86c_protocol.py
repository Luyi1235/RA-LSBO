"""PR86-C frozen prospective objective-aware, sequential mathematical protocol."""
from risk_learning.core import TASKS as RELEASED

HELDOUT = tuple("tradeoff_s%d" % i for i in range(4, 8))
METHODS = ("iid_random", "boundary", "marginal_entropy",
           "constrained_ei", "objective_weighted_boundary")
ANCHORS = 12
PROFILES = {
    "ci": dict(tasks=(RELEASED[0], RELEASED[2]), repeats=1, total=24,
               test=64, pool=96, seed=1160000, checkpoints=(12, 24)),
    "smoke": dict(tasks=RELEASED, repeats=1, total=48,
                  test=128, pool=96, seed=1170000, checkpoints=(12, 24, 48)),
    "full": dict(tasks=RELEASED + HELDOUT, repeats=3, total=48,
                 test=512, pool=192, seed=1180000, checkpoints=(12, 24, 48)),
}


def plan(profile):
    if profile not in PROFILES:
        raise ValueError("Unknown PR86-C profile")
    cfg = PROFILES[profile]
    tasks = RELEASED + HELDOUT
    units = []
    for task in cfg["tasks"]:
        for repeat in range(cfg["repeats"]):
            seed = cfg["seed"] + 1000 * tasks.index(task) + 32 * repeat
            units.append(dict(task=task, repeat=repeat, anchor_seed=seed+11,
                              pool_seed=seed+23, test_seed=seed+37))
    online = cfg["total"] - ANCHORS
    n = len(units)
    return dict(
        schema="pr86c_objective_aware_prospective_v1", profile=profile, units=units,
        methods=list(METHODS), anchor_count=ANCHORS,
        online_per_method=online, total_observations_per_method=cfg["total"],
        candidate_pool=cfg["pool"], test_count=cfg["test"], checkpoints=list(cfg["checkpoints"]),
        shared_anchor_calls=n*ANCHORS, arm_online_calls=n*len(METHODS)*online,
        heldout_test_calls=n*cfg["test"],
        total_math_calls=n*(ANCHORS+len(METHODS)*online+cfg["test"]),
        expected_risk_refits=n*len(METHODS)*(online+1),
        expected_risk_heads=n*len(METHODS)*(online+1)*2,
        expected_risk_kernel_constructions=n*len(METHODS)*(online+1)*16,
        objective_head_attempts_data_dependent=True,
        objective_weight=0.65, information_weight=0.35,
        objective_direction="minimize",
        objective_observation_rule="Exact only if observed region<=0; otherwise missing. Never impute.",
        feasibility_rule="Exact feasible iff region and resource are known <=0 and objective is exact.",
        primary_endpoints=[
            "best_observed_exact_feasible_objective_or_null_at_n12_n24_n48",
            "exact_feasible_discovery_at_n12_n24_n48",
            "paired_lexicographic_win_loss_tie_at_n48_no_imputation",
        ],
        acquisition_rule={
            "iid_random":"uniform without replacement",
            "boundary":"p_joint*(1-p_joint)",
            "marginal_entropy":"H(p_region)+H(p_resource), no reserved calibration",
            "constrained_ei":"p_joint*EI_min; if no exact feasible incumbent use p_joint",
            "objective_weighted_boundary":
                "0.65 rank(p_joint*EI_min)+0.35 rank(EI_min*H(p_joint)); "
                "if no exact feasible incumbent use EI_min=1 for both terms",
        },
        calibration_policy="No on-policy calibration for any arm. Raw GP probabilities drive every decision.",
        interpretation_limits=[
            "These are analytic mixed-design tasks; no VAE, aircraft, RCAIDE or flight controls.",
            "Current two GP risk heads concern constraint violation, not simulator completion.",
            "The objective GP only fits exact observed objective labels; missing never becomes zero.",
            "Risk-only entropy is NOT a global VOI or expected Brier reduction.",
            "Joint feasibility uses marginal independence approximation, not a safety guarantee.",
            "All arms share exactly 12 random anchors and pay for 36 individually selected online outcomes.",
            "Independent heldout truth is never passed to candidate selection or model fitting.",
            "Previously withheld tradeoff_s4..s7 are first released by prospective FULL, not CI/smoke.",
            "All candidate scores and weights frozen before FULL; no seed or metric tuning.",
        ],
        test_labels_fit_isolated=True, all_methods_same_online_budget=True,
        aircraft_calls=0, rcaide_calls=0, vae_fits=0, fantasy_fits=0,
    )
