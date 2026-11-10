"""PR86-B frozen sequential risk-information acquisition protocol."""
from risk_learning.core import TASKS

METHODS = ("iid_random", "maximin", "latent_uncertainty", "boundary", "calibration_aware_voi")
ANCHORS = 12
PROFILES = {
    "ci": dict(tasks=(TASKS[0], TASKS[2]), repeats=1, total=24, test=64, pool=96, block=1130000, checkpoints=(12, 24)),
    "smoke": dict(tasks=TASKS, repeats=1, total=48, test=128, pool=96, block=1140000, checkpoints=(12, 24, 48)),
    "full": dict(tasks=TASKS, repeats=3, total=48, test=512, pool=192, block=1150000, checkpoints=(12, 24, 48)),
}

def plan(profile):
    if profile not in PROFILES:
        raise ValueError("Unknown PR86-B profile")
    cfg = PROFILES[profile]
    units = []
    for task in cfg["tasks"]:
        for repeat in range(cfg["repeats"]):
            seed = cfg["block"] + 1000 * TASKS.index(task) + 32 * repeat
            units.append(dict(task=task, repeat=repeat, anchor_seed=seed + 11,
                              pool_seed=seed + 23, test_seed=seed + 37, calibration_seed=seed + 41))
    n = len(units)
    online = cfg["total"] - ANCHORS
    calls_per_unit = ANCHORS + len(METHODS) * online + cfg["test"]
    fits_per_unit = len(METHODS) * (online + 1)
    return dict(
        schema="pr86b_sequential_risk_information_v1", profile=profile,
        units=units, tasks=list(cfg["tasks"]), methods=list(METHODS), anchor_count=ANCHORS,
        total_observations_per_method=cfg["total"], online_per_method=online,
        candidate_pool=cfg["pool"], calibration_anchor_count=(cfg["total"]-ANCHORS)//4, test_count=cfg["test"],
        checkpoints=list(cfg["checkpoints"]), calls_per_unit=calls_per_unit,
        total_math_calls=n*calls_per_unit, shared_anchor_calls=n*ANCHORS,
        arm_online_calls=n*len(METHODS)*online, heldout_test_calls=n*cfg["test"],
        expected_model_refits=n*fits_per_unit, expected_gp_heads=n*fits_per_unit*2,
        expected_gp_kernel_constructions=n*fits_per_unit*16,
        primary_method="calibration_aware_voi",
        baseline="Laplace-smoothed joint feasibility prevalence from the 12 common random anchors",
        test_labels_fit_isolated=True, candidate_pool_labels_never_materialized=True,
        sequential_refit_after_each_real_feedback=True,
        aircraft_calls=0, rcaide_calls=0, vae_fits=0, fantasy_fits=0,
        limits=[
            "This phase evaluates risk prediction and feedback acquisition, not objective optimization.",
            "The joint feasibility probability is the product of two marginal heads; it is an independence approximation.",
            "BALD uses a fixed probit observation approximation and is not exact expected Brier reduction.",
            "Calibration-aware VOI reserves random anchor evaluations and uses an anchor-only intercept correction; no test labels are used.",
            "All arms share 12 initial anchors; online outcomes are arm-specific and separately charged.",
            "No unevaluated candidate truth may be generated, cached, or exposed to acquisition code.",
            "The 96-observation PR86-A curve is a historical task-specific reference, not a universal target."
        ])

if __name__ == "__main__":
    import json, sys
    print(json.dumps(plan(sys.argv[1] if len(sys.argv)>1 else "full"), indent=2))
