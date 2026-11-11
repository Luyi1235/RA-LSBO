"""PR86-D frozen nested initial-sample allocation protocol."""
from risk_learning.core import TASKS as RELEASED
from pr86c_protocol import HELDOUT, METHODS

ALL_TASKS = RELEASED + HELDOUT
ANCHOR_SIZES = (12, 24)
TOTAL_PER_ARM = 48
FULL_CONFIG = dict(tasks=ALL_TASKS, repeats=3, test=512, pool=192, seed=1190000)
CI_CONFIG = dict(tasks=(ALL_TASKS[0], ALL_TASKS[2]), repeats=1,
                 test=64, pool=96, seed=1200000)


def _cohort_budget(n0):
    return TOTAL_PER_ARM-n0


def plan(profile):
    if profile not in ("ci", "full"):
        raise ValueError("PR86-D supports only ci/full; no separate smoke profile")
    cfg = CI_CONFIG if profile == "ci" else FULL_CONFIG
    units = []
    for task in cfg["tasks"]:
        for repeat in range(cfg["repeats"]):
            seed = cfg["seed"]+1000*ALL_TASKS.index(task)+32*repeat
            units.append(dict(task=task, repeat=repeat,
                anchor_seed=seed+11, pool_seed=seed+23, test_seed=seed+37))
    n = len(units)
    online = {str(k):_cohort_budget(k) for k in ANCHOR_SIZES}
    per_unit_online = len(METHODS)*sum(online.values())
    anchors = max(ANCHOR_SIZES)
    test_calls = n*cfg["test"]
    anchor_calls = n*anchors
    online_calls = n*per_unit_online
    refits = n*len(METHODS)*sum(v+1 for v in online.values())
    return dict(
        schema="pr86d_nested_initial_sample_allocation_v1",
        profile=profile, tasks=list(cfg["tasks"]), units=units,
        methods=list(METHODS), anchor_sizes=list(ANCHOR_SIZES),
        total_observations_per_arm=TOTAL_PER_ARM,
        shared_anchor_count=anchors,
        nested_anchor_prefix=True,
        online_per_method_by_anchor_size=online,
        checkpoints_by_anchor_size={
            "12":[12,24,36,48],
            "24":[24,36,48],
        },
        candidate_pool=cfg["pool"], test_count=cfg["test"],
        objective_min_exact_labels=2,
        shared_anchor_calls=anchor_calls,
        arm_online_calls=online_calls,
        heldout_test_calls=test_calls,
        total_math_calls=anchor_calls+online_calls+test_calls,
        expected_risk_refits=refits,
        expected_risk_heads=2*refits,
        expected_risk_kernel_constructions=16*refits,
        objective_head_attempts_data_dependent=True,
        objective_observation_rule="Exact iff observed region<=0; otherwise missing; never impute.",
        feasible_objective_rule="Exact objective and both region/resource constraints <=0.",
        anchor_comparison_rule="n0=12 sees only first 12 of the shared 24 anchors; labels 13..24 are masked from that cohort.",
        candidate_rule="Same unlabeled candidate pool per task/repeat across both cohorts and all methods; charge each arm-specific selected evaluation separately.",
        test_rule="Same independent 512-point heldout set per task/repeat; scoring only, never fit/acquisition.",
        methods_description={
            "iid_random":"uniform without replacement",
            "boundary":"maximize p_joint*(1-p_joint)",
            "marginal_entropy":"maximize H(p_region)+H(p_resource)",
            "constrained_ei":"maximize p_joint*EI_min; cold fallback p_joint",
            "objective_weighted_boundary":"frozen PR86-C rank blend 0.65 rank(p_joint*EI)+0.35 rank(EI*H(p_joint)); cold fallback EI=1",
        },
        frozen_objective_gp_minimum=2,
        objective_gp_single_label_prior=False,
        aircraft_calls=0, rcaide_calls=0, vae_fits=0, fantasy_fits=0,
        interpretation_limits=[
            "This is a within-known-task-family replication with new candidate/test seed blocks, not unseen-task-family generalization.",
            "The ablation tests initial allocation at fixed per-arm total n=48; it does not test unlimited extra data.",
            "Do not change the >=2 exact-objective GP threshold in this ablation.",
            "Do not impute missing objective labels or use heldout truth in fitting/acquisition.",
            "Resource-infeasible but region-valid exact objective labels may train the objective GP; they cannot be incumbents.",
            "The two risk heads model engineering constraints, not simulator-completion events.",
            "Joint feasibility is the marginal independence approximation, not a safety guarantee.",
            "No VAE, aircraft, RCAIDE or flight-control claim.",
        ],
    )
