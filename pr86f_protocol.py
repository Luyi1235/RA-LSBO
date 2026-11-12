"""Prospective rank-versus-magnitude ablation; no historical outcomes are inputs."""
BASE_SHA = "2091e0e4c9d3d86a88c55715d2a23a2cd0f36db4"
METHODS = ("constrained_ei", "objective_weighted_boundary", "direct_weighted_boundary")
TASKS = ("legacy_branin_sparse", "legacy_rosenbrock_sparse") + tuple("tradeoff_s%d" % i for i in range(8))


def plan(profile):
    if profile not in ("ci", "full"):
        raise ValueError("Only ci/full profiles are registered")
    full = profile == "full"
    tasks, repeats = (TASKS, 3) if full else (TASKS[:1], 1)
    block, total, pool, tests = (1210000, 48, 192, 512) if full else (1220000, 28, 32, 32)
    units = []
    for task in tasks:
        for repeat in range(repeats):
            seed = block + 1000*TASKS.index(task) + 32*repeat
            units.append(dict(task=task, repeat=repeat, anchor_seed=seed+11,
                              pool_seed=seed+23, test_seed=seed+37))
    count, online = len(units), total-24
    fits = count*len(METHODS)*(online+1)
    return dict(schema="pr86f_rank_magnitude_v1", profile=profile,
        units=units, methods=list(METHODS), anchors=24, total_per_arm=total,
        online_per_arm=online, candidate_pool=pool, test_count=tests,
        checkpoints=[24,36,48] if full else [24,28],
        weights=[0.65,0.35], entropy_units="nats; not normalized by log(2)",
        minimum_exact_objectives=2, calibration="none; raw risk probabilities",
        shared_anchor_calls=count*24, online_calls=count*len(METHODS)*online,
        test_calls=count*tests, math_calls=count*(24+len(METHODS)*online+tests),
        trajectories=count*len(METHODS), risk_refits=fits, risk_heads=2*fits,
        risk_kernels=16*fits, objective_fits="data-dependent, separately billed",
        primary_pair=["direct_weighted_boundary","objective_weighted_boundary"],
        strong_baseline="constrained_ei", known_task_family=True,
        aircraft_calls=0, vae_fits=0, completion_risk_models=0,
        cold_rule="EI=1 if objective GP or exact feasible incumbent unavailable",
        objective_contract="exact iff region<=0; missing otherwise; incumbent also requires resource<=0",
        interpretation="Prospective new-seed within-known-task-family ablation; not exact VOI or safety evidence")
