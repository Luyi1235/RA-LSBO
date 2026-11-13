"""Frozen reduced-aircraft cold-start pilot roster and evaluation budgets."""

CASE = "aircraft_sparse"
ARMS = {
    "iid_explicit": ("iid", False),
    "coverage_explicit": ("coverage", False),
    "boundary_explicit": ("boundary", False),
    "brier_explicit": ("brier", False),
    "boundary_latent": ("boundary", True),
    "brier_latent": ("brier", True),
}
PHASE_ARMS = {
    "coverage_brier": ("brier", False),
    "coverage_boundary": ("boundary", False),
}
OPT_ARMS = {
    "ei_explicit": ("ei", False),
    "cei_explicit": ("cei", False),
    "ei_latent": ("ei", True),
    "cei_latent": ("cei", True),
}
LOCAL_ARMS = {"cei_latent_local": ("cei", True)}
LOCAL_SEARCH = {"local-check": (10, 8), "local-smoke": (10, 8), "local-pilot": (36, 24)}
OBJECTIVE_UPDATE_EVERY = 8
PHASE_SWITCH = {"phase-check": 5, "phase-smoke": 5, "phase-pilot": 24}
PROFILES = {
    "local-check": (1, 8, 12, 24, 0, 12, 28420200, (8, 10, 12)),
    "local-smoke": (1, 8, 12, 24, 0, 12, 28420100, (8, 10, 12)),
    "local-pilot": (12, 12, 48, 128, 0, 32, 28421000, (12, 24, 36, 48)),
    "energy-check": (1, 8, 17, 24, 0, 12, 28416000, (8, 16, 17)),
    "energy-smoke": (1, 8, 11, 24, 0, 12, 28416100, (8, 11)),
    "energy-pilot": (12, 12, 48, 128, 0, 64, 28417000, (12, 24, 36, 48)),
    "phase-check": (1, 4, 7, 16, 12, 12, 28413000, (4, 5, 7)),
    "phase-smoke": (1, 4, 7, 16, 12, 12, 28413100, (4, 5, 7)),
    "phase-pilot": (12, 12, 48, 128, 96, 128, 28414000, (12, 24, 36, 48)),
    "check": (1, 4, 6, 16, 12, 12, 28410000, (4, 6)),
    "smoke": (1, 4, 6, 16, 12, 12, 28411000, (4, 6)),
    "pilot": (6, 12, 48, 128, 96, 128, 28412000, (12, 24, 36, 48)),
}
SCHEMA = "ra_lsbo_aircraft_risk_cold_start_v1"


def plan(profile):
    repeats, initial, total, pool, reference, heldout, seed, checkpoints = PROFILES[profile]
    switch = PHASE_SWITCH.get(profile)
    arms = (list(ARMS) if switch is None else
            [name for name, (_, latent) in ARMS.items() if not latent] + list(PHASE_ARMS))
    if profile.startswith("energy-"):
        arms = list(OPT_ARMS)
    shared = 0 if switch is None else len(PHASE_ARMS)*(switch-initial)
    if profile in LOCAL_SEARCH:
        switch, _ = LOCAL_SEARCH[profile]
        arms = ["cei_latent", *LOCAL_ARMS]
        shared = switch-initial
    spec = dict(schema=SCHEMA, profile=profile, case=CASE, step_seconds=20,
                arms=arms, seeds=[seed + 100*i for i in range(repeats)],
                initial=initial, total=total, pool=pool, reference=reference,
                heldout=heldout, checkpoints=list(checkpoints),
                budget=dict(initial=repeats*initial,
                            online=repeats*(len(arms)*(total-initial)-shared),
                            heldout=repeats*heldout),
                rcaide_mda_calls=0)
    if profile in PHASE_SWITCH:
        spec["coverage_switch"] = switch
        spec["shared_coverage_arm"] = "coverage_explicit"
    if profile in LOCAL_SEARCH:
        spec["local_switch"], spec["augmentation"] = LOCAL_SEARCH[profile]
        spec["shared_search_arm"] = "cei_latent"
    if profile.startswith("energy-") or profile in LOCAL_SEARCH:
        spec["objective_update_every"] = OBJECTIVE_UPDATE_EVERY
    return spec
