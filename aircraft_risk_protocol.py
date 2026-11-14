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
LOCAL_FACTOR_ARMS = {
    "ei_latent_global": ("ei", True),
    "ei_latent_local": ("ei", True),
}
LOCAL_FACTOR_SCENARIOS = {
    "local-factor-check": 4.,
    "local-factor-smoke": 4.,
    "local-factor-nominal": 0.,
    "local-factor-headwind": 4.,
}
LOCAL_SEARCH.update({
    "local-factor-check": (10, 8),
    "local-factor-smoke": (10, 8),
    "local-factor-nominal": (36, 24),
    "local-factor-headwind": (36, 24),
})
DISCOVERY_ARMS = {
    "discovery_coverage": ("coverage", True),
    "discovery_constraints": ("constraints", True),
    "discovery_direct": ("direct", True),
    "discovery_staged": ("staged", True),
    "discovery_collapsed": ("collapsed_lookahead", True),
    "discovery_lookahead": ("staged_lookahead", True),
}
DISCOVERY_SCENARIOS = {
    "discovery-smoke": 4.,
    "discovery-nominal": 0.,
    "discovery-headwind": 4.,
}
FEEDBACK_ARMS = {
    "feedback_coverage": ("coverage", True),
    "feedback_exact": ("constraints", True),
    "feedback_sign": ("constraints", True),
    "feedback_bounds": ("constraints", True),
}
FEEDBACK_MODES = {
    "feedback_coverage": "bounds", "feedback_exact": "exact",
    "feedback_sign": "sign", "feedback_bounds": "bounds",
}
FEEDBACK_SCENARIOS = {"feedback-smoke": 4., "feedback-nominal": 0., "feedback-headwind": 4.}
FEEDBACK_PAIRS = (("feedback_bounds", "feedback_sign"), ("feedback_sign", "feedback_exact"),
                  ("feedback_bounds", "feedback_exact"), ("feedback_bounds", "feedback_coverage"))
SIGN_LSBO_ARMS = {
    "lsbo_coverage_explicit": ("coverage", False),
    "lsbo_coverage_latent": ("coverage", True),
    "lsbo_sign_explicit": ("constraints", False),
    "lsbo_sign_latent": ("constraints", True),
    "lsbo_exact_latent": ("constraints", True),
}
SIGN_LSBO_FEEDBACK = {
    "lsbo_coverage_explicit": "bounds", "lsbo_coverage_latent": "bounds",
    "lsbo_sign_explicit": "sign", "lsbo_sign_latent": "sign", "lsbo_exact_latent": "exact",
}
SIGN_LSBO_SCENARIOS = {
    "sign-lsbo-smoke": (4., 0.),
    "sign-lsbo-headwind": (4., 0.),
    "sign-lsbo-payload": (0., 24.),
}
SIGN_LSBO_PAIRS = (
    ("lsbo_sign_latent", "lsbo_coverage_latent"),
    ("lsbo_sign_latent", "lsbo_exact_latent"),
    ("lsbo_sign_latent", "lsbo_sign_explicit"),
    ("lsbo_sign_explicit", "lsbo_coverage_explicit"),
    ("lsbo_coverage_latent", "lsbo_coverage_explicit"),
)
OBJECTIVE_UPDATE_EVERY = 8
PHASE_SWITCH = {"phase-check": 5, "phase-smoke": 5, "phase-pilot": 24}
PROFILES = {
    "sign-lsbo-smoke": (1, 4, 8, 48, 0, 12, 28429000, (4, 8)),
    "sign-lsbo-headwind": (16, 4, 36, 256, 0, 32, 28437000, (4, 8, 12, 20, 28, 36)),
    "sign-lsbo-payload": (16, 4, 36, 256, 0, 32, 28437000, (4, 8, 12, 20, 28, 36)),
    "feedback-smoke": (1, 4, 7, 32, 0, 16, 28429000, (4, 7)),
    "feedback-nominal": (16, 4, 20, 128, 0, 32, 28433000, (4, 8, 12, 16, 20)),
    "feedback-headwind": (16, 4, 20, 128, 0, 32, 28433000, (4, 8, 12, 16, 20)),
    "discovery-smoke": (1, 4, 7, 32, 0, 16, 28429000, (4, 7)),
    "discovery-nominal": (16, 4, 20, 128, 0, 32, 28430000, (4, 8, 12, 16, 20)),
    "discovery-headwind": (16, 4, 20, 128, 0, 32, 28430000, (4, 8, 12, 16, 20)),
    "local-factor-check": (1, 8, 17, 24, 0, 12, 28424610, (8, 10, 16, 17)),
    "local-factor-smoke": (1, 8, 12, 24, 0, 12, 28424100, (8, 10, 12)),
    "local-factor-nominal": (12, 12, 48, 128, 0, 32, 28425000, (12, 24, 36, 48)),
    "local-factor-headwind": (12, 12, 48, 128, 0, 32, 28427000, (12, 24, 36, 48)),
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
    if profile in DISCOVERY_SCENARIOS:
        arms = list(DISCOVERY_ARMS)
    if profile in FEEDBACK_SCENARIOS:
        arms = list(FEEDBACK_ARMS)
    if profile in SIGN_LSBO_SCENARIOS:
        arms = list(SIGN_LSBO_ARMS)
    shared = 0 if switch is None else len(PHASE_ARMS)*(switch-initial)
    if profile in LOCAL_SEARCH:
        switch, _ = LOCAL_SEARCH[profile]
        arms = ["cei_latent", *LOCAL_ARMS]
        if profile in LOCAL_FACTOR_SCENARIOS:
            arms.extend(LOCAL_FACTOR_ARMS)
        shared = (len(arms)-1)*(switch-initial)
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
    if profile in LOCAL_FACTOR_SCENARIOS:
        spec["headwind_m_s"] = LOCAL_FACTOR_SCENARIOS[profile]
        spec["local_factorial"] = True
    if profile in DISCOVERY_SCENARIOS:
        spec.update(headwind_m_s=DISCOVERY_SCENARIOS[profile], staged_discovery=True)
    if profile in FEEDBACK_SCENARIOS:
        spec.update(headwind_m_s=FEEDBACK_SCENARIOS[profile], constraint_feedback=dict(FEEDBACK_MODES))
    if profile in SIGN_LSBO_SCENARIOS:
        headwind, payload = SIGN_LSBO_SCENARIOS[profile]
        spec.update(headwind_m_s=headwind, payload_delta_kg=payload,
                    constraint_feedback=dict(SIGN_LSBO_FEEDBACK), sign_lsbo=True)
    if (profile.startswith("energy-") or profile in LOCAL_SEARCH
            or profile in DISCOVERY_SCENARIOS or profile in FEEDBACK_SCENARIOS or profile in SIGN_LSBO_SCENARIOS):
        spec["objective_update_every"] = OBJECTIVE_UPDATE_EVERY
    return spec
