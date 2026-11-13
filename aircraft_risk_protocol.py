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
PROFILES = {
    "check": (1, 4, 6, 16, 12, 12, 28410000, (4, 6)),
    "smoke": (1, 4, 6, 16, 12, 12, 28411000, (4, 6)),
    "pilot": (6, 12, 48, 128, 96, 128, 28412000, (12, 24, 36, 48)),
}
SCHEMA = "ra_lsbo_aircraft_risk_cold_start_v1"


def plan(profile):
    repeats, initial, total, pool, reference, heldout, seed, checkpoints = PROFILES[profile]
    return dict(schema=SCHEMA, profile=profile, case=CASE, step_seconds=20,
                arms=list(ARMS), seeds=[seed + 100*i for i in range(repeats)],
                initial=initial, total=total, pool=pool, reference=reference,
                heldout=heldout, checkpoints=list(checkpoints),
                budget=dict(initial=repeats*initial,
                            online=repeats*len(ARMS)*(total-initial),
                            heldout=repeats*heldout),
                rcaide_mda_calls=0)
