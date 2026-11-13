"""Generate equal-sized global and local augmentations from a common CEI prefix."""
from random import Random

from ra_lsbo.budget import digest

RADII = (.025, .06, .12)


def local_anchor(history, checked_label):
    """Return the best observed usable design with a deterministic tie-break."""
    usable = [row for row in history if checked_label(row) is True
              and row["observation"]["status"] == "completed"
              and row["observation"]["objective"]["kind"] == "exact"]
    best = min(usable, key=lambda row: (row["observation"]["objective"]["value"],
                                      row["observation"]["design_id"]), default=None)
    return best["candidate"] if best else None


def local_candidates(problem, anchor, seed, count, excluded):
    """Sample active continuous coordinates near the feasible incumbent."""
    rng = Random(seed)
    space = problem.space
    discrete = anchor["discrete"]
    center = space.unit(problem.candidate_class(**anchor))
    active = space.mask(discrete)
    candidates = []
    seen = set(excluded)
    for _ in range(count*1000):
        radius = RADII[len(candidates) % len(RADII)]
        unit = [min(1., max(0., u + rng.uniform(-radius, radius))) if use else u
                for u, use in zip(center, active)]
        candidate = space.from_unit(unit, discrete).to_dict()
        identity = digest(candidate)
        if identity in seen or any(value > 0 for value in problem.precheck(candidate).values()):
            continue
        candidates.append(candidate)
        seen.add(identity)
        if len(candidates) == count:
            return candidates
    raise RuntimeError("Could not generate the registered local candidate count")


def check_augmentation(row, n, count, expected_mode, anchor, existing):
    if (set(row) != {"event", "n", "mode", "anchor", "candidates"}
            or row["event"] != "augmentation" or row["n"] != n
            or row["mode"] != expected_mode or row["anchor"] != (digest(anchor) if anchor else None)
            or len(row["candidates"]) != count):
        raise ValueError("Candidate augmentation does not match the registered switch")
    ids = [digest(candidate) for candidate in row["candidates"]]
    if len(set(ids)) != count or set(ids) & existing:
        raise ValueError("Candidate augmentation contains duplicate or held-out designs")
    if anchor and any(candidate["discrete"] != anchor["discrete"] for candidate in row["candidates"]):
        raise ValueError("Local candidate changed the incumbent configuration")
