"""Generate a frozen, screened LHS roster in the native X-57 design space."""

from collections import Counter
from itertools import chain

import numpy as np

from .risk_core.observations import finite

LAYOUT = "prop_layout"
LAYOUTS = ("wingtip", "leading_edge_distributed")
PER_LAYOUT = 16


def violations(checks):
    return [name for name, value in checks.items() if not finite(value) or value > 0]


def active_names(space, candidate):
    return [v.name for v in space.variables if space.active(v, candidate.discrete)]


def coverage(space, points):
    discrete = {
        v.name: dict(Counter(str(p.discrete[v.name]) for p in points if space.active(v, p.discrete)))
        for v in space.discrete
    }
    units = [space.unit(p) for p in points]
    bins = {}
    for column, v in enumerate(space.continuous):
        values = [u[column] for p, u in zip(points, units) if space.active(v, p.discrete)]
        n = len(values)
        bins[v.name] = {"active_count": n, "bins": sorted(min(n - 1, int(n * x)) for x in values)}
    return {"count": len(points), "discrete": discrete, "continuous": bins}


class RosterShortfall(RuntimeError):
    def __init__(self, audit):
        super().__init__("Cannot fill a layout's screened LHS quota")
        self.audit = audit


def _slots(space, layout, rng):
    default = space.default().discrete
    rows = [dict(default, **{LAYOUT: layout}) for _ in range(PER_LAYOUT)]
    for variable in space.discrete:
        if variable.name == LAYOUT:
            continue
        active = [d for d in rows if space.active(variable, d)]
        levels = rng.permutation(len(variable.values))
        assignments = np.resize(levels, len(active))
        rng.shuffle(assignments)
        for d, index in zip(active, assignments):
            d[variable.name] = variable.values[index]
    lows = np.zeros((PER_LAYOUT, len(space.continuous)))
    widths = np.ones_like(lows)
    for column, variable in enumerate(space.continuous):
        active = [i for i, d in enumerate(rows) if space.active(variable, d)]
        if active:
            lows[active, column] = rng.permutation(len(active)) / len(active)
            widths[active, column] = 1 / len(active)
    return rows, lows, widths


def _select(space, pool):
    remaining = list(pool)
    selected, counts, bins = [], Counter(), set()
    conditional_sizes = {
        v.name: max(1, round(PER_LAYOUT * sum(space.active(v, p.discrete) for p in pool) / len(pool)))
        for v in space.continuous
    }
    features = {}
    for point in pool:
        levels = [(v.name, str(point.discrete[v.name])) for v in space.discrete
                  if v.name != LAYOUT and space.active(v, point.discrete)]
        cells = [(v.name, min(conditional_sizes[v.name] - 1, int(conditional_sizes[v.name] * u)))
                 for v, u in zip(space.continuous, space.unit(point)) if space.active(v, point.discrete)]
        features[id(point)] = levels, cells
    while remaining and len(selected) < PER_LAYOUT:
        def score(point):
            levels, cells = features[id(point)]
            return (sum(counts[key] == 0 for key in levels),
                    sum(cell not in bins for cell in cells) - sum(counts[key] for key in levels) / 4)

        index = max(range(len(remaining)), key=lambda i: score(remaining[i]))
        point = remaining.pop(index)
        levels, cells = features[id(point)]
        counts.update(levels)
        bins.update(cells)
        selected.append(point)
    return selected


def freeze_roster(problem, seed, *, max_designs=64):
    space = problem.space
    rng = np.random.default_rng(seed)
    by_layout, audit = {}, {}
    for layout in LAYOUTS:
        reasons, attempted_levels, eligible_levels = Counter(), Counter(), Counter()
        supply, keys = [], set()
        for design in range(1, max_designs + 1):
            rows, lows, widths = _slots(space, layout, rng)
            drafted = [space.from_unit(u, d) for u, d in zip(
                lows + widths * rng.random(lows.shape), rows)]
            if design == 1:
                before = coverage(space, drafted)
            for point in drafted:
                levels = [(v.name, str(point.discrete[v.name])) for v in space.discrete
                          if space.active(v, point.discrete)]
                attempted_levels.update(levels)
                key = space.key(point)
                rejected = ["duplicate"] if key in keys else violations(problem.precheck(point))
                if rejected:
                    reasons.update(rejected)
                else:
                    supply.append(point)
                    keys.add(key)
                    eligible_levels.update(levels)
            if len(supply) >= 4 * PER_LAYOUT and eligible_levels.keys() >= attempted_levels.keys():
                break
        best = _select(space, supply) if supply else []
        audit[layout] = {
            "designs_tried": design, "before": before, "after": coverage(space, best),
            "rejections_by_reason": dict(reasons),
            "attempted_count": design * PER_LAYOUT, "screened_supply": len(supply),
            "attempted_levels": {v.name: {str(level): attempted_levels[v.name, str(level)]
                                         for level in v.values} for v in space.discrete},
            "eligible_levels": {v.name: {str(level): eligible_levels[v.name, str(level)]
                                         for level in v.values} for v in space.discrete},
        }
        by_layout[layout] = best
    if any(len(points) != PER_LAYOUT for points in by_layout.values()):
        raise RosterShortfall(audit)
    return list(chain.from_iterable(zip(*(by_layout[layout] for layout in LAYOUTS)))), audit
