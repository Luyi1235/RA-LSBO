"""Fixed-configuration acquisition search with exact cheap design constraints."""

import numpy as np
import torch
from scipy.optimize import minimize


def fixed_branch_search(acquisition, bounds, seed, space, precheck, maxiter=60):
    """Never call an expensive evaluator or relax a feasibility threshold."""
    active = np.flatnonzero(space.mask(seed.discrete))
    unit = space.unit(seed)
    lower, upper = bounds.detach().cpu().numpy()
    start = np.clip(unit[active], lower, upper)
    evaluations = 0

    def candidate(x):
        value = unit.copy()
        value[active] = x
        return space.from_unit(value, seed.discrete)

    def checks(x):
        return precheck(candidate(x))

    def feasible(values):
        return all(v is not None and np.isfinite(v) and v <= 0 for v in values.values())

    initial = checks(start)
    if not feasible(initial):
        raise ValueError("Fixed-branch acquisition search needs a cheap-feasible seed")
    names = tuple(initial)
    scales = np.array([max(1., abs(initial[name])) for name in names])

    def inequalities(x):
        values = checks(x)
        if tuple(values) != names or any(v is None or not np.isfinite(v) for v in values.values()):
            raise ValueError("Invalid or changing cheap-constraint vector")
        return -np.array([values[name] for name in names]) / scales

    def objective(x):
        nonlocal evaluations
        evaluations += 1
        tensor = torch.tensor(x, dtype=torch.double, requires_grad=True)
        value = -acquisition(tensor.reshape(1, 1, -1)).sum()
        gradient = torch.autograd.grad(value, tensor)[0]
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite fixed-branch acquisition or gradient")
        return float(value.detach()), gradient.detach().numpy()

    report = dict(method="slsqp_exact_precheck_v1", seed_checks=initial,
                  threshold_relaxation=0., success=False)
    start_value = objective(start)[0]
    constraints = [{"type": "ineq", "fun": inequalities}] if names else []
    try:
        result = minimize(objective, start, jac=True, method="SLSQP",
                          bounds=list(zip(lower, upper)), constraints=constraints,
                          options={"maxiter": maxiter, "ftol": 1e-9})
        target = np.clip(result.x, lower, upper)
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite constrained-search point")
        raw_checks = checks(target)
        report.update(success=bool(result.success), message=str(result.message),
                      iterations=int(result.nit), raw_checks=raw_checks)
        # SLSQP's tolerance can leave a tiny positive residual. Keep the exact
        # precheck and retreat inside the same branch instead of relaxing it.
        fraction = 1.
        if not feasible(raw_checks):
            lo, hi = 0., 1.
            for _ in range(32):
                mid = (lo + hi) / 2
                if feasible(checks(start + mid * (target - start))):
                    lo = mid
                else:
                    hi = mid
            fraction = lo * (1. - 1e-8)
        point = start + fraction * (target - start)
        final_checks = checks(point)
        final_value = objective(point)[0]
        improved = feasible(final_checks) and final_value < start_value
        if not improved:
            point, final_checks = start, initial
        report.update(retained_improvement=bool(improved), feasible_fraction=fraction,
                      final_checks=final_checks, acquisition_before=-start_value,
                      acquisition_after=-final_value if improved else -start_value)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        point = start
        report.update(error=f"{type(error).__name__}: {error}", final_checks=initial,
                      retained_improvement=False)
    report["acquisition_evaluations"] = evaluations
    return torch.as_tensor(point, dtype=torch.double), report
