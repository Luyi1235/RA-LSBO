"""Residual-only fallback: reject undefined battery trials, never fabricate states."""

import numpy as np
from .nasa_battery import BatteryDomainError


class DomainSolveError(RuntimeError):
    pass


def solve(evaluate, seeds, *, tolerance=1e-6, max_evaluations=200, report=None, progress=None,
          select_best_seed=False):
    """Damped Newton/Broyden iteration on the unchanged, unscaled residuals.

    Only BatteryDomainError is recoverable. Backtracking shortens the control
    step, not the requested battery power. Every accepted point is evaluated by
    the real model; small steps alone never qualify as convergence.
    """
    report = report if report is not None else {}
    report.update(evaluations=0, rejected_domain_trials=0, converged=False)

    def fail(message):
        detail = report.get('last_domain_error', 'none')
        raise DomainSolveError(f'{message}; evaluations={report["evaluations"]}; '
                               f'last battery-domain error: {detail}')

    def trial(x):
        if report['evaluations'] >= max_evaluations:
            fail('Battery-domain fallback evaluation budget exhausted')
        report['evaluations'] += 1
        try:
            residual = np.asarray(evaluate(x), dtype=float).reshape(-1)
        except BatteryDomainError as error:
            report['rejected_domain_trials'] += 1
            report['last_domain_error'] = str(error)
            return None
        if residual.size != x.size or not np.all(np.isfinite(residual)):
            fail('Invalid residual shape or nonfinite residual')
        return residual

    def jacobian(x, residual):
        jac = np.empty((len(residual), len(x)))
        for j in range(len(x)):
            h = 1e-5 * max(1., abs(x[j]))
            for direction in (1., -1., .1, -.1):
                probe = x.copy()
                probe[j] += direction * h
                value = trial(probe)
                if value is not None:
                    jac[:, j] = (value - residual) / (direction * h)
                    break
            else:
                fail('Cannot differentiate residual inside battery domain')
        return jac

    selected = None
    valid_seeds = []
    report['seed_trials'] = []
    for index, seed in enumerate(seeds):
        candidate = np.asarray(seed, dtype=float).reshape(-1).copy()
        value = trial(candidate)
        norm = None if value is None else float(np.max(np.abs(value)))
        report['seed_trials'].append({'index': index, 'residual_inf': norm})
        if value is not None:
            merit = float(np.dot(value, value))
            valid_seeds.append((merit, candidate, value, index))
            if selected is None or merit < selected[0]:
                selected = merit, candidate, value, index
            if not select_best_seed or norm <= tolerance:
                break
    if selected is None:
        fail('No electrically evaluable initial control guess')
    _, x, residual, report['selected_seed'] = selected
    remaining = sorted((item for item in valid_seeds if item[3] != selected[3]), key=lambda item: item[0])
    report['seed_restarts'] = []

    jac, age = None, 0
    while True:
        norm = float(np.max(np.abs(residual)))
        report['residual_inf'] = norm
        if progress is not None:
            progress(report)
        if norm <= tolerance:
            # Restore final histories after any rejected or derivative trial.
            final = trial(x)
            if final is None or np.max(np.abs(final)) > tolerance:
                fail('Final residual verification failed')
            report.update(converged=True, residual_inf=float(np.max(np.abs(final))))
            return x
        if jac is None or age >= 8:
            jac, age = jacobian(x, residual), 0
        step = np.linalg.lstsq(jac, -residual, rcond=1e-10)[0]
        if not np.all(np.isfinite(step)):
            fail('Nonfinite Newton step')
        # The adapter scales control/current unknowns before entering this solver.
        step *= min(1., .25 / max(float(np.max(np.abs(step))), 1e-30))
        merit = float(np.dot(residual, residual))
        for backtrack in range(14):
            alpha = .5 ** backtrack
            candidate = x + alpha * step
            value = trial(candidate)
            if value is not None and np.dot(value, value) < merit * (1. - 1e-4 * alpha):
                delta = candidate - x
                if np.dot(delta, delta) > 1e-24:
                    jac += np.outer(value - residual - jac @ delta, delta) / np.dot(delta, delta)
                x, residual, age = candidate, value, age + 1
                break
        else:
            if age > 0:
                jac = None
                continue
            if remaining and report['evaluations'] + len(x) + 20 < max_evaluations:
                _, x, residual, index = remaining.pop(0)
                report['seed_restarts'].append(index)
                jac, age = None, 0
                continue
            fail('No residual-reducing step inside battery domain')
