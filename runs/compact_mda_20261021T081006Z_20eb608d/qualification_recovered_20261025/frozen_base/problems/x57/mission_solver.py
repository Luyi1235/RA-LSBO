"""Install a scoped battery-domain fallback without editing the RCAIDE package."""

from copy import deepcopy
import sys
import time
import numpy as np
from .domain_solver import solve, DomainSolveError
from .nasa_battery import BatteryDomainError


SOLVER_VERSION = 'battery_domain_bounded_resume_v1'


def _operating_summary(segment, report):
    energy = getattr(segment.state.conditions, 'energy', {})
    busses = []
    for key, bus in energy.items():
        if not hasattr(bus, 'battery_modules') or not hasattr(bus, 'power_draw'):
            continue
        power = np.asarray(bus.power_draw, dtype=float)
        if power.size and np.all(np.isfinite(power)):
            entry = {'bus': str(key), 'maximum_terminal_power_W': float(np.max(power)),
                     'minimum_terminal_power_W': float(np.min(power))}
            soc = np.asarray(bus.state_of_charge, dtype=float)
            if soc.size and np.all(np.isfinite(soc)):
                entry['initial_charge_soc'] = float(soc.reshape(-1)[0])
                entry['final_charge_soc'] = float(soc.reshape(-1)[-1])
            frames = getattr(segment.state.conditions, 'frames', None)
            if frames is not None:
                times = np.asarray(frames.inertial.time).reshape(-1)
                watts = power.reshape(-1)
                if times.size == watts.size and times.size > 1:
                    dt = np.diff(times)
                    if np.all(np.isfinite(dt)) and np.all(dt > 0):
                        entry['duration_s'] = float(times[-1] - times[0])
                        entry['terminal_energy_Wh'] = float(np.dot(watts[:-1], dt) / 3600.)
            busses.append(entry)
    if busses:
        report['electrical_summary'] = busses


def _fallback(segment, pristine, report):
    unknowns = pristine.unknowns
    keys = [key for key in unknowns.keys() if key != 'tag']
    if not keys or any(key != 'body_angle' and not key.startswith('throttle_')
                       and not key.endswith('_motor_current') for key in keys):
        raise RuntimeError(f'Unsupported battery-domain fallback unknowns: {keys}')
    report['unknown_fields'] = keys
    scales = deepcopy(unknowns)
    for key in keys:
        factor = max(1., float(np.max(np.abs(unknowns[key])))) if key.endswith('_motor_current') else 1.
        scales[key] = np.ones_like(unknowns[key]) * factor
    scale = scales.pack_array()
    seeds = []
    previous = getattr(pristine.initials, 'unknowns', {})
    if all(key in previous for key in keys):
        guess = deepcopy(unknowns)
        for key in keys:
            guess[key] = np.ones_like(unknowns[key]) * float(np.asarray(previous[key]).reshape(-1)[-1])
        seeds.append(guess.pack_array() / scale)
        # Reduce current and voltage separately: scaling both by the same factor
        # need not preserve a useful motor/propeller operating point.
        for throttle_factor, current_factor in ((.9, .6), (.8, .4), (.7, .2)):
            guess = deepcopy(unknowns)
            for key in keys:
                factor = throttle_factor if key.startswith('throttle_') else (
                    current_factor if key.endswith('_motor_current') else 1.)
                guess[key] = np.ones_like(unknowns[key]) * float(np.asarray(previous[key]).reshape(-1)[-1]) * factor
            seeds.append(guess.pack_array() / scale)
    for factor in (1., .75, .5, .25, .1):
        guess = deepcopy(unknowns)
        for key in keys:
            if key.startswith('throttle_') or key.endswith('_motor_current'):
                guess[key] = np.asarray(unknowns[key]) * factor
        seeds.append(guess.pack_array() / scale)

    best_norm, best_x = float('inf'), None
    def evaluate(x, record=True):
        nonlocal best_norm, best_x
        # A failed battery integration leaves partial histories. Recreate the
        # entire pre-solve state for every evaluation, including segment initials.
        segment.state = deepcopy(pristine)
        segment.conditions = segment.state.conditions
        segment.state.unknowns.unpack_array(x * scale)
        segment.process.iterate(segment)
        residual = segment.state.residuals.pack_array()
        norm = float(np.max(np.abs(residual)))
        if record and np.isfinite(norm) and norm < best_norm:
            best_norm = norm
            best_x = np.array(x, copy=True)
            report['best_evaluable_trial'] = {
                'residual_inf': norm,
                'unknown_ranges': {key: [float(np.min(segment.state.unknowns[key])),
                                        float(np.max(segment.state.unknowns[key]))] for key in keys}}
            report['best_evaluable_trial']['residual_ranges'] = {
                key: [float(np.min(value)), float(np.max(value))]
                for key, value in segment.state.residuals.items() if isinstance(value, np.ndarray) and value.size}
            _operating_summary(segment, report['best_evaluable_trial'])
        return residual

    settings = pristine.numerics.solver
    # Native SLSQP interprets its limit as iterations, while this fallback counts
    # residual calls (a full finite-difference Jacobian alone costs N calls).
    budget = min(1200, max(int(settings.max_evaluations), 10 * (scale.size + 1) + 32))
    report['unknown_dimension'] = int(scale.size)
    report['max_residual_evaluations'] = budget
    def progress(info):
        print(f'[RCAIDE] {segment.tag} fallback: evaluations={info["evaluations"]}, '
              f'residual_inf={info["residual_inf"]:.6g}, '
              f'rejected={info["rejected_domain_trials"]}', flush=True)
    try:
        solve(evaluate, seeds, tolerance=float(settings.tolerance_solution),
              max_evaluations=budget, report=report, progress=progress, select_best_seed=True)
    except DomainSolveError as original_error:
        # Near-converged original-model iterates can run out of calls while
        # rebuilding a finite-difference Jacobian. Resume once, not indefinitely.
        if (best_x is not None and 'evaluation budget exhausted' in str(original_error)
                and best_norm <= 1e-2):
            detail = {'initial_residual_inf': best_norm,
                      'max_residual_evaluations': min(400, 3 * (scale.size + 1) + 32),
                      'scope': 'one_bounded_resume_original_discharge_model'}
            report['budget_continuation'] = detail
            try:
                solve(evaluate, [best_x.copy()], tolerance=float(settings.tolerance_solution),
                      max_evaluations=detail['max_residual_evaluations'], report=detail, progress=progress)
                report.update(converged=True, residual_inf=detail['residual_inf'])
                return
            except DomainSolveError as error:
                detail['failure'] = str(error)
            finally:
                report['total_original_model_evaluations'] = report.get('evaluations', 0) + detail.get('evaluations', 0)
        if best_x is None or 'nonnegative' not in report.get('last_domain_error',''):
            raise
        from scipy.optimize import root
        from .solver_seed_extension import signed_seed_extension
        detail = {'scope': 'numerical_seed_only_not_a_recharge_model', 'evaluations': 0}
        report['seed_continuation'] = detail
        def extended(x):
            if detail['evaluations'] >= 400:
                raise DomainSolveError('Numerical seed continuation budget exhausted')
            detail['evaluations'] += 1
            return evaluate(x, record=False)
        try:
            with signed_seed_extension():
                answer = root(extended, best_x, method='hybr', options={'xtol':1e-8,'maxfev':398})
                residual = extended(answer.x)
                if not np.all(np.isfinite(residual)):
                    raise DomainSolveError('Nonfinite numerical seed residual')
                detail['residual_inf'] = float(np.max(np.abs(residual)))
                powers = [np.asarray(bus.power_draw) for bus in segment.state.conditions.energy.values()
                          if hasattr(bus,'battery_modules') and hasattr(bus,'power_draw')]
                detail['minimum_trial_terminal_power_W'] = min((float(np.min(p)) for p in powers),default=None)
            # Never accept extended histories: restore pristine state and call
            # the original discharge equations at this seed, with full residuals.
            final = evaluate(answer.x)
            norm = float(np.max(np.abs(final)))
            if not np.isfinite(norm) or norm > float(settings.tolerance_solution):
                raise DomainSolveError('Seed failed original discharge residual verification')
            detail['original_discharge_verified'] = True
            report.update(converged=True, residual_inf=norm)
        except (BatteryDomainError, DomainSolveError) as error:
            detail['original_discharge_verified'] = False
            detail['failure'] = str(error)
            segment.state = deepcopy(pristine)
            segment.conditions = segment.state.conditions
            raise DomainSolveError(f'{original_error}; seed continuation not admissible: {error}') from error


def _wrap(native, diagnostics):
    def checked(segment):
        started = time.perf_counter()
        report = {'segment': str(segment.tag), 'method': 'native', 'converged': False}
        start, end, rate = (getattr(segment, key, None)
                            for key in ('altitude_start', 'altitude_end', 'climb_rate'))
        if start is not None and end is not None and rate:
            report['specified_duration_s'] = float((end - start) / rate)
        diagnostics.setdefault('mission_solver_segments', []).append(report)
        pristine = deepcopy(segment.state)
        output = sys.stdout
        settings = pristine.numerics.solver
        try:
            if getattr(settings, 'lsbo_domain_first', False):
                if settings.objective is not None:
                    raise RuntimeError('Residual-only solver cannot replace a mission optimization objective')
                report['method'] = 'domain_backtracking_primary'
                segment.converged = False
                _fallback(segment, pristine, report)
                segment.converged = True
                segment.state.numerics.solver.converged = True
                return
            try:
                native(segment)
            except BatteryDomainError as error:
                report['native_failure'] = str(error)
            else:
                if bool(segment.converged):
                    report['converged'] = True
                    return
                report['native_failure'] = 'native solver did not converge'
            finally:
                sys.stdout = output
            if settings.objective is not None:
                raise RuntimeError('Residual-only fallback cannot replace a mission optimization objective')
            segment.converged = False
            report['method'] = 'domain_backtracking'
            print(f'[RCAIDE] {segment.tag}: retry with battery-domain backtracking; '
                  f'{report["native_failure"]}', flush=True)
            _fallback(segment, pristine, report)
            segment.converged = True
            segment.state.numerics.solver.converged = True
        except Exception as error:
            segment.converged = False
            segment.state.numerics.solver.converged = False
            report['converged'] = False
            report['failure'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            sys.stdout = output
            report['elapsed_seconds'] = time.perf_counter() - started
            if report['converged']:
                try:
                    _operating_summary(segment, report)
                except Exception as error:
                    report['summary_error'] = str(error)
            print(f'[RCAIDE] {segment.tag}: method={report["method"]}, '
                  f'converged={report["converged"]}, '
                  f'fallback_evaluations={report.get("evaluations", 0)}, '
                  f'resume_evaluations={report.get("budget_continuation", {}).get("evaluations", 0)}, '
                  f'rejected_domain_trials={report.get("rejected_domain_trials", 0)}, '
                  f'residual_inf={report.get("residual_inf", "native")}', flush=True)
            if 'electrical_summary' in report:
                print(f'[RCAIDE] {segment.tag}: {report["electrical_summary"]}', flush=True)
    return checked


def install_mission_solver(nexus, diagnostics):
    diagnostics['mission_solver_version'] = SOLVER_VERSION
    seen = set()
    for mission in nexus.missions.values():
        if not hasattr(mission, 'segments'):
            continue  # RCAIDE mission containers also store a string tag.
        for segment in mission.segments.values():
            if id(segment) in seen:
                continue
            seen.add(id(segment))
            if getattr(segment, '_lsbo_domain_solver_installed', False):
                raise ValueError('Battery-domain solver already installed')
            native = segment.process.converge.solver
            segment.process.converge.solver = _wrap(native, diagnostics)
            segment._lsbo_domain_solver_installed = True
