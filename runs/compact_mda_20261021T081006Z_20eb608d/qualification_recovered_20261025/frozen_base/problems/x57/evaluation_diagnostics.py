"""Capture exceptions before the legacy evaluator reduces them to a class name."""

import math
import sys
import time
import traceback
from functools import wraps


DIAGNOSTICS_VERSION = 'procedure_traceback_stdout_restore_20260926'


def _values(container):
    return list(container.values()) if hasattr(container, 'values') else list(container)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    except (TypeError, ValueError):
        return repr(value)


def battery_snapshot(nexus):
    """Read-only boundary context; diagnostics must never mask the real exception."""
    result = []
    try:
        vehicles = [('configuration:' + str(getattr(v, 'tag', i)), v)
                    for i, v in enumerate(_values(nexus.vehicle_configurations))]
        for i, analysis in enumerate(_values(getattr(nexus, 'analyses', {}))):
            energy = getattr(analysis, 'energy', None)
            if energy is not None:
                vehicles.append(('analysis:' + str(i), energy.vehicle))
        for label, vehicle in vehicles:
            for bus in _values(vehicle.networks.electric.busses):
                modules = []
                for module in _values(bus.battery_modules):
                    modules.append({'tag': str(module.tag),
                                    'maximum_energy_J': _number(module.maximum_energy),
                                    'initial_maximum_energy_J': _number(module.initial_maximum_energy),
                                    'cell_capacity_Ah': _number(module.cell.nominal_capacity)})
                result.append({'vehicle': label, 'bus': str(bus.tag),
                               'maximum_energy_J': _number(bus.maximum_energy),
                               'voltage_V': _number(bus.voltage), 'modules': modules})
    except Exception as error:
        result.append({'snapshot_error': f'{type(error).__name__}: {error}'})
    return result


def install_procedure_diagnostics(nexus, diagnostics):
    """Function closures preserve a shared report through legacy deepcopy(nexus)."""
    diagnostics['procedure_diagnostics_version'] = DIAGNOSTICS_VERSION

    def wrap_step(step, stage):
        @wraps(step)
        def checked(candidate_nexus, *args, **kwargs):
            diagnostics['last_started_stage'] = stage
            started = time.perf_counter()
            output = sys.stdout
            before = battery_snapshot(candidate_nexus)
            print(f'[RCAIDE] START {stage}', flush=True)
            try:
                result = step(candidate_nexus, *args, **kwargs)
            except Exception as error:
                # The RCAIDE optimization driver redirects stdout to devnull
                # without finally restoring it when a residual evaluation fails.
                sys.stdout = output
                trace = traceback.format_exc()
                frames = traceback.extract_tb(error.__traceback__)
                origin = frames[-1] if frames else None
                failure = {'stage': stage, 'exception': type(error).__name__,
                           'message': str(error), 'traceback': trace,
                           'origin': {'file': origin.filename, 'line': origin.lineno,
                                      'function': origin.name} if origin else None,
                           'battery_before_stage': before,
                           'battery_after_failure': battery_snapshot(candidate_nexus)}
                diagnostics['_solver_failure_report'] = failure
                diagnostics['failure_stage'] = stage
                diagnostics['failure_message'] = f'{type(error).__name__}: {error}'
                diagnostics['failure_origin'] = failure['origin']
                print(f'[RCAIDE] FAILED {stage}: {type(error).__name__}: {error}\n{trace}', flush=True)
                raise
            sys.stdout = output
            diagnostics['last_completed_stage'] = stage
            print(f'[RCAIDE] END {stage} ({time.perf_counter() - started:.2f}s)', flush=True)
            return result
        return checked

    def walk(process, prefix):
        for name, step in list(process.items()):
            stage = prefix + '.' + str(name)
            if hasattr(step, 'items') and hasattr(step, 'evaluate'):
                walk(step, stage)
            elif callable(step):
                process[name] = wrap_step(step, stage)

    walk(nexus.procedure, 'procedure')
