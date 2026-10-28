"""Isolated v49 evaluation with a versioned battery post-processing correction."""

import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from lsbo.records import Evaluation
from .problem import BACKEND_ID, REQUIRED_CONSTRAINTS, LEGACY_CONSTRAINTS, required_constraints_for


def translate_result(raw, required_constraints=REQUIRED_CONSTRAINTS):
    from .operating_limits import CONSTRAINT_NAMES, CONSTRAINT_POLICY
    if (raw.metrics.get('operating_limit_policy') == CONSTRAINT_POLICY
            and not set(CONSTRAINT_NAMES).issubset(required_constraints)):
        raise ValueError('Audited operating constraints require the extended feasibility schema')
    energy = raw.mission_energy_Wh
    valid = energy is not None and math.isfinite(float(energy))
    success = valid and raw.status in ("ok", "constraint_violation")
    result = Evaluation(float(energy) if valid else None, success,
                        dict(raw.constraints), raw.status, dict(raw.metrics))
    result.metrics["backend_feasible"] = bool(raw.feasible)
    segments = result.metrics.get('mission_solver_segments', [])
    if success and segments and all(s.get('converged') for s in segments):
        powers = [e['minimum_terminal_power_W'] for s in segments for e in s.get('electrical_summary', [])
                  if 'minimum_terminal_power_W' in e]
        if len(powers) == len(segments) and all(math.isfinite(v) for v in powers):
            result.metrics['solver_discharge_margin_W'] = min(powers)
            result.metrics['solver_discharge_margin_source'] = 'converged_original_discharge_model'
    elif not success:
        for segment in segments:
            detail = segment.get('seed_continuation', {})
            value = detail.get('minimum_trial_terminal_power_W')
            if (detail.get('residual_inf', float('inf')) <= 1e-6 and value is not None and
                    math.isfinite(value) and value < 0 and not detail.get('original_discharge_verified', False)):
                result.metrics['solver_discharge_margin_W'] = value
                result.metrics['solver_discharge_margin_source'] = 'inadmissible_seed_root_not_a_valid_mission'
                break
    if result.feasible(LEGACY_CONSTRAINTS) != bool(raw.feasible):
        raise ValueError("Backend feasibility disagrees with the six-constraint adapter contract")
    if success:
        if 'usable_energy_constraint' in result.metrics:
            result.constraints['usable_energy'] = float(result.metrics['usable_energy_constraint'])
        elif 'usable_energy' not in result.constraints:
            raise ValueError('Missing NASA usable-energy constraint from audited evaluation')
        if not math.isfinite(result.constraints['usable_energy']):
            raise ValueError('Non-finite usable-energy constraint')
        from .propulsion_geometry import GEOMETRY_CONSTRAINTS
        geometry = result.metrics.get('propulsion_geometry', {}).get('constraints', {})
        for name in GEOMETRY_CONSTRAINTS:
            if name in geometry:
                result.constraints[name] = float(geometry[name])
            if name not in result.constraints or not math.isfinite(result.constraints[name]):
                raise ValueError('Missing/nonfinite propulsion geometry constraint: ' + name)
        electrical = result.metrics.get('battery_operating_constraints', {})
        for name in CONSTRAINT_NAMES:
            if name in required_constraints:
                if result.metrics.get('operating_limit_policy') != CONSTRAINT_POLICY:
                    raise ValueError('Missing audited operating-limit policy')
                value = electrical.get(name)
                if value is None or not math.isfinite(value):
                    raise ValueError('Missing/nonfinite operating constraint: ' + name)
                result.constraints[name] = float(value)
        if not result.feasible(required_constraints):
            result.status = 'constraint_violation'
        result.metrics['nasa_feasible'] = result.feasible(required_constraints)
    return result


def solver_working_directory(model_dir, artifact_dir):
    """Keep relative legacy assets while isolating native solver scratch files."""
    if artifact_dir is None:
        return model_dir
    work = Path(artifact_dir).resolve() / 'solver_work'
    work.mkdir(parents=True, exist_ok=False)
    shutil.copytree(model_dir / 'Airfoils', work / 'Airfoils')
    return work


def evaluate(payload, artifact_dir=None):
    if payload["config"].get("backend") != BACKEND_ID:
        raise ValueError("Unsupported physical backend; use current geometry-v2 config, not old run config")
    required_constraints = required_constraints_for(payload['config'])
    from lsbo.space import DesignSpace, Candidate
    from .problem import variable_schema
    from .propulsion_geometry import normalize_geometry
    from .constraints import analytical_checks, rejection_result, CONSTRAINT_POLICY, CHEAP_DESIGN_CONSTRAINTS
    canonical = DesignSpace.from_dict({'variables': variable_schema(payload['config'])}).canonical(Candidate(**payload['candidate']))
    candidate = canonical.to_dict()
    geometry_settings = normalize_geometry(payload['config'].get('geometry'))
    checks, plan = analytical_checks(canonical, geometry_settings)
    rejected = rejection_result(checks, plan)
    if rejected is not None:
        return rejected
    from .verify_model import verify_snapshot
    verify_snapshot()
    model_dir = Path(__file__).resolve().parent / "model"
    # These legacy modules use absolute local imports and relative asset paths.
    # Parallel MDA workers must never share a writable solver cwd.
    os.chdir(solver_working_directory(model_dir, artifact_dir))
    sys.path.insert(0, str(model_dir))
    from . import runtime
    import candidate_evaluator
    for module in (runtime.Procedure, runtime.Vehicle_Module, candidate_evaluator):
        if Path(module.__file__).resolve().parent != model_dir:
            raise RuntimeError("Physical module was loaded from outside the bundled snapshot")
    if 'design_study' in payload['config']:
        import design_space
        if Path(design_space.__file__).resolve().parent != model_dir:
            raise RuntimeError("Study bounds require the bundled legacy design space")
        from .design_study import install_legacy
        install_legacy(design_space, payload['config']['design_study'])
    print("Building original v49 vehicle, analyses and mission", flush=True)
    nexus = runtime.setup()
    from .mission_profile import configure_mission
    mission_diagnostics = configure_mission(nexus, payload['config'].get('mission'))
    from .battery_energy import install_energy_model
    energy_diagnostics = install_energy_model(nexus)
    from .postprocess import install_postprocess
    from .operating_limits import LEGACY_POLICY
    diagnostics = install_postprocess(nexus, payload['config'].get('operating_limit_policy', LEGACY_POLICY))
    if 'design_study' in payload['config']:
        diagnostics['design_study'] = payload['config']['design_study']
    diagnostics.update(energy_diagnostics)
    diagnostics.update(mission_diagnostics)
    no_load_fraction = None
    if 'component_scaling' in payload['config']:
        from .component_scaling import prepare
        no_load_fraction = prepare(candidate_evaluator,nexus,diagnostics,payload['config']['component_scaling'])
    from .motor_design import install_motor_design
    install_motor_design(candidate_evaluator, diagnostics, no_load_fraction=no_load_fraction)
    from .propulsion_geometry import install_propulsion_geometry
    install_propulsion_geometry(candidate_evaluator, nexus, diagnostics, geometry_settings)
    if 'propulsion_research' in payload['config']:
        from .propulsion_research import install
        install(candidate_evaluator, nexus, diagnostics, payload['config']['propulsion_research'])
    if 'component_scaling' in payload['config']:
        from .component_scaling import install_geometry
        install_geometry(candidate_evaluator,diagnostics,payload['config']['component_scaling'])
    if 'section_consistency' in payload['config']:
        if 'propulsion_research' not in payload['config']:
            raise ValueError('Section study requires research physics')
        from .section_consistency import install
        install(nexus, diagnostics, payload['config']['section_consistency'])
    if 'dep_groups' in payload['config']:
        from .dep_groups import install
        install(nexus, diagnostics, payload['config']['dep_groups'])
    if 'powered_lift_probe' in payload['config']:
        from .powered_lift_probe import install
        probe_execute = install(nexus, candidate_evaluator, diagnostics, payload['config']['powered_lift_probe'], artifact_dir)
    if 'simplified_powered_lift' in payload['config']:
        from .simplified_powered_lift import install
        install(nexus, candidate_evaluator, diagnostics, payload['config'], artifact_dir)
    if 'mission_slipstream' in payload['config']:
        from .mission_slipstream import install
        install(nexus, diagnostics, payload['config'])
    if 'design_study' in payload['config']:
        from .design_study import install_area_audit
        install_area_audit(nexus, diagnostics, candidate['continuous']['wing_area'])
    if payload.get('diagnostic_mode') is not None:
        if payload['diagnostic_mode']!='powered_lift_only' or 'powered_lift_probe' not in payload['config']:
            raise ValueError('Unsupported isolated diagnostic mode')
        from .powered_lift_probe import geometry_only
        return geometry_only(nexus,candidate_evaluator,candidate,probe_execute,diagnostics)
    from .mission_solver import install_mission_solver
    install_mission_solver(nexus, diagnostics)
    from .evaluation_diagnostics import install_procedure_diagnostics
    install_procedure_diagnostics(nexus, diagnostics)
    print("[RCAIDE] Energy model: " + json.dumps(energy_diagnostics), flush=True)
    print("[RCAIDE] Mission solver: " + diagnostics['mission_solver_version'], flush=True)
    print("[RCAIDE] Motor design: " + diagnostics['motor_design_version'], flush=True)
    print("[RCAIDE] Mission profile: " + json.dumps(mission_diagnostics), flush=True)
    print("Evaluating NASA 30Q reference: charge SOC, OCV-integrated energy, 47 kWh usable output limit", flush=True)
    from .battery_diagnostics import summarize_domain_trials
    with summarize_domain_trials(diagnostics, artifact_dir):
        raw = candidate_evaluator.evaluate_candidate(nexus, candidate["continuous"], candidate["discrete"])
    if ('design_study' in payload['config'] and raw.status in ('ok', 'constraint_violation')
            and 'design_study_realized_wing_area_m2' not in diagnostics):
        raise ValueError("Successful study evaluation is missing its realized-area audit")
    failure = diagnostics.pop('_solver_failure_report', None)
    if failure is not None:
        if artifact_dir is not None:
            failure_path = Path(artifact_dir) / 'solver_error.json'
            failure_path.write_text(json.dumps(failure, indent=2, allow_nan=False), encoding='utf-8')
            diagnostics['solver_error_file'] = str(failure_path)
        else:
            diagnostics['solver_failure_report'] = failure
    audit = diagnostics.pop("_energy_audit_report", None)
    if audit is not None and artifact_dir is not None:
        audit_path = Path(artifact_dir) / "energy_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        diagnostics["energy_audit_file"] = str(audit_path)
    raw.metrics.update(diagnostics)
    raw.metrics['constraint_policy'] = CONSTRAINT_POLICY
    raw.metrics['mda_evaluated'] = True
    if raw.status in ('ok', 'constraint_violation'):
        for name in CHEAP_DESIGN_CONSTRAINTS:
            if name not in raw.constraints or not math.isclose(raw.constraints[name], checks[name], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Precheck/postcheck contract mismatch: ' + name)
    print("[RCAIDE] Post-processing: " + json.dumps(diagnostics), flush=True)
    result = translate_result(raw, required_constraints)
    print(f'[RCAIDE] Candidate result: status={result.status}, objective={result.objective}, '
          f'feasible={result.feasible(required_constraints)}; '
          f'failure_details={diagnostics.get("solver_error_file", "none")}', flush=True)
    return result


def main():
    request, response = (Path(value).resolve() for value in sys.argv[1:3])
    started = time.perf_counter()
    try:
        payload = json.loads(request.read_text(encoding="utf-8"))
        result = evaluate(payload, response.parent)
    except Exception as error:
        traceback.print_exc()
        result = Evaluation(None, False, status="worker_error:" + type(error).__name__,
                            metrics={"message": str(error)})
    result.elapsed_seconds = time.perf_counter() - started
    temporary = response.with_suffix(".tmp")
    temporary.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(response)


if __name__ == "__main__":
    main()
