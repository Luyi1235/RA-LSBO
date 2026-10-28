"""Bundled v49 aircraft evaluator with a versioned NASA battery adapter."""

import json
from pathlib import Path

from lsbo.space import DesignSpace
from .model.design_space import CONTINUOUS_SPECS, DISCRETE_SPECS, default_discrete
from .propulsion_geometry import GEOMETRY_CONSTRAINTS, normalize_geometry
from .constraints import ANALYTICAL_CONSTRAINTS, CONSTRAINT_POLICY, analytical_checks, rejection_result


LEGACY_CONSTRAINTS = ("soc", "stall_speed", "flutter", "structure", "span_min", "span_max")
REQUIRED_CONSTRAINTS = LEGACY_CONSTRAINTS + ("usable_energy",) + GEOMETRY_CONSTRAINTS
BACKEND_ID = "workspace_v49_nasa_modII_geometry_v2"


def required_constraints_for(config):
    from .operating_limits import LEGACY_POLICY, CONSTRAINT_POLICY, CONSTRAINT_NAMES, normalize_policy
    policy = normalize_policy(config.get('operating_limit_policy', LEGACY_POLICY))
    return REQUIRED_CONSTRAINTS + (CONSTRAINT_NAMES if policy == CONSTRAINT_POLICY else ())


def variable_schema(config=None):
    defaults = default_discrete()
    variables = []
    for spec in DISCRETE_SPECS:
        variable = dict(name=spec.name, kind=spec.kind, default=defaults[spec.name], values=list(spec.values))
        if spec.name == "propeller_count":
            variable["active_when"] = {"prop_layout": ["leading_edge_distributed"]}
        variables.append(variable)
    for spec in CONTINUOUS_SPECS:
        variable = dict(name=spec.name, kind="continuous", default=spec.default, bounds=[spec.lower, spec.upper])
        if spec.active_when:
            parent, value = spec.active_when
            variable["active_when"] = {parent: [value]}
        variables.append(variable)
    if config and 'design_study' in config:
        from .design_study import apply_schema
        variables = apply_schema(variables, config['design_study'])
    return variables


class X57Problem:
    required_constraints = REQUIRED_CONSTRAINTS
    analytical_constraints = ANALYTICAL_CONSTRAINTS
    constraint_policy = CONSTRAINT_POLICY
    coverage_variables = ('prop_layout', 'propeller_count')
    solver_domain_metric = 'solver_discharge_margin_W'

    def __init__(self, config_path, work_dir):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if self.config.get("backend") != BACKEND_ID:
            raise ValueError("Use the current NASA-battery config; older energy backends are incompatible")
        if set(self.config) - {"name", "backend", "runtime", "mission", "geometry", "propulsion_research", "section_consistency", "dep_groups", "powered_lift_probe", "simplified_powered_lift", "operating_limit_policy", "mission_slipstream", "component_scaling", "design_study"}:
            raise ValueError("Unsupported physical config fields")
        self.required_constraints = required_constraints_for(self.config)
        from .mission_profile import normalize_settings
        self.config['mission'] = normalize_settings(self.config.get('mission'))
        self.config['geometry'] = normalize_geometry(self.config.get('geometry'))
        if 'propulsion_research' in self.config:
            from .propulsion_research import normalize
            self.config['propulsion_research'] = normalize(self.config['propulsion_research'])
        if 'section_consistency' in self.config:
            if 'propulsion_research' not in self.config:
                raise ValueError('Section study requires the opt-in research physics configuration')
            from .section_consistency import normalize
            self.config['section_consistency'] = normalize(self.config['section_consistency'])
        if 'dep_groups' in self.config:
            research = self.config.get('propulsion_research', {})
            if (not research.get('synchronize_mission_mass') or not research.get('correct_nacelle_drag')
                    or research.get('native_wake', True) or 'section_consistency' not in self.config):
                raise ValueError('Group study requires consistent mass/drag/sections and wake disabled')
            from .dep_groups import normalize
            self.config['dep_groups'] = normalize(self.config['dep_groups'])
        if 'powered_lift_probe' in self.config:
            if ('section_consistency' not in self.config or 'dep_groups' in self.config
                    or self.config.get('propulsion_research',{}).get('native_wake',True)):
                raise ValueError('Probe requires consistent wake-off reference without group switching')
            from .powered_lift_probe import normalize
            self.config['powered_lift_probe'] = normalize(self.config['powered_lift_probe'])
        if 'simplified_powered_lift' in self.config:
            from .simplified_powered_lift import validate_config
            self.config['simplified_powered_lift'] = validate_config(self.config)
        if 'mission_slipstream' in self.config:
            from .mission_slipstream import normalize
            self.config['mission_slipstream'] = normalize(self.config['mission_slipstream'])
        if 'component_scaling' in self.config:
            if not self.config.get('propulsion_research',{}).get('correct_nacelle_drag'):
                raise ValueError('Component scaling requires corrected nacelle aggregation')
            from .component_scaling import normalize
            self.config['component_scaling'] = normalize(self.config['component_scaling'])
        if 'design_study' in self.config:
            from .design_study import normalize
            self.config['design_study'] = normalize(self.config['design_study'])
        self.space = DesignSpace.from_dict({"variables": variable_schema(self.config)})
        self.name = self.config["name"]
        self.work_dir = Path(work_dir)

    def precheck(self, candidate):
        candidate = self.space.canonical(candidate)
        return analytical_checks(candidate, self.config['geometry'])[0]

    def evaluate(self, candidate):
        candidate = self.space.canonical(candidate)
        checks, plan = analytical_checks(candidate, self.config['geometry'])
        rejected = rejection_result(checks, plan)
        if rejected is not None:
            return rejected
        from .adapter import evaluate_isolated
        return evaluate_isolated(candidate, self.config, self.work_dir)
