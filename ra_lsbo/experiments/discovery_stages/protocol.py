"""New plans only. Historical plans and profiles are never rewritten."""
import platform
import subprocess
from pathlib import Path

from ...budget import digest
from ..comparison.protocol import build_plan as legacy_plan
from ..core.protocol import BASE, ROOT, budget_for, sha
from .methods import REFERENCE, STAGES, specification

SCHEMA = 'ra_lsbo_discovery_stages_v1'
DOCUMENT = 'docs/DISCOVERY_STAGES_PROTOCOL_20261029.md'


def code_identity():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unversioned-checkout'


def build_plan(stage='discovery', profile='pilot'):
    if stage not in STAGES or profile not in {'smoke', 'pilot'}:
        raise ValueError('Only exploratory math stages and smoke/pilot are implemented; no confirm or MDA')
    plan = legacy_plan(profile, 'mathematical')
    plan.update(schema=SCHEMA, stage=stage, arms=list(STAGES[stage]),
                budget=budget_for(plan['profile'], STAGES[stage]),
                reference=REFERENCE[stage], methods={m: specification(m, plan['profile']) for m in STAGES[stage]},
                settings={}, adaptations={}, source_commit=code_identity(),
                staged_policy={'coverage_period': 5, 'near_tolerance': 1e-8,
                    'coverage_scope': 'scheduled all phases; singleton gate fallback; exact precheck only',
                    'feedback_modes': ['exact', 'sign', 'interval'],
                    'cold_local_rule': 'disabled until first usable feasible, unless coverage-only control',
                    'nominal_horizon_retained_after_native_stop': True,
                    'publication_confirmation': False,
                    'feedback_limitation': 'current math tasks mainly exact+sign; not a finite-interval test'},
                statistical_endpoints={'primary': 'cold_start_discovery_at_horizon',
                    'secondary': ['first_feasible_online_censoring', 'warm_start_improvement',
                                  'conditional_paired_objective', 'lexicographic_paired_descriptive'],
                    'inference': 'development seeds already inspected; no confirmatory superiority claim'})
    plan['runtime']['platform'] = platform.platform()
    plan['comparison_protocol']['original_lsbo_in_primary_roster'] = 'original_lsbo' in STAGES[stage]
    plan['comparison_protocol'].pop('reason_original_lsbo_omitted', None)
    plan['comparison_protocol']['candidate_generation'] = (
        'common screened global stream; registered RA variants may augment; native LSBO has original search')
    for path in [ROOT/DOCUMENT, *sorted((ROOT/'tests').glob('test_discovery_stages*.py'))]:
        plan['sources'][str(path)] = sha(path)
    return plan


def verify_plan(plan):
    if plan.get('schema') != SCHEMA:
        raise ValueError('Wrong staged schema; never reuse an old plan')
    # Rebuild the full prospective specification, not just a few asserted keys.
    expected = build_plan(plan['stage'], plan['profile']['name'])
    omit = {'registered_utc', 'prerequisite'}
    if digest({k:v for k,v in plan.items() if k not in omit}) != digest({k:v for k,v in expected.items() if k not in omit}):
        raise ValueError('Staged specification, source, runtime, commit or task changed')
    return None
