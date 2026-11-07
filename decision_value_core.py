"""Phase B0: fixed risk rankings on archived pools; stdlib, no learners/evaluator.

Posthoc diagnostic only. A budget is a prefix of one frozen ranking, NOT an
adaptive rollout. Candidate IDs break ties independently of any outcome.
"""
import hashlib
import json
import math

SCHEMA = 'phase_b0_frozen_risk_ranking_v1'
PARENT = 'bc0164882faa6ff6390d62d0e1452717af00416b'
EVIDENCE = 'fba37629d003e2b528b7c3e3c5f50d3f245eb21d'
EVIDENCE_PATH = 'results/risk-learnability-phase-a-20261107/full'
BUDGETS = (1, 2, 4, 6, 12)
TIE_SALT = 'RA_LSBO_PHASE_B0_OUTCOME_BLIND_TIE_V1'
TASKS = ('legacy_branin_sparse', 'legacy_rosenbrock_sparse',
         'tradeoff_s0', 'tradeoff_s1', 'tradeoff_s2', 'tradeoff_s3')
SIZES = (12, 24, 48, 96, 192)
MODELS = ('joint_prevalence', 'native_censored', 'native_sign', 'privileged_exact')
NEW_BILL = dict(new_true_evaluations=0, new_gp_fits=0, new_vae_fits=0,
                new_fantasy_fits=0, aircraft_calls=0, rcaide_calls=0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def fixed_plan():
    return dict(schema=SCHEMA, input_code_sha=PARENT, input_result_sha=EVIDENCE,
                input_profile='full', tasks=list(TASKS), repeats=[0, 1, 2],
                sizes=list(SIZES), methods=list(MODELS), budgets=list(BUDGETS),
                pool_size=512, source_model_rows=360, historical_true_evaluations=12672,
                policy='descending archived joint-feasibility probability',
                ties='sha256 of fixed salt and candidate ID; no outcome access',
                adaptive_updates=False, objective_in_selection=False,
                true_test_labels_only_for_scoring=True, posthoc=True,
                independent_confirmation=False, new_bill=dict(NEW_BILL))


def rank_candidates(ids, probabilities):
    """Selection interface intentionally accepts neither truth nor objective."""
    if not ids or len(ids) != len(probabilities) or len(set(ids)) != len(ids):
        raise ValueError('Nonempty aligned unique candidate IDs required')
    if any(not isinstance(cid, str) or len(cid) != 64 or
           any(c not in '0123456789abcdef' for c in cid) for cid in ids):
        raise ValueError('Canonical candidate SHA256 required')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities):
        raise ValueError('Finite probabilities in [0,1] required')
    keys = [digest([TIE_SALT, cid]) for cid in ids]
    return sorted(range(len(ids)), key=lambda i: (-probabilities[i], keys[i], ids[i]))


def random_reference(pool_size, feasible, budget):
    """Exact finite-pool, uniform without-replacement expectation, not MC."""
    if (any(type(x) is not int for x in (pool_size, feasible, budget)) or
            pool_size < 1 or not 0 <= feasible <= pool_size or not 1 <= budget <= pool_size):
        raise ValueError('Invalid finite population or budget')
    survival = 1.0
    for j in range(budget):
        survival *= max(pool_size-feasible-j, 0)/(pool_size-j)
    return dict(pool_size=pool_size, feasible_in_pool=feasible, budget=budget,
                expected_feasible=budget*feasible/pool_size,
                probability_any_feasible=1.0-survival,
                type='exact_uniform_without_replacement_expectation')


def score_prefix(order, labels, budget):
    if (type(budget) is not int or not 1 <= budget <= len(labels) or
            sorted(order) != list(range(len(labels))) or
            any(type(y) is not bool for y in labels)):
        raise ValueError('Permutation, Boolean labels and valid budget required')
    flags = [labels[i] for i in order[:budget]]
    first = next((j+1 for j, y in enumerate(flags) if y), None)
    return dict(budget=budget, feasible_selected=sum(flags), any_feasible=bool(sum(flags)),
                first_feasible_rank=first, nonfeasible_selected=budget-sum(flags),
                fixed_ranking_no_updates=True)


def average(values):
    if not values or any(x is None for x in values):
        return None
    return math.fsum(values)/len(values)


def aggregate(rows):
    """Task-level means + all 3 repeat values; no pooled cross-task objectives."""
    out = {}
    for task in TASKS:
        for n in SIZES:
            for method in MODELS:
                group = sorted([r for r in rows if (r['task'], r['n'], r['method']) ==
                                (task, n, method)], key=lambda r: r['repeat'])
                if [r['repeat'] for r in group] != [0, 1, 2]:
                    raise ValueError('Expected exactly three repeat rows')
                good = [r for r in group if r['status'] == 'OK']
                row = dict(planned_repeats=3, successful_repeats=len(good),
                           failed_repeats=3-len(good),
                           joint_metrics_mean={}, decisions={})
                for key in ('brier', 'log_loss', 'auroc', 'precision_at_10pct',
                            'recall_at_10pct', 'false_feasible_rate', 'false_infeasible_rate'):
                    row['joint_metrics_mean'][key] = average([r['metrics'][key] for r in good])
                row['brier_per_repeat'] = [dict(repeat=r['repeat'], value=r['metrics']['brier']
                                              if r['status']=='OK' else None) for r in group]
                for k in BUDGETS:
                    ds = [next(d for d in r['decisions'] if d['budget']==k) for r in good]
                    row['decisions'][str(k)] = dict(
                        evaluated_repeats=len(ds), discovered_repeats=sum(d['any_feasible'] for d in ds),
                        mean_feasible_selected=average([d['feasible_selected'] for d in ds]),
                        random_expected_discovered_repeats=math.fsum(
                            d['random_reference']['probability_any_feasible'] for d in ds),
                        random_mean_feasible_selected=average([
                            d['random_reference']['expected_feasible'] for d in ds]),
                        first_feasible_ranks=[d['first_feasible_rank'] for d in ds],
                        repeat_ids=[r['repeat'] for r in good],
                        not_a_statistical_significance_test=True)
                out['%s/n%d/%s' % (task, n, method)] = row
    return out
