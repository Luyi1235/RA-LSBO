"""Outcome-adaptive versus fixed-second scoring on the exact next candidate pool.

Both arms condition every observable fantasy using the SAME frozen EP adapter.
Only max/expectation order changes. Actual observations and subsequent online
replanning are full in BOTH arms; fixed-second is a scoring ablation, not a
commitment to execute a previously planned point after new feedback arrives.
"""
from copy import deepcopy
import time
import numpy as np
from ra_lsbo.budget import digest
from pr70_future_pool import NextPoolBO, NextPoolFactory
from ra_lsbo.experiments.information_value_stage3.planner import (
    FantasyFitError, choose as parent_choose, one_step, probability, shortlist, nodes, outcomes)
from recourse_value_core import ADAPTIVE, FIXED, decompose, require


def choose_recourse(forecast, best, remaining, factory, *, arm, seed, iteration):
    require(arm in (ADAPTIVE, FIXED) and type(remaining) is int and remaining >= 1, 'Invalid arm/budget')
    t0 = time.perf_counter()
    if remaining == 1:
        i, r = parent_choose(forecast, best, remaining, factory, arm='cei_myopic', seed=seed, iteration=iteration)
        r.update(arm=arm, continuation_support='actual_next_pool', future_supports={},
                 selected_future_support=None, recourse_decomposition=None,
                 fixed_second_is_scoring_only=True)
        return i, r
    base = one_step(forecast, best)
    require(np.isfinite(base).all() and not np.any(base < 0), 'Invalid Q1')
    firsts = shortlist(forecast, base)
    unit_nodes = nodes(seed, iteration)
    supports = {str(i): deepcopy(factory.support(i)) for i in firsts}
    myopic = int(np.argmax(base))
    r = {'arm': arm, 'depth': 2, 'remaining_actual_calls': remaining,
         'criterion': 'first_feasible_probability' if best is None else 'expected_feasible_objective_improvement',
         'myopic_index': myopic, 'myopic_scores': base.tolist(), 'first_action_indices': firsts,
         'fantasies': [], 'fallback': None, 'actual_feedback_always_full': True,
         'condition_on_abort_interval': True, 'quadrature_nodes_per_region_branch': len(unit_nodes),
         'probability_guarantee': False, 'stop_action_in_experiment': False,
         'continuation_support': 'actual_next_pool', 'fixed_second_is_scoring_only': True}
    selected, scores, decomposition = myopic, [None]*len(base), None
    try:
        for i in firsts:
            points = supports[str(i)]['model_candidates']
            for weight, outcome in outcomes(forecast, i, unit_nodes, best is not None):
                terminal = best is None and outcome.feasible
                row = {'first_index': i, 'weight': weight, 'observation': outcome.packet(),
                       'conditioned': not terminal, 'abort_interval_used': bool(outcome.aborted)}
                if terminal:
                    values = np.zeros(len(points), float)
                    second, second_hash = None, None
                else:
                    updated = factory.condition(i, outcome, use_abort_interval=True)
                    next_best = min(best, outcome.objective) if best is not None and outcome.feasible else best
                    values = one_step(updated, next_best)
                    require(len(values) == len(points), 'Future prediction/support length mismatch')
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise FantasyFitError('Invalid future continuation vector')
                    second = int(np.argmax(values))
                    second_hash = digest(points[second])
                row.update(future_values=values.tolist(), second_index=second,
                           second_candidate_sha256=second_hash, continuation=float(np.max(values)))
                r['fantasies'].append(row)
        decomposition = decompose(base.tolist(), firsts, r['fantasies'],
                                  {i: len(supports[str(i)]['model_candidates']) for i in firsts})
        which = 'adaptive' if arm == ADAPTIVE else 'fixed'
        selected = decomposition[which+'_index']
        for i in firsts:
            scores[i] = decomposition['roots'][str(i)][which+'_score']
    except FantasyFitError as error:
        r.update(fallback='fantasy_numerical_failure_full_myopic', error=str(error),
                 partial_lookahead_scores_not_used=scores)
        scores = base.tolist()
    r.update(selected_index=selected, scores=scores, future_supports=supports,
             selected_future_support=supports[str(selected)], recourse_decomposition=decomposition,
             planner_seconds=time.perf_counter()-t0)
    return selected, r


class RecourseValueBO(NextPoolBO):
    def __init__(self, problem, base, seed, output, *, case, horizon, arm):
        require(arm in (ADAPTIVE, FIXED), 'Unsupported recourse arm')
        super().__init__(problem, base, seed, output, case=case, horizon=horizon)
        self.recourse_arm = arm

    def _choose_acquisition(self, predictions, best, iteration, acquisition, records):
        candidates = self._predicted_candidates
        require(self._posterior is not None and bool(candidates), 'Missing bound posterior')
        key = digest([iteration, best, candidates, [r.to_dict() for r in records]])
        if key in self._decision_cache:
            return deepcopy(self._decision_cache[key])
        factory = NextPoolFactory(self._posterior, candidates, self.problem, self.next_case,
                                  self.seed, iteration, records)
        index, receipt = choose_recourse(predictions, best, self.settings.horizon-iteration,
            factory, arm=self.recourse_arm, seed=self.seed, iteration=iteration)
        pf = probability(predictions['constraints']['region']) * probability(predictions['constraints']['resource'])
        receipt.update(candidate_pool_hash=digest(candidates),
            pre_outcome_history_hash=digest([r.to_dict() for r in records]),
            selected_candidate_hash=digest(candidates[index]), selected_predicted_failure=float(1-pf[index]),
            fantasy_fits=factory.diagnostics(), actual_action='evaluate_fixed_budget')
        meta = {'mode': self.recourse_arm+':'+receipt['criterion'], 'eligible': [True]*len(candidates),
                'scores': receipt['scores'], 'fallback': receipt['fallback'], 'kappa': 0.,
                'feasibility_proxy_no_status': pf.tolist(), 'information_value_decision': receipt,
                'probability_guarantee': False}
        self._decision_cache[key] = deepcopy((index, meta))
        return index, meta
