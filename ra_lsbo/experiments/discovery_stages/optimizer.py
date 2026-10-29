"""New decision policies; the original RA-v2 implementation remains unchanged."""
from dataclasses import asdict

import numpy as np

from ...budget import digest
from ...partial_bo.selector import PartialBOSettings
from ...partial_refinement.optimizer import serial
from ...soc_revision.acquisition import choose_acquisition
from ...soc_revision.optimizer import RiskRefiningBO
from ...soc_revision.posterior import RiskPosterior
from .policies import (COVERAGE_PERIOD, NEAR_TOLERANCE, coverage_pick, feedback_view,
                       filter_near, nearest_distance)


class DiscoveryBO(RiskRefiningBO):
    def __init__(self, problem, base, seed, settings, *, cold_global=True, protect=True,
                 feedback='interval', random_discovery=False):
        super().__init__(problem, base, seed, settings)
        self.cold_global, self.protect = cold_global, protect
        self.feedback, self.random_discovery = feedback, random_discovery

    def _finish(self, candidate, records, metadata, prefix):
        prior = min((r.observation.objective.value for r in records
                     if r.usable_feasible and self._branch(r.candidate) == self._branch(candidate)), default=None)
        self.pending = {'design': digest(candidate), 'contract': self.problem.contract_id,
                        'records': list(records), 'prefix_digest': prefix,
                        'prior_branch_best': prior, 'branch': self._branch(candidate)}
        metadata.update(raw_history_digest=prefix, settings=asdict(self.settings),
                        cold_start=not any(r.usable_feasible for r in records),
                        feedback_view=self.feedback, protection_enabled=self.protect,
                        near_tolerance=NEAR_TOLERANCE if self.protect else None,
                        selected_history_distance=nearest_distance(
                            self.problem, candidate, [r.candidate for r in records]),
                        branch_radii=dict(self.radii), branch_stalls=dict(self.stalls),
                        probability_guarantee=False)
        return candidate, metadata

    def propose(self, records, global_pool, iteration):
        if self.pending is not None or iteration != self.next_iteration or not 0 <= iteration < self.settings.horizon:
            raise ValueError('Settle the pending staged proposal before continuing')
        prefix = digest([r.to_dict() for r in records])
        if self.expected_prefix is not None and prefix != self.expected_prefix:
            raise ValueError('Staged history prefix changed')
        seen = {r.observation.design_id for r in records}
        if (not global_pool or len({digest(c) for c in global_pool}) != len(global_pool)
                or any(c != self.problem.canonical(c) or digest(c) in seen
                       or any(not np.isfinite(v) or v > 0 for v in self.problem.precheck(c).values())
                       for c in global_pool)):
            raise ValueError('Invalid staged candidate pool')
        cold = not any(r.usable_feasible for r in records)
        pool, _, rejected_global = (filter_near(self.problem, global_pool, records) if self.protect
                                    else (list(global_pool), list(range(len(global_pool))), []))
        if not pool:
            raise RuntimeError('No novel global candidates; no automatic tolerance relaxation or repeat')
        scheduled = self.protect and iteration % COVERAGE_PERIOD == 0
        common = {'global_pool_size': len(global_pool), 'novel_global_pool_size': len(pool),
                  'global_near_rejections': rejected_global, 'local_reports': [],
                  'coverage_scheduled': scheduled, 'local_enabled': False,
                  'coverage_reason': None, 'selected_from_local_augmentation': False}
        if scheduled or (self.random_discovery and cold):
            if self.random_discovery and cold:
                # Same seed offset and per-step stream independent of posterior fitting.
                rng = np.random.default_rng(self.seed*10000+iteration+920000000)
                index, rule = int(rng.integers(len(pool))), {'rule': 'uniform_novel_global_pool'}
                reason = 'random_discovery'
            else:
                index, rule = coverage_pick(self.problem, pool, records, self.seed, iteration)
                reason = 'scheduled_coverage'
            return self._finish(pool[index], records, {**common, 'mode': reason, 'coverage_reason': reason,
                'coverage_rule': rule, 'posterior_fit_skipped': True, 'pool': pool,
                'selected_index': index, 'augmented_pool_size': len(pool)}, prefix)
        training = feedback_view(records, self.feedback)
        posterior = RiskPosterior(self.problem, self.base, training, self.seed, self.settings)
        # Information ablations must not discard the already known raw incumbent.
        posterior.best = min((r.observation.objective.value for r in records if r.usable_feasible), default=None)
        acquisition = PartialBOSettings(horizon=self.settings.horizon,
                                        boundary_fraction=self.settings.boundary_fraction)
        _, before = choose_acquisition(posterior.predict(pool), posterior.best, iteration,
                                        acquisition, self.settings.probability_exploitation)
        local = self.settings.local and not (self.cold_global and cold)
        candidates, search = (self._augment(posterior, records, pool, before) if local else
            (pool, {'local_reports': [], 'local_rejections': {},
                    'global_pool_size': len(pool), 'augmented_pool_size': len(pool)}))
        prefilter_count = len(candidates)
        candidates, _, rejected = (filter_near(self.problem, candidates, records) if self.protect
                                   else (candidates, [], []))
        if not candidates:
            raise RuntimeError('No novel model candidates; stop without a model call')
        predictions = posterior.predict(candidates)
        index, selection = choose_acquisition(predictions, posterior.best, iteration,
                                              acquisition, self.settings.probability_exploitation)
        candidate = candidates[index]
        global_ids = {digest(c) for c in pool}
        metadata = {**common, **posterior.diagnostics(), **search, **selection,
                    'global_pool_size': len(global_pool), 'local_enabled': local,
                    'posterior_fit_skipped': False, 'prefilter_augmented_pool_size': prefilter_count,
                    'augmented_pool_size': len(candidates), 'local_near_rejections': rejected,
                    'pool': candidates, 'forecasts': serial(predictions), 'selected_index': index,
                    'initial_pool_mode': before['mode'], 'initial_pool_fallback': before['fallback'],
                    'model_selected_index': index,
                    'selected_from_local_augmentation': digest(candidate) not in global_ids}
        eligible = int(sum(selection['eligible']))
        metadata['model_eligible_count'] = eligible
        # Coverage is outside learned gates. Do not label a singleton as entropy search.
        if self.protect and eligible <= 1:
            coverage_index, rule = coverage_pick(self.problem, pool, records, self.seed, iteration)
            candidate = pool[coverage_index]
            metadata.update(model_mode=metadata['mode'], mode='degenerate_gate_coverage',
                            coverage_reason='eligible_at_most_one', coverage_rule=rule,
                            coverage_global_index=coverage_index,
                            selected_index=next(i for i,c in enumerate(candidates) if c == candidate),
                            selected_from_local_augmentation=False)
        return self._finish(candidate, records, metadata, prefix)
