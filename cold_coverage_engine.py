"""One opt-in change: inherited CEI coverage_scope none -> cold."""
from ra_lsbo.experiments.information_value_stage3.native import FantasyValueBO
from ra_lsbo.experiments.discovery_stages.policies import COVERAGE_PERIOD
from cold_coverage_core import PERIOD, TREATMENT, require


class ColdCoverageBO(FantasyValueBO):
    def __init__(self, problem, base, seed, output, *, horizon):
        require(COVERAGE_PERIOD == PERIOD, 'Frozen coverage cadence changed')
        super().__init__(problem, base, seed, output, arm='cei_myopic', horizon=horizon)
        self.coverage_scope = 'cold'

    def propose(self, records, global_pool, iteration):
        cold = not any(r.usable_feasible for r in records)
        candidate, meta = super().propose(records, global_pool, iteration)
        meta['cold_coverage_ablation'] = {'arm': TREATMENT, 'cold_before': cold,
            'scope': 'cold', 'period': PERIOD,
            'underlying_model_selector': 'cei_myopic',
            'all_real_feedback_preserved': True, 'new_failure_threshold': False}
        return candidate, meta
