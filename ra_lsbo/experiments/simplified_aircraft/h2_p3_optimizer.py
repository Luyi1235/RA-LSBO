"""H2 comparator with a fixed effective weight and unchanged SOC scoring."""
from .h2_optimizer import CausalH2CorpusBO
from .h2_risk_controller import CausalSOCController, H2Settings


class FixedSOCController(CausalSOCController):
    def __init__(self):
        super().__init__(H2Settings())

    @property
    def weight(self):
        return self.settings.prior_onehot

    def snapshot(self):
        return {**super().snapshot(), 'P3_weight_policy': 'fixed_0.5'}


class FixedH2CorpusBO(CausalH2CorpusBO):
    def __init__(self, problem, base, seed, settings, output):
        super().__init__(problem, base, seed, settings, output)
        self.h2 = FixedSOCController()
