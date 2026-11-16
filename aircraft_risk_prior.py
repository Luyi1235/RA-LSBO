"""Layout-level partial pooling for the aircraft staged risk GP."""
import numpy as np

from ra_lsbo.partial_bo.models import EvidenceGP
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from ra_lsbo.risk_core.observations import OutputObservation


# Global / same-layout / local latent variance. Keep unit marginal variance.
PRIOR_VARIANCES = {
    "global": (.5, 0., .5),
    "layout": (.25, .25, .5),
}
LAYOUT_CODE = {"wingtip": 0., "distributed": 1.}


def layout_features(features, candidates):
    layout = [LAYOUT_CODE[row["discrete"]["prop_layout"]] for row in candidates]
    return np.column_stack((features, layout))


class SharedRiskEvidenceGP(EvidenceGP):
    """A random intercept shared everywhere, a layout intercept and a local GP."""

    def __init__(self, x, observations, seed=0, settings=None, *, sharing):
        if sharing not in PRIOR_VARIANCES:
            raise ValueError("Unknown risk sharing prior")
        self.sharing = sharing
        super().__init__(x, observations, seed=seed, settings=settings)

    def _kernel(self, left, right):
        shared, layout, local = PRIOR_VARIANCES[self.sharing]
        same_layout = left[:, None, -1] == right[None, :, -1]
        return (local * super()._kernel(left[:, :-1], right[:, :-1])
                + self.settings.signal_variance * (shared + layout * same_layout))

    def to_dict(self):
        return {**super().to_dict(), "schema": "aircraft-shared-risk-gp-v1", "sharing": self.sharing}

    @classmethod
    def from_dict(cls, state):
        if state.get("schema") != "aircraft-shared-risk-gp-v1":
            raise ValueError("Unsupported shared risk GP state")
        inputs = np.asarray(state["x"], dtype=float).reshape(-1, state["input_dimension"])
        return cls(inputs, [OutputObservation(**row) for row in state["observations"]],
                   seed=state["seed"], settings=MixedGPSettings(**state["settings"]),
                   sharing=state["sharing"])
