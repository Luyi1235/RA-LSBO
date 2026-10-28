"""Replay a historical file identity from its verified content-addressed blob."""

import hashlib
from pathlib import Path

from ra_lsbo.partial_observation import LIFT_SOURCE, LIFT_SOURCE_SHA256


class HistoricalLiftSource:
    def __init__(self, content=None):
        if content is None:
            blob = (Path(__file__).resolve().parents[1]/
                    "runs/system_validation_v5_20261020T192312Z_5f6531eb/source_snapshot"/
                    (LIFT_SOURCE_SHA256+".source"))
            content = blob.read_bytes() if blob.exists() else LIFT_SOURCE.read_bytes()
            if hashlib.sha256(content).hexdigest() != LIFT_SOURCE_SHA256:
                raise ValueError("Historical lift fixture has no verified source")
        self.content = content

    def resolve(self):
        return LIFT_SOURCE.resolve()

    def read_bytes(self):
        return self.content
