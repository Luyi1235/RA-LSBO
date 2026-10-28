"""Replay a historical file identity from its verified content-addressed blob."""

import hashlib
from pathlib import Path

from ra_lsbo.partial_observation import LIFT_SOURCE, LIFT_SOURCE_SHA256


class HistoricalLiftSource:
    """Prefer the committed fixture blob so a fresh clone reproduces the hash."""

    def __init__(self, content=None):
        if content is None:
            root = Path(__file__).resolve().parents[1]
            candidates = [
                root/"tests"/"fixtures"/(LIFT_SOURCE_SHA256+".source"),
                root/"runs"/"system_validation_v5_20261020T192312Z_5f6531eb"/
                "source_snapshot"/(LIFT_SOURCE_SHA256+".source"),
            ]
            blob = next((path for path in candidates if path.exists()), None)
            content = blob.read_bytes() if blob is not None else LIFT_SOURCE.read_bytes()
            if hashlib.sha256(content).hexdigest() != LIFT_SOURCE_SHA256:
                raise ValueError("Historical lift fixture has no verified source")
        self.content = content

    def resolve(self):
        return LIFT_SOURCE.resolve()

    def read_bytes(self):
        return self.content
