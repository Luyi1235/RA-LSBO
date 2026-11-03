"""Six-arm nonphysical research loop: genuine BO engines with synthetic/archived feedback.

This is the *real BO implementation*, never the confirmation production runner:
it intentionally has no evaluate() method. The 7th requested CBOB arm is
explicitly unavailable. For code/algorithm science only.
"""
from .offline_historical_trial import OfflineHistoricalTrial
from .real_candidate_engines import create_engine


class OfflineCandidateTrial(OfflineHistoricalTrial):
    def __init__(self, case, seed, method, output, initial_records):
        super().__init__(case, seed, method, output, initial_records,
                         engine_builder=create_engine)
