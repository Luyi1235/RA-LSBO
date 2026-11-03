"""Offline, isolated propose/accept bridge for existing historical engines.

This is not a confirmatory adapter or a truth runner: it never calls evaluate.
Only records supplied by the caller are consumed, and methods retain their
historical names until the confirmatory method mapping is independently frozen.
"""
from copy import deepcopy
import math

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from .real_legacy_adapters import build_historical_engine


class OfflineHistoricalTrial:
    def __init__(self, case, seed, method, output, initial_records):
        self.problem, self.engine, self.probe = build_historical_engine(case, seed, method, output)
        self._history = []
        self._pending = None
        self._failed = False
        if len(initial_records) != self.probe.initial:
            raise ValueError('exactly 12 shared historical records required')
        for record in initial_records:
            self._check_record(record)
            if digest(record.candidate) in {digest(row.candidate) for row in self._history}:
                raise ValueError('duplicate historical candidate')
            self._history.append(deepcopy(record))

    def _check_record(self, record):
        if not isinstance(record, DesignRecord):
            raise TypeError('DesignRecord required')
        if (record.observation.contract_id != self.problem.contract_id or
                record.candidate != self.problem.canonical(record.candidate)):
            raise ValueError('candidate or contract mismatch')
        if any(not math.isfinite(value) or value > 0
               for value in self.problem.precheck(record.candidate).values()):
            raise ValueError('record failed the analytical screen')
        if record.observation.status in ('unknown', 'infrastructure_error') or record.diagnostics.get('numerical_failure'):
            raise ValueError('untrusted observation cannot enter optimizer history')

    @property
    def history_digest(self):
        return digest([record.to_dict() for record in self._history])

    def propose(self, pool):
        if self._failed or self._pending is not None:
            raise RuntimeError('trial failed or has unsettled proposal')
        iteration = len(self._history) - self.probe.initial
        if iteration >= self.probe.horizon:
            raise RuntimeError('historical horizon exhausted')
        before = self.history_digest
        try:
            candidate, metadata = self.engine.propose(deepcopy(self._history), deepcopy(pool), iteration)
            if self.history_digest != before:
                raise RuntimeError('owned history was modified')
            if self.problem.canonical(candidate) != candidate or any(
                    not math.isfinite(value) or value > 0
                    for value in self.problem.precheck(candidate).values()):
                raise ValueError('optimizer proposed noncanonical or analytically infeasible candidate')
            if digest(candidate) in {digest(row.candidate) for row in self._history}:
                raise ValueError('optimizer proposed a previously observed design')
        except Exception:
            self._failed = True
            raise
        self._pending = deepcopy(candidate)
        return deepcopy(candidate), deepcopy(metadata)

    def accept(self, record):
        if self._failed or self._pending is None:
            raise RuntimeError('no live pending proposal')
        try:
            self._check_record(record)
            if record.candidate != self._pending:
                raise ValueError('observation does not match pending proposal')
            self.engine.accept(deepcopy(record))
            self._history.append(deepcopy(record))
        except Exception:
            self._failed = True
            raise
        self._pending = None
