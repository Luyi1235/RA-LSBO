"""Actual historical random policy; all feedback constructed without truth calls."""
import concurrent.futures
import multiprocessing
import os
import pytest

from confirm12.offline_historical_trial import OfflineHistoricalTrial
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool


def records(problem, candidates):
    return [problem.record(candidate, 1.0, {'soc': -0.1}, {'offline_fixture': True})
            for candidate in candidates]


def test_real_random_propose_accept_and_history_isolation(monkeypatch, tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase, 'evaluate', lambda *a, **k: pytest.fail('truth called'))
    problem = problem_for('aircraft_ordinary')
    pool, _ = candidate_pool(problem, 12345, 30)
    shared = records(problem, pool[:12])
    first = OfflineHistoricalTrial('aircraft_ordinary', 41, 'random', tmp_path / 'a', shared)
    other = OfflineHistoricalTrial('aircraft_ordinary', 42, 'random', tmp_path / 'b', shared)
    before = other.history_digest
    candidate, metadata = first.propose(pool[12:])
    assert candidate in pool[12:] and metadata['method'] == 'random'
    with pytest.raises(RuntimeError, match='unsettled'):
        first.propose(pool[12:])
    first.accept(records(problem, [candidate])[0])
    assert other.history_digest == before != first.history_digest
    assert len(shared) == 12


def test_foreign_feedback_stops_trial(monkeypatch, tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase, 'evaluate', lambda *a, **k: pytest.fail('truth called'))
    problem = problem_for('aircraft_sparse')
    pool, _ = candidate_pool(problem, 123, 15)
    trial = OfflineHistoricalTrial('aircraft_sparse', 11, 'random', tmp_path / 'run', records(problem, pool[:12]))
    candidate, _ = trial.propose(pool[12:])
    foreign = next(value for value in pool[12:] if value != candidate)
    with pytest.raises(ValueError, match='pending proposal'):
        trial.accept(records(problem, [foreign])[0])
    with pytest.raises(RuntimeError, match='failed'):
        trial.propose(pool[12:])


_worker_barrier = None


def _initialize_worker(barrier):
    global _worker_barrier
    _worker_barrier = barrier


def _actual_random_worker(job):
    output, seed = job
    from ra_lsbo.experiments.core.problems import AircraftCase
    def forbidden(*args, **kwargs):
        raise AssertionError('truth call forbidden')
    AircraftCase.evaluate = forbidden
    problem = problem_for('aircraft_ordinary')
    pool, _ = candidate_pool(problem, seed + 100, 20)
    trial = OfflineHistoricalTrial('aircraft_ordinary', seed, 'random', output, records(problem, pool[:12]))
    _worker_barrier.wait(timeout=120)
    candidate, _ = trial.propose(pool[12:])
    trial.accept(records(problem, [candidate])[0])
    return os.getpid(), trial.history_digest


@pytest.mark.skipif(os.environ.get('CONFIRM12_RUN_REAL_RANDOM_SPAWN') != '1',
                    reason='explicit local resource opt-in; no physical evaluations')
def test_twelve_real_random_policy_processes(tmp_path, monkeypatch):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    ctx = multiprocessing.get_context('spawn')
    barrier = ctx.Barrier(12)
    with concurrent.futures.ProcessPoolExecutor(max_workers=12, mp_context=ctx,
            initializer=_initialize_worker, initargs=(barrier,)) as pool:
        result = list(pool.map(_actual_random_worker,
            [(tmp_path / str(i), 70000 + i) for i in range(12)]))
    assert len({pid for pid, _ in result}) == 12
    assert len({identity for _, identity in result}) == 12
