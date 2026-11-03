from confirm12.interval_missing_candidate import mask_training_records
from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import OutputObservation, RunObservation


def test_interval_masking_retains_exact_missing_and_original():
    candidate = {'x': 1}
    run = RunObservation(digest(candidate), 'contract', 'run', 'completed',
        OutputObservation('exact', 'obj', value=3),
        (('soc', OutputObservation('interval', 'soc', lower=0.2)),
         ('power', OutputObservation('exact', 'power', value=-1)),
         ('current', OutputObservation('missing', 'current', reason='not observed'))))
    record = DesignRecord(candidate, run, {})
    masked, = mask_training_records([record])
    assert run.constraint('soc').kind == 'interval'
    assert masked.observation.constraint('soc').kind == 'missing'
    assert masked.observation.constraint('power') == run.constraint('power')
    assert masked.observation.constraint('current') == run.constraint('current')
    assert masked.observation.objective == run.objective and masked.observation.status == run.status
    assert masked.observation.design_id == run.design_id and masked.observation.contract_id == run.contract_id


def test_interval_objective_not_replaced_by_exact():
    candidate = {'x': 2}
    run = RunObservation(digest(candidate), 'c', 'r', 'completed',
        OutputObservation('interval', 'obj', upper=5), ())
    record = DesignRecord(candidate, run, {})
    assert mask_training_records([record])[0].observation.objective.kind == 'missing'
