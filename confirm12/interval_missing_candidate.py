"""Unqualified interval-output ablation using the existing record transform."""
from ra_lsbo.partial_bo.records import DesignRecord, without_intervals


def mask_training_records(records):
    """Copy training records, retaining raw history; never wire into a runner."""
    records = tuple(records)
    if not all(isinstance(record, DesignRecord) for record in records):
        raise TypeError('expected DesignRecord inputs')
    return tuple(without_intervals(record) for record in records)
