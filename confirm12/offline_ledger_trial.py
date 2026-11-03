"""Offline coupling of a REAL frozen H1 policy to DesignRecord ledger tickets.

No evaluate() is called here.  The caller must separately supply a
DesignRecord from a trusted source; this module cannot start an airplane run.
It is for contract and integration testing, not scientific confirmation.
"""
import json
from pathlib import Path
import sqlite3

from ra_lsbo.partial_bo.records import DesignRecord
from . import structured_record_ledger as ledger
from .offline_historical_trial import OfflineHistoricalTrial


def load_shared_initial(path, expected_plan, case, seed):
    """Use ONLY precisely 12 fully settled common initial records, never per-arm."""
    from ra_lsbo.budget import digest
    ledger._validate_plan(expected_plan)
    if seed not in expected_plan["cases"][case]["seeds"]:
        raise ValueError("Unregistered seed")
    with sqlite3.connect(str(path)) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        if meta["plan_sha"] != digest(expected_plan) or meta["scope"] != ledger.SCOPE:
            raise RuntimeError("Frozen plan identity changed")
        rows = db.execute(
            "SELECT idx,record,state,candidate FROM calls WHERE case_name=? "
            "AND seed=? AND role='initial' AND arm='' ORDER BY idx",
            (case,seed)).fetchall()
    if len(rows) != 12 or [row[0] for row in rows] != list(range(12)):
        raise RuntimeError("Exactly twelve sequential shared initial records required")
    if any(state != "settled" for _,_,state,_ in rows):
        raise RuntimeError("Unsettled initial record cannot train an optimizer")
    records = [DesignRecord.from_dict(json.loads(blob)) for _,blob,_,_ in rows]
    if any(row.candidate != json.loads(raw) for row,(_,_,_,raw) in zip(records,rows)):
        raise RuntimeError("Ledger initial observations do not match frozen candidates")
    return records


class OfflineLedgerTrial:
    """Real H1 propose and accept, fake/archived feedback supplied separately."""
    def __init__(self, db_path, frozen_plan, case, seed, historical_method, output):
        allowed = frozen_plan["cases"][case]["arms_by_seed"].get(str(seed), [])
        if historical_method not in allowed:
            raise ValueError("Historical method not registered for this seed")
        self.db_path = Path(db_path)
        self.case, self.seed, self.method = case, seed, historical_method
        self.policy = OfflineHistoricalTrial(
            case,seed,historical_method,output,
            load_shared_initial(db_path,frozen_plan,case,seed))
        self.ticket = None
        self.failed = False

    def propose(self, pool):
        if self.ticket is not None or self.failed:
            raise RuntimeError("Unsettled or failed offline ledger trial")
        candidate, metadata = self.policy.propose(pool)
        iteration = len(self.policy._history) - self.policy.probe.initial
        try:
            self.ticket = ledger.start(self.db_path,self.case,self.seed,"online",
                                       self.method,iteration,candidate,self.policy.problem)
        except Exception:
            self.failed = True
            ledger.stop(self.db_path)
            raise
        return candidate, metadata

    def accept(self, record):
        if self.ticket is None or self.failed:
            raise RuntimeError("No pending offline evidence ticket")
        try:
            state = ledger.settle(self.db_path,self.ticket,record)
            if state != "settled":
                raise RuntimeError("Untrusted record settled as unknown; stop")
            self.policy.accept(record)
            self.ticket = None
        except Exception:
            self.failed = True
            ledger.stop(self.db_path)
            raise
