"""Atomic true-aircraft ledger. Ticket before evaluate; preserve unknowns.

Campaign provenance only, NOT a scientific statement of task qualification.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord

SCHEMA = "ra_lsbo_real_aircraft_campaign_ledger_v1"
SCOPE = "REAL_AIRCRAFT_USER_INITIATED"
ROLES = ("initial", "online", "verification")
KNOWN_TERMINAL = ("settled", "unknown")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_plan(plan):
    if (plan.get("schema") != SCHEMA or plan.get("scope") != SCOPE or
            plan.get("execution_authorized") is not True or
            plan.get("phase") != "full_seven_arm_experiment" or
            plan.get("authorization_basis") != "explicit_user_ide_execute"):
        raise ValueError("Explicit user-initiated full-run plan required")
    if not isinstance(plan.get("max_calls"), int) or plan["max_calls"] < 1 or plan["max_calls"] > 151040:
        raise ValueError("Invalid total maximum")
    if (plan.get("initial") != 12 or plan.get("online") != 18 or
            plan.get("verification") != 1):
        raise ValueError("Unexpected registered call contract")
    cases = plan.get("cases")
    if not isinstance(cases, dict) or set(cases) != {"aircraft_ordinary", "aircraft_sparse"}:
        raise ValueError("Two explicit task contracts required")
    for case, spec in cases.items():
        seeds = spec.get("seeds")
        if (not isinstance(seeds, list) or len(seeds) != len(set(seeds)) or
                not all(type(s) is int and s >= 0 for s in seeds)):
            raise ValueError("Registered unique integer seed list required: " + case)
        if set(spec.get("contracts", {})) != {"20s", "5s"}:
            raise ValueError("Both physics step contracts required")
        if not all(isinstance(x, str) and len(x) >= 8 for x in spec["contracts"].values()):
            raise ValueError("Invalid physical model contract identity")
        allowed = spec.get("arms_by_seed")
        if not isinstance(allowed, dict) or set(allowed) != {str(s) for s in seeds}:
            raise ValueError("Every seed needs a frozen arm list")
        for names in allowed.values():
            if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                    or any(not isinstance(n, str) or not n for n in names)):
                raise ValueError("Invalid arm registration")
    return plan


@contextmanager
def _db(path, *, write=False):
    # sqlite BEGIN IMMEDIATE serializes the entire multi-worker budget check.
    db = sqlite3.connect(str(path), isolation_level=None, timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA synchronous=FULL")
    try:
        if write:
            db.execute("BEGIN IMMEDIATE")
        yield db
        if write:
            db.execute("COMMIT")
    except BaseException:
        if write:
            db.execute("ROLLBACK")
        raise
    finally:
        db.close()


def _event(db, call_id, kind, payload_hash):
    last = db.execute("SELECT seq,chain FROM events ORDER BY seq DESC LIMIT 1").fetchone()
    seq = (last[0] + 1) if last else 1
    prev = last[1] if last else "0"*64
    chain = digest({"seq":seq,"call_id":call_id,"kind":kind,
                    "payload":payload_hash,"prev":prev})
    db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
               (seq,call_id,kind,payload_hash,prev,chain))


def create(path, plan):
    """New write-once ledger of user-initiated real evaluator calls."""
    _validate_plan(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    with _db(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE calls (
            id TEXT PRIMARY KEY,case_name TEXT NOT NULL,seed INTEGER NOT NULL,
            role TEXT NOT NULL,arm TEXT NOT NULL,idx INTEGER NOT NULL,
            candidate TEXT NOT NULL,candidate_hash TEXT NOT NULL,contract TEXT NOT NULL,
            state TEXT NOT NULL,record TEXT,record_hash TEXT,
            UNIQUE(case_name,seed,role,arm,idx));
        CREATE TABLE events (
            seq INTEGER PRIMARY KEY,call_id TEXT NOT NULL,kind TEXT NOT NULL,
            payload TEXT NOT NULL,prev TEXT NOT NULL,chain TEXT NOT NULL);
        CREATE TRIGGER no_event_update BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT,'immutable evidence event'); END;
        CREATE TRIGGER no_event_delete BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT,'immutable evidence event'); END;
        """)
    with _db(path, write=True) as db:
        db.executemany("INSERT INTO meta VALUES (?,?)", (
            ("plan",canonical(plan)),("plan_sha",digest(plan)),
            ("state","RUNNING_REAL"),("scope",SCOPE)))
        _event(db,"campaign","created",digest(plan))


def _identity(plan, case, seed, role, arm, index):
    if case not in plan["cases"]:
        raise ValueError("Unknown case")
    spec = plan["cases"][case]
    if type(seed) is not int or seed not in spec["seeds"]:
        raise ValueError("Unregistered task seed")
    if role not in ROLES or type(index) is not int:
        raise ValueError("Unregistered call role or index")
    if role == "initial":
        if arm != "" or not 0 <= index < plan["initial"]:
            raise ValueError("Shared initialization cannot belong to an algorithm")
    else:
        if (arm not in spec["arms_by_seed"][str(seed)] or
                not 0 <= index < plan[role]):
            raise ValueError("Unregistered arm/iteration")
    return {"case":case,"seed":seed,"role":role,"arm":arm,"index":index}


def start(path, case, seed, role, arm, index, candidate, problem):
    """Reserve a global call ticket before physical evaluation."""
    with _db(path, write=True) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        plan = _validate_plan(json.loads(meta["plan"]))
        if digest(plan) != meta["plan_sha"] or meta["state"] != "RUNNING_REAL":
            raise RuntimeError("Frozen plan changed or ledger stopped")
        ident = _identity(plan,case,seed,role,arm,index)
        contract = plan["cases"][case]["contracts"]["5s" if role == "verification" else "20s"]
        if (getattr(problem,"contract_id",None) != contract or
                problem.canonical(candidate) != candidate):
            raise ValueError("Wrong problem contract or noncanonical design")
        values = list(problem.precheck(candidate).values())
        if any(not math.isfinite(float(v)) or v > 0 for v in values):
            raise ValueError("Analytically invalid candidate")
        if db.execute("SELECT count(*) FROM calls").fetchone()[0] >= plan["max_calls"]:
            raise RuntimeError("Global call cap exceeded")
        if role != "initial":
            n = db.execute("SELECT count(*) FROM calls WHERE case_name=? AND seed=? AND role='initial' AND state='settled'",
                           (case,seed)).fetchone()[0]
            if n != plan["initial"]:
                raise RuntimeError("Twelve shared initial observations must settle first")
            prior = db.execute("SELECT count(*) FROM calls WHERE case_name=? AND seed=? AND role='online' AND arm=? AND state='settled'",
                               (case,seed,arm)).fetchone()[0]
            if (role == "online" and prior != index) or (role == "verification" and prior != plan["online"]):
                raise RuntimeError("Sequential online history incomplete")
        else:
            prior = db.execute("SELECT count(*) FROM calls WHERE case_name=? AND seed=? AND role='initial' AND state='settled'",
                               (case,seed)).fetchone()[0]
            if prior != index:
                raise RuntimeError("Shared initial prefix incomplete")
        call_id = digest({"plan_sha":meta["plan_sha"],"call":ident})
        candidate_hash = digest(candidate)
        db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (call_id,case,seed,role,arm,index,canonical(candidate),
                    candidate_hash,contract,"started",None,None))
        _event(db,call_id,"started",candidate_hash)
    return call_id


def settle(path, call_id, record):
    """Validate an exact ticket-bound physical evaluator return record."""
    if not isinstance(record,DesignRecord):
        raise TypeError("A real-structure DesignRecord is required")
    payload = record.to_dict()
    with _db(path, write=True) as db:
        row = db.execute("SELECT candidate,candidate_hash,contract,state FROM calls WHERE id=?",
                         (call_id,)).fetchone()
        if row is None or row[3] != "started":
            raise RuntimeError("Unknown, repeated or already finalized call ticket")
        candidate, candidate_hash, contract, _ = row
        if (record.candidate != json.loads(candidate) or
                digest(record.candidate) != candidate_hash or
                record.observation.contract_id != contract):
            raise ValueError("Returned observation does not match ticket candidate/physics contract")
        observed_id = record.observation.design_id
        if observed_id != candidate_hash:
            raise ValueError("Observation has a foreign design ID")
        stamp = record.diagnostics.get("confirm12_real_call")
        if not isinstance(stamp, dict) or stamp.get("call_id") != call_id or (
                stamp.get("evaluator") != "AircraftCase.evaluate"):
            raise ValueError("Missing exact real-evaluator ticket receipt")
        trusted = (record.observation.status not in ("unknown","infrastructure_error") and
                   not record.diagnostics.get("numerical_failure",False))
        state = "settled" if trusted else "unknown"
        record_hash = digest(payload)
        db.execute("UPDATE calls SET state=?,record=?,record_hash=? WHERE id=? AND state='started'",
                   (state,canonical(payload),record_hash,call_id))
        _event(db,call_id,state,record_hash)
        if not trusted:
            db.execute("UPDATE meta SET value='INCOMPLETE' WHERE key='state'")
    return state


def stop(path):
    with _db(path, write=True) as db:
        db.execute("UPDATE meta SET value='INCOMPLETE' WHERE key='state' AND value='RUNNING_VALIDATION'")


def audit(path, expected_plan):
    """Read-only real call ticket and evidence chain audit."""
    _validate_plan(expected_plan)
    with _db(path) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        events = db.execute("SELECT seq,call_id,kind,payload,prev,chain FROM events ORDER BY seq").fetchall()
        rows = db.execute("SELECT id,case_name,seed,role,arm,idx,candidate,candidate_hash,contract,state,record,record_hash FROM calls").fetchall()
    errors = []
    if (meta.get("plan") != canonical(expected_plan) or meta.get("plan_sha") != digest(expected_plan)
            or meta.get("scope") != SCOPE):
        errors.append("Plan or scope mismatch")
    prev = "0"*64
    event_map = {}
    for n,(seq,cid,kind,payload,old,chain) in enumerate(events,1):
        proper = digest({"seq":seq,"call_id":cid,"kind":kind,"payload":payload,"prev":old})
        if seq != n or old != prev or chain != proper:
            errors.append("Event chain mismatch")
        prev = chain
        event_map.setdefault(cid,[]).append((kind,payload))
    if event_map.pop("campaign",None) != [("created",digest(expected_plan))]:
        errors.append("Missing campaign creation receipt")
    statuses = {}
    if len(rows) > expected_plan["max_calls"]:
        errors.append("Budget exceeded")
    for cid,case,seed,role,arm,index,raw,chash,contract,status,raw_record,rhash in rows:
        statuses[status] = statuses.get(status,0)+1
        try:
            ident = _identity(expected_plan,case,seed,role,arm,index)
            expected_id = digest({"plan_sha":digest(expected_plan),"call":ident})
            expected_contract = expected_plan["cases"][case]["contracts"]["5s" if role=="verification" else "20s"]
            if cid != expected_id or digest(json.loads(raw)) != chash or contract != expected_contract:
                errors.append("Call identity/contract mismatch")
            events_for = [("started",chash)]
            if status in ("settled","unknown"):
                evidence = DesignRecord.from_dict(json.loads(raw_record))
                if (evidence.candidate != json.loads(raw) or evidence.observation.contract_id != contract
                        or digest(evidence.to_dict()) != rhash):
                    errors.append("Record evidence mismatch")
                events_for.append((status,rhash))
            elif status != "started":
                errors.append("Illegal call state")
            if status in ("settled", "unknown"):
                receipt = evidence.diagnostics.get("confirm12_real_call")
                if not isinstance(receipt, dict) or (
                        receipt.get("call_id") != cid or
                        receipt.get("evaluator") != "AircraftCase.evaluate"):
                    errors.append("Missing real-evaluator receipt")
            if event_map.pop(cid,None) != events_for:
                errors.append("Call event mismatch")
        except (ValueError,KeyError,TypeError,AttributeError,IndexError):
            errors.append("Malformed call/evidence row")
    if event_map:
        errors.append("Orphan evidence events")
    if statuses.get("started",0) or statuses.get("unknown",0):
        errors.append("Unsettled or unknown physical evidence")
    return {"ledger_contract_passed":not errors,"errors":errors,"calls":len(rows),
            "states":statuses,"ledger_state":meta.get("state"),
            "event_chain_head":prev,"scope":SCOPE,
            "evaluator_receipts_consistent":not errors,
            "scientific_qualification_inferred":False,
            "user_initiated_experiment":True}
