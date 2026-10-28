"""Single-writer, durable reservation ledger for every expensive call."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

from .audit import load_json
from .records import RiskEvaluation, finite_number


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class BudgetRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetPolicy:
    max_calls: int
    max_risk_charge: float
    max_bootstrap_calls: int
    max_discovery_calls: int
    max_boundary_calls: int
    min_development_probability: float = .8
    max_observed_nonfeasible: int = 1000000
    allow_baseline: bool = False

    def __post_init__(self):
        for name in ("max_calls", "max_bootstrap_calls", "max_discovery_calls",
                     "max_boundary_calls", "max_observed_nonfeasible"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("Budget counts must be nonnegative integers")
        if (self.max_calls < 1 or not finite_number(self.max_risk_charge)
                or self.max_risk_charge < 0 or not finite_number(self.min_development_probability)
                or not 0 <= self.min_development_probability <= 1
                or type(self.allow_baseline) is not bool):
            raise ValueError("Invalid risk budget policy")


class RiskLedger:
    """New files only; recovery is read-only, and unfinished calls are never retried."""

    def __init__(self, path, policy=None):
        self.path = Path(path)
        self.tickets, self.events = {}, []
        self.stream = None
        if policy is None:
            for line in self.path.read_bytes().splitlines():
                self._consume(load_json(line))
            if not self.events:
                raise ValueError("Empty ledger")
        else:
            if not isinstance(policy, BudgetPolicy):
                raise TypeError("BudgetPolicy required")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("x", encoding="utf-8")
            self._append("policy", {"policy": asdict(policy)})

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def summary(self):
        active = [ticket for ticket in self.tickets.values()
                  if ticket["state"] not in ("cancelled", "precheck_rejected")]
        settled = [ticket for ticket in active if ticket["state"] == "settled"]
        return {
            "allocated_calls": len(active),
            "charged_calls": sum(ticket["state"] in ("started", "settled") for ticket in active),
            "reserved_calls": sum(ticket["state"] == "reserved" for ticket in active),
            "unresolved_started_calls": sum(ticket["state"] == "started" for ticket in active),
            "risk_charge": sum(ticket["charge"] for ticket in active),
            "mode_calls": {mode: sum(ticket["mode"] == mode for ticket in active)
                           for mode in ("bootstrap", "discovery", "boundary", "development", "baseline")},
            "settled_calls": len(settled),
            "observed_nonfeasible": sum(ticket["risk"]["feasible_label"] is not True for ticket in settled),
            "known_infeasible": sum(ticket["risk"]["feasible_label"] is False for ticket in settled),
            "unknown_outcomes": sum(ticket["risk"]["feasible_label"] is None for ticket in settled),
            "precheck_rejections_after_reservation": sum(
                ticket["state"] == "precheck_rejected" for ticket in self.tickets.values()),
        }

    def assess(self, candidate, mode, evidence, source):
        if mode not in ("bootstrap", "discovery", "boundary", "development", "baseline"):
            raise ValueError("Unknown candidate mode")
        if not isinstance(source, str) or not source or not isinstance(candidate, dict):
            raise ValueError("Candidate and source required")
        key = digest(candidate)
        if any(ticket["candidate_key"] == key and ticket["state"] != "cancelled"
               for ticket in self.tickets.values()):
            return {"allowed": False, "reason": "duplicate_candidate", "charge": None}
        basis = evidence.get("basis")
        p = evidence.get("probability")
        if basis not in ("unavailable", "raw", "empirical_calibrated"):
            raise ValueError("Unknown probability basis")
        if p is not None and (not finite_number(p) or not 0 <= p <= 1):
            raise ValueError("Invalid candidate probability")
        calibrated = basis == "empirical_calibrated"
        if calibrated and (p is None or not evidence.get("model_id") or not evidence.get("calibration_id")):
            raise ValueError("Calibrated evidence requires probability and model identifiers")
        if evidence.get("model_id") and evidence.get("candidate_key") != key:
            raise ValueError("Prediction is not bound to the final candidate")
        reason = None
        if mode == "baseline" and not self.policy.allow_baseline:
            reason = "baseline_bypass_forbidden"
        elif mode in ("development", "boundary") and not calibrated:
            reason = "calibration_unavailable"
        elif mode == "development" and p < self.policy.min_development_probability:
            reason = "development_threshold"
        # Bootstrap and untrusted exploration retain a worst-case debit of one.
        charge = 1-p if calibrated and mode in ("development", "boundary") else 1.
        summary = self.summary()
        if reason is None and summary["allocated_calls"] >= self.policy.max_calls:
            reason = "call_budget"
        if reason is None and summary["risk_charge"] + charge > self.policy.max_risk_charge + 1e-12:
            reason = "risk_budget"
        if reason is None and mode in ("bootstrap", "discovery", "boundary"):
            if summary["mode_calls"][mode] >= getattr(self.policy, "max_" + mode + "_calls"):
                reason = mode + "_budget"
        if reason is None and (summary["observed_nonfeasible"] + summary["reserved_calls"]
                               + summary["unresolved_started_calls"] >= self.policy.max_observed_nonfeasible):
            reason = "observed_failure_stop"
        return {"allowed": reason is None, "reason": reason, "charge": charge}

    def reserve(self, candidate, mode, evidence, source):
        assessment = self.assess(candidate, mode, evidence, source)
        if not assessment["allowed"]:
            raise BudgetRejected(assessment["reason"])
        token = "call_" + str(len(self.events))
        self._append("reserve", {
            "token": token, "candidate": deepcopy(candidate), "candidate_key": digest(candidate),
            "mode": mode, "evidence": deepcopy(evidence), "source": source,
            "charge": assessment["charge"],
        })
        return token

    def start(self, token):
        if self.tickets[token]["state"] != "reserved":
            raise ValueError("Call cannot be started twice or after cancellation")
        self._append("start", {"token": token})

    def cancel(self, token):
        if self.tickets[token]["state"] != "reserved":
            raise ValueError("Only a never-started reservation can be cancelled")
        self._append("cancel", {"token": token})

    def settle(self, token, risk):
        RiskEvaluation(**risk)
        ticket = self.tickets[token]
        if ticket["state"] in ("settled", "precheck_rejected"):
            if ticket["risk"] == risk:
                return
            raise ValueError("Cannot replace a settled result")
        if ticket["state"] != "started":
            raise ValueError("Only started calls can be settled")
        self._append("settle", {"token": token, "risk": deepcopy(risk)})

    def _append(self, kind, payload):
        if self.stream is None:
            raise RuntimeError("Ledger is closed or read-only")
        event = {"sequence": len(self.events), "previous": self.events[-1]["hash"] if self.events else None,
                 "kind": kind, "payload": payload}
        event["hash"] = digest(event)
        # Public operations validate before this durable write. A failed write
        # stops the caller before dispatch; no automatic replay/resubmission.
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self._consume(event)

    def _consume(self, event):
        unsigned = {key: value for key, value in event.items() if key != "hash"}
        if (event["sequence"] != len(self.events) or event["hash"] != digest(unsigned)
                or event["previous"] != (self.events[-1]["hash"] if self.events else None)):
            raise ValueError("Invalid ledger hash chain or sequence")
        kind, payload = event["kind"], event["payload"]
        if not self.events:
            if kind != "policy":
                raise ValueError("Ledger must begin with a policy")
            self.policy = BudgetPolicy(**payload["policy"])
        elif kind == "reserve":
            assessment = self.assess(payload["candidate"], payload["mode"], payload["evidence"], payload["source"])
            if (payload["token"] in self.tickets or not assessment["allowed"]
                    or assessment["charge"] != payload["charge"]
                    or digest(payload["candidate"]) != payload["candidate_key"]):
                raise ValueError("Invalid reservation")
            self.tickets[payload["token"]] = dict(deepcopy(payload), state="reserved")
        elif kind in ("start", "cancel", "settle"):
            ticket = self.tickets[payload["token"]]
            if kind in ("start", "cancel"):
                if ticket["state"] != "reserved":
                    raise ValueError("Invalid reservation transition")
                ticket["state"] = "started" if kind == "start" else "cancelled"
            else:
                risk = payload["risk"]
                RiskEvaluation(**risk)
                if ticket["state"] != "started":
                    raise ValueError("Result without a unique dispatch")
                ticket["risk"] = deepcopy(risk)
                ticket["state"] = "precheck_rejected" if risk["state"] == "precheck_rejected" else "settled"
        else:
            raise ValueError("Unknown ledger event")
        self.events.append(deepcopy(event))
