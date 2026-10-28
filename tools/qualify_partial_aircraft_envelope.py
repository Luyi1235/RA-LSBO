"""Prospective 24-call mission sensitivity supplement, total campaign <=500."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft, candidate_pool
from tools.run_partial_bo_development import (
    BASE, CallLedger, lines, read, sha, verify_sources, wilson, write_new)

RANGES_KM = (110., 125., 140.)
SUPPLEMENT_MAXIMUM, CAMPAIGN_MAXIMUM = 24, 500


class MissionEnvelopeAircraft(SimplifiedElectricAircraft):
    def __init__(self, distance_km):
        if distance_km not in RANGES_KM:
            raise ValueError("Unregistered range")
        super().__init__(BASE)
        parent_contract = self.contract_id
        self.settings = {**self.settings, "cruise_distance_m": distance_km*1000.,
                         "family": "reduced_aircraft_range_sensitivity_v1"}
        self.contract_id = digest({"parent": parent_contract, "settings": self.settings,
                                   "supplement_source": sha(__file__)})


def prepare(parent):
    main = read(parent/"plan.json")
    verify_sources(main)
    folder = parent/"mission_envelope"
    folder.mkdir(exist_ok=False)
    problem = MissionEnvelopeAircraft(RANGES_KM[0])
    panel, rejected = candidate_pool(problem, 95031, 6)
    write_new(folder/"plan.json", {
        "schema": "mission_envelope_supplement_v1", "reason": "100km random qualification 14/30, too broad for sparse claim",
        "parent_plan_sha256": sha(parent/"plan.json"), "parent_maximum": main["maximum_new_model_calls"],
        "supplement_maximum": SUPPLEMENT_MAXIMUM, "campaign_maximum": CAMPAIGN_MAXIMUM,
        "budget_formula": "476 parent upper bound + 3*(2 engineering anchors+6 paired random)=500",
        "ranges_km": RANGES_KM, "common_anchors": problem.anchors(), "common_random_panel": panel,
        "panel_seed": 95031, "exact_screen_rejections": rejected, "supplement_source_sha256": sha(__file__),
        "contracts": {str(distance): MissionEnvelopeAircraft(distance).contract_id for distance in RANGES_KM},
        "purpose": "physical task qualification only, no optimizer runs or method outcome selection",
        "selection": "report all ranges; no sparse certification from six random samples; later independent qualification needed",
        "training": False, "retune_after_results": False, "new_rcaide_mda_calls": 0})
    (folder/"source_snapshot.source").write_bytes(Path(__file__).read_bytes())


def verify(parent, plan):
    verify_sources(read(parent/"plan.json"))
    if sha(parent/"plan.json") != plan["parent_plan_sha256"] or sha(__file__) != plan["supplement_source_sha256"]:
        raise ValueError("Parent or supplement changed after registration")
    if plan["parent_maximum"]+SUPPLEMENT_MAXIMUM != CAMPAIGN_MAXIMUM:
        raise ValueError("Campaign budget mismatch")


def execute(parent):
    folder = parent/"mission_envelope"
    plan = read(folder/"plan.json")
    verify(parent, plan)
    if not (parent/"results.json").exists():
        raise RuntimeError("Settle the parent study first")
    if (folder/"calls.jsonl").exists():
        raise RuntimeError("Single pass only; no automatic retry")
    ledger, results = CallLedger(folder, SUPPLEMENT_MAXIMUM), []
    for distance in RANGES_KM:
        problem = MissionEnvelopeAircraft(distance)
        if problem.contract_id != plan["contracts"][str(distance)]:
            raise ValueError("Mission contract changed")
        records = []
        for index, candidate in enumerate(plan["common_anchors"]+plan["common_random_panel"]):
            parent_calls = sum(event["event"] == "started" for event in lines(parent/"calls.jsonl"))
            supplement_calls = sum(event["event"] == "started" for event in lines(folder/"calls.jsonl"))
            if parent_calls+supplement_calls >= CAMPAIGN_MAXIMUM:
                raise RuntimeError("Complete campaign exhausted")
            records.append(ledger.evaluate(problem, candidate, f"range_{distance}/{index}"))
        count = sum(record.usable_feasible for record in records[2:])
        results.append({"range_km": distance,
            "anchors_usable": [record.usable_feasible for record in records[:2]],
            "random_usable_feasible": count, "random_total": 6,
            "wilson95": wilson(count, 6),
            "objective_interval_count": sum(record.observation.objective.kind == "interval" for record in records),
            "termination_reasons": [record.diagnostics["termination_reason"] for record in records],
            "sparse_certified": False})
    write_new(folder/"results.json", {"ranges": results, "supplement_calls": 24,
        "campaign_calls": parent_calls+24, "new_rcaide_mda_calls": 0,
        "interpretation": "Independent physical sensitivity only; no optimization benefit tested"})
    print(results, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("parent", type=Path)
    arguments = parser.parse_args()
    (prepare if arguments.action == "prepare" else execute)(arguments.parent)


if __name__ == "__main__":
    main()
