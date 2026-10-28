"""Secondary native-LSBO control: reuse its verified prefix, complete the budget."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.mechanism_system import SystemSettings
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.system_experiment import OriginalLSBOWarmStart
from tools.run_partial_bo_development import (
    BASE, CallLedger, STOP_MESSAGE, append, clean, lines, metrics, read, sha, verify_sources, write_new)
from tools.run_partial_refinement_study import HORIZON, MAXIMUM, SEEDS, TASKS, problem_for

MAXIMUM_ADDITIONAL = len(TASKS)*len(SEEDS)*(HORIZON-5)
CAMPAIGN_MAXIMUM = MAXIMUM+MAXIMUM_ADDITIONAL


def prepare(parent):
    plan = read(parent/"plan.json")
    verify_sources(plan)
    folder = parent/"lsbo_full_budget_control"
    folder.mkdir(exist_ok=False)
    write_new(folder/"plan.json", {"schema": "native_lsbo_full_budget_control_v1",
        "parent_plan_sha256": sha(parent/"plan.json"), "supplement_source_sha256": sha(__file__),
        "protocol_sha256": sha(ROOT/"docs/LSBO_BUDGET_CONTROL_PROTOCOL_20261026.md"),
        "parent_maximum": MAXIMUM, "additional_maximum": MAXIMUM_ADDITIONAL,
        "campaign_maximum": CAMPAIGN_MAXIMUM,
        "rule": "all qualified native LSBO trials; change only max_consecutive_solver_failures from5 to12",
        "reuse": "verify candidate and metadata equality before each cached-prefix update",
        "new_calls": "only unexecuted suffix; no reallocation to RA arms or changes to parent results",
        "source_policy": "secondary control registered after observing native early stop; not a primary preregistered contrast",
        "seed_selection": "all three paired seeds in every qualified task",
        "horizon": HORIZON, "seeds": SEEDS, "tasks": TASKS,
        "budget_formula": "430 parent+2 tasks*3 seeds*max(12-5,0)=472",
        "new_rcaide_mda_calls": 0})
    (folder/"source_snapshot.source").write_bytes(Path(__file__).read_bytes())


def verify(parent, plan):
    verify_sources(read(parent/"plan.json"))
    if (sha(parent/"plan.json") != plan["parent_plan_sha256"] or sha(__file__) != plan["supplement_source_sha256"]
            or sha(ROOT/"docs/LSBO_BUDGET_CONTROL_PROTOCOL_20261026.md") != plan["protocol_sha256"]
            or plan["campaign_maximum"] != CAMPAIGN_MAXIMUM or CAMPAIGN_MAXIMUM > 500):
        raise ValueError("Secondary control source or complete-plan budget changed")


def run_variant(parent, specification, proposals, packets, output, ledger=None):
    task, seed = specification["task"], specification["seed"]
    initial = [DesignRecord.from_dict(packet) for packet in specification["initial"]]
    native = OriginalLSBOWarmStart(problem_for(task), BASE,
        [record.legacy_row(index) for index, record in enumerate(initial)], seed,
        SystemSettings(calls=HORIZON, failure_cap=HORIZON, gp_steps=read(parent/"plan.json")["native_gp_steps"]), output)
    native.engine.settings = replace(native.engine.settings, max_consecutive_solver_failures=HORIZON)
    native.settings = asdict(native.engine.settings)
    history, online, saved = list(initial), [], []
    if len(proposals) != len(packets) or len(proposals) < 5:
        raise ValueError("Incomplete or unexpectedly short native prefix")
    for iteration in range(HORIZON):
        proposal, metadata = native.propose()
        candidate = proposal["candidate"]
        if iteration < len(proposals):
            previous = proposals[iteration]
            if candidate != previous["candidate"] or digest(clean(metadata)) != digest(previous["metadata"]):
                raise ValueError("Changing the stop threshold altered the inherited prefix")
            record = DesignRecord.from_dict(packets[iteration])
            reused = True
        else:
            if ledger is None:
                raise ValueError("Replay requires all settled observations; never call an oracle")
            record = ledger.evaluate(problem_for(task), candidate, f"{task}/{seed}/native_full_budget/{iteration}")
            reused = False
        if record.candidate != candidate or record.observation.contract_id != problem_for(task).contract_id:
            raise ValueError("Cached observation design or physical contract mismatch")
        entry = {"iteration": iteration, "candidate": candidate, "metadata": metadata,
                 "prefix_digest": digest([row.to_dict() for row in history]), "reused_native_prefix": reused}
        saved.append(entry)
        if ledger is not None:
            append(output/"proposals.jsonl", entry)
            append(output/"observations.jsonl", record.to_dict())
        history.append(record)
        online.append(record)
        try:
            native.accept(record.legacy_row(len(history)-1), metadata)
        except RuntimeError as error:
            if str(error) != STOP_MESSAGE or iteration != HORIZON-1:
                raise
    return {"task": task, "seed": seed, "method": "original_lsbo_full_budget",
        "settings": native.settings, **metrics(initial, online, False)}, saved, [record.to_dict() for record in online]


def execute(parent):
    folder = parent/"lsbo_full_budget_control"
    plan = read(folder/"plan.json")
    verify(parent, plan)
    result = read(parent/"results.json")
    if (folder/"results.json").exists() or (folder/"calls.jsonl").exists():
        raise RuntimeError("Single-pass secondary control; no automatic retries")
    ledger = CallLedger(folder, MAXIMUM_ADDITIONAL)
    summaries, replay_count, reused = [], 0, 0
    for manifest in sorted((parent/"optimization").glob("*/original_lsbo_*/manifest.json")):
        specification, source = read(manifest), manifest.parent
        output = folder/(specification["task"]+"_"+str(specification["seed"]))
        output.mkdir(exist_ok=False)
        proposals, packets = lines(source/"proposals.jsonl"), lines(source/"observations.jsonl")
        write_new(output/"manifest.json", {**specification,
            "primary_proposal_sha256": sha(source/"proposals.jsonl"),
            "primary_observation_sha256": sha(source/"observations.jsonl"), "primary_prefix_count": len(packets)})
        summary, saved, observed = run_variant(parent, specification, proposals, packets, output, ledger)
        replayed, restored, restored_packets = run_variant(parent, specification, saved, observed, output/"audit_replay")
        # Reuse flags describe provenance, not proposal behavior in the replay.
        for entry in restored:
            entry["reused_native_prefix"] = entry["iteration"] < len(packets)
        if (digest(clean(summary)) != digest(clean(replayed)) or digest(clean(saved)) != digest(clean(restored))
                or digest(observed) != digest(restored_packets)):
            raise ValueError("Full-budget baseline replay differs")
        replay_count += len(restored)
        reused += len(packets)
        write_new(output/"summary.json", summary)
        summaries.append(summary)
        print(summary, flush=True)
    events = lines(folder/"calls.jsonl")
    starts = [event for event in events if event["event"] == "started"]
    settled = [event for event in events if event["event"] == "settled"]
    if len(starts) != len(settled) or result["new_model_calls"]+len(starts) > CAMPAIGN_MAXIMUM:
        raise ValueError("Control settlement or complete-plan budget mismatch")
    for start, end in zip(starts, settled):
        record = DesignRecord.from_dict(end["record"])
        if (start["call_id"] != end["call_id"] or start["candidate"] != record.candidate
                or start["contract"] != record.observation.contract_id or digest(record.to_dict()) != end["record_digest"]):
            raise ValueError("Control truth/evidence linkage mismatch")
    write_new(folder/"results.json", {"summaries": summaries, "new_model_calls": len(starts),
        "campaign_new_model_calls": result["new_model_calls"]+len(starts),
        "cached_prefix_calls_reused": reused, "full_trajectory_replays": replay_count,
        "passed": True, "new_rcaide_mda_calls": 0, "secondary_control": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (prepare if args.action == "prepare" else execute)(args.parent)
