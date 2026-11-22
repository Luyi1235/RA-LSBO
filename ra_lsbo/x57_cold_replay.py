"""Compare native and adapted updates on a saved X-57 run, without workers."""

from dataclasses import asdict
import importlib
import json

from .budget import digest
from .x57_cold_campaign import ObservationAdapter, identity, update_native
from .x57_cold_feedback import normalize
from .x57_cold_worker import dump, sha256
from .x57_live import X57LiveProblem


def replay_saved(base, run_dir, output, prefix):
    if prefix < 1:
        raise ValueError("Replay prefix must be positive")
    source = run_dir / "evaluations.jsonl"
    manifest_file = run_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    rows = [json.loads(line) for line in source.read_bytes().splitlines()][:prefix]
    if len(rows) != prefix or manifest["method"] != "lsbo":
        raise ValueError("Require a sufficiently long saved native LSBO run")
    config_file = output / "replay_config.json"
    dump(config_file, manifest["problem_config"])
    live = X57LiveProblem(base, config_file, output / "unused")
    if list(live.required_constraints) != manifest["required_constraints"]:
        raise ValueError("Saved required constraints differ from replay evaluator")
    module = importlib.import_module("lsbo.optimizer")

    class NativeReplay:
        def __getattr__(self, name):
            return getattr(live.problem, name)

        def evaluate(self, candidate):
            return module.Evaluation(**self.raw)

    settings = module.Settings(**manifest["settings"])
    # Use native settings from the saved run and evaluate the next proposal from
    # the same fixed history, without trying to recreate the old RNG history.
    direct = module.Optimizer(NativeReplay(), settings, output / "direct")
    adapter = ObservationAdapter(live.problem, module.Evaluation, native_feedback=True)
    adapted = module.Optimizer(adapter, settings, output / "adapted")
    for original in rows:
        point = live.candidate_class(**live.canonical(original["candidate"]))
        checks = live.problem.precheck(point)
        raw = original["result"]
        feedback = normalize(raw, live.required_constraints, checks, "rcaide")
        row = {"candidate_id": live.space.key(point), "worker_result": raw, "feedback": feedback,
               "worker_dir": str(raw["metrics"].get("worker_dir", "archived"))}
        direct.problem.raw = raw
        direct.evaluate(point, original["phase"], original["proposal"])
        update_native(adapted, point, row, original["phase"], original["proposal"])
        for field in ("seen", "radius", "branch_stalls", "refinement_radii",
                      "refinement_stalls", "solver_failures", "resume_failure_limit"):
            if getattr(direct, field) != getattr(adapted, field):
                raise RuntimeError(f"Update replay diverged at {original['evaluation']}: {field}")
        reference, actual = (engine.observations[-1].result for engine in (direct, adapted))
        for field in ("objective", "solver_success", "constraints", "status"):
            if getattr(reference, field) != getattr(actual, field):
                raise RuntimeError(f"Observation replay diverged at {original['evaluation']}: {field}")
    phase = direct.search_phase()
    candidate, proposal = direct.propose(phase)
    replayed, result = adapted.propose(phase)
    if (phase != adapted.search_phase() or live.space.key(candidate) != live.space.key(replayed)
            or digest(proposal) != digest(result)):
        raise RuntimeError("Next proposal or proposal diagnostics differ")
    report = {"replayed_observations": prefix, "source": str(source), "source_sha256": sha256(source),
              "source_manifest_sha256": sha256(manifest_file), "settings": asdict(settings),
              "identity": identity(live), "update_state_match": True, "proposal_match": True,
              "next_candidate": candidate.to_dict(), "proposal": proposal, "worker_launches": 0}
    dump(output / "replay_report.json", report)
    return report
