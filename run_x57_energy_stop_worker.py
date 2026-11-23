"""Run the frozen worker with a process-local converged-segment stop hook."""

from pathlib import Path
import sys

from ra_lsbo.x57_cold_worker import dump
from ra_lsbo.x57_energy_stop import MISSION_SEGMENTS, STATUS, first_crossing, stopped_metrics


class EnergyStop(BaseException):
    """Unwind through legacy handlers that translate ordinary solver exceptions."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics


def install(module):
    original = module.install_mission_solver

    def with_stop(nexus, diagnostics):
        original(nexus, diagnostics)
        segments = list(nexus.missions.base_mission.segments.values())
        if tuple(str(segment.tag) for segment in segments) != MISSION_SEGMENTS:
            raise ValueError("The frozen discharge mission order differs from the stop policy")

        def check_after_segment(segment):
            reports = diagnostics["mission_solver_segments"]
            if reports[-1]["segment"] != str(segment.tag):
                raise ValueError("Segment ended without a matching solver diagnostic")
            if first_crossing(reports):
                raise EnergyStop(diagnostics)

        for segment in segments:
            segment.process.post_process["x57_energy_stop"] = check_after_segment

    module.install_mission_solver = with_stop


def main():
    base = Path.cwd().resolve()
    sys.path.insert(0, str(base))
    from problems.x57 import mission_solver, worker

    if Path(worker.__file__).resolve().parent != base / "problems" / "x57":
        raise RuntimeError("Unexpected physical worker import")
    install(mission_solver)
    evaluate = worker.evaluate

    def evaluate_with_stop(payload, artifact_dir):
        try:
            return evaluate(payload, artifact_dir)
        except EnergyStop as stop:
            diagnostics = stop.diagnostics
            metrics = stopped_metrics(diagnostics["mission_solver_segments"], payload["config"]["mission"])
            evidence = metrics["engineering_energy_stop"]
            metrics["battery_domain_trial_log"] = diagnostics.get("battery_domain_trial_log")
            dump(artifact_dir / "energy_stop_audit.json", evidence)
            print(f"[RCAIDE] Engineering stop after {evidence['stop_after_segment']}: "
                  f"{evidence['terminal_energy_lower_Wh']:.3f} Wh", flush=True)
            return worker.Evaluation(None, False, {}, STATUS, metrics)

    worker.evaluate = evaluate_with_stop
    worker.main()


if __name__ == "__main__":
    main()
