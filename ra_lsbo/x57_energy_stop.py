"""Evidence contract for stopping a discharge-only X-57 mission."""

import math

from .risk_core.observations import finite

CAP_WH = 47000.
STATUS = "engineering_energy_stop"
POLICY = "first_converged_discharge_prefix_over_47000_Wh_v1"
MISSION_SEGMENTS = ("DER", "ICA", "climb_1", "climb_2", "cruise", "decent", "Downleg",
                    "Reserve_Climb", "Reserve_Cruise", "Reserve_Descent", "Baseleg")


def prefix_energy(segments):
    accepted = []
    last_soc = 1.
    for expected, segment in zip(MISSION_SEGMENTS, segments):
        buses = segment.get("electrical_summary", [])
        if segment.get("segment") != expected or segment.get("converged") is not True or len(buses) != 1:
            break
        bus = buses[0]
        energy, duration, min_power, max_power, initial_soc, final_soc = (
            bus.get(key) for key in ("terminal_energy_Wh", "duration_s", "minimum_terminal_power_W",
                                     "maximum_terminal_power_W", "initial_charge_soc", "final_charge_soc"))
        if (bus.get("bus") != "bus" or not all(finite(v) for v in
                                               (energy, duration, min_power, max_power, initial_soc, final_soc))
                or duration <= 0 or min_power < 0 or min_power > max_power or energy < 0
                or not min_power * duration / 3600 - 1e-6 <= energy <= max_power * duration / 3600 + 1e-6
                or not 0 <= final_soc <= initial_soc <= 1
                or not math.isclose(initial_soc, last_soc, rel_tol=0, abs_tol=1e-8)):
            break
        accepted.append({"segment": expected, "terminal_energy_Wh": energy, "duration_s": duration,
                         "initial_soc": initial_soc, "final_soc": final_soc})
        last_soc = final_soc
    lower = math.fsum(s["terminal_energy_Wh"] for s in accepted) if accepted else None
    return lower, accepted


def first_crossing(segments):
    _, accepted = prefix_energy(segments)
    for index, segment in enumerate(accepted[:len(MISSION_SEGMENTS) - 1], 1):
        lower = math.fsum(row["terminal_energy_Wh"] for row in accepted[:index])
        if lower > CAP_WH:
            return {"policy": POLICY, "stop_after_segment": segment["segment"],
                    "next_segment_not_started": MISSION_SEGMENTS[index],
                    "completed_segments": index, "terminal_energy_lower_Wh": lower,
                    "terminal_energy_cap_Wh": CAP_WH, "natural_completion": None,
                    "prefix": accepted[:index]}
    return None


def stopped_metrics(segments, mission):
    evidence = first_crossing(segments)
    if evidence is None or len(segments) != evidence["completed_segments"]:
        raise ValueError("Worker stop must occur immediately at the first audited crossing")
    return {"mda_evaluated": True, "usable_terminal_energy_Wh": CAP_WH,
            "mission_profile": mission, "mission_solver_segments": segments,
            "engineering_energy_stop": evidence}


def verified_stop(raw):
    metrics = raw.get("metrics") or {}
    if (raw.get("status") != STATUS or raw.get("solver_success") is not False
            or raw.get("objective") is not None or (raw.get("constraints") or {}).get("usable_energy") is not None
            or metrics.get("mda_evaluated") is not True or metrics.get("usable_terminal_energy_Wh") != CAP_WH):
        return None
    segments = metrics.get("mission_solver_segments", [])
    calculated = first_crossing(segments)
    if (calculated and len(segments) == calculated["completed_segments"]
            and calculated == metrics.get("engineering_energy_stop")):
        return calculated
    return None
