"""Reduced-aircraft cases and declared flight-scenario generator."""

from ..core.problems import AircraftCase, FlightScenario, flight_scenarios

CASES = ("aircraft_ordinary", "aircraft_sparse")


def make_problem(base, case, **kwargs):
    if case not in CASES:
        raise ValueError(f"Unknown simplified-aircraft case: {case}")
    _, difficulty = case.split("_")
    return AircraftCase(base, difficulty, **kwargs)
