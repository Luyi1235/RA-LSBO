"""Risk-aware simplified electric-aircraft design experiments."""

from .problems import CASES, FlightScenario, flight_scenarios, make_problem

SUITE = "simplified_aircraft"

__all__ = ["CASES", "FlightScenario", "SUITE", "flight_scenarios", "make_problem"]
