import unittest

from run_aircraft_risk_timing_replay import timing_feedback


SETTINGS = {"scenario": {"headwind_m_s": 4.}, "climb_height_m": 60.,
            "nominal_cruise_distance_m": 5000., "reserve_duration_s": 30.,
            "integration_step_s": 10., "soc_reserve_requirement": .2, "full_trajectory": False}


def record(elapsed, *, complete, reason=None, soc=None):
    return {
        "observation": {"status": "completed", "evidence_id": "test"},
        "candidate": {"continuous": {"cruise_airspeed": 54.}},
        "diagnostics": {
            "mission_completed": complete,
            "termination_reason": reason or ("complete" if complete else "physical_soc_reserve_abort"),
            "trace": [{"elapsed_s": elapsed, "stage": "reserve" if complete else "cruise",
                       "soc": soc if soc is not None else (.3 if complete else .19)}],
        },
    }


class TimingReplayTests(unittest.TestCase):
    def test_stop_is_bounded_by_last_integration_step(self):
        trace = record(130., complete=False)
        sign = timing_feedback(trace, SETTINGS, use_interval=False)
        narrow = timing_feedback(trace, SETTINGS, use_interval=True)
        self.assertEqual((sign.lower, sign.upper, sign.lower_inclusive), (0., None, False))
        self.assertAlmostEqual(narrow.lower, .35)
        self.assertAlmostEqual(narrow.upper, .4)

    def test_success_has_only_a_one_sided_bound(self):
        trace = record(200., complete=True)
        a = timing_feedback(trace, SETTINGS, use_interval=False)
        b = timing_feedback(trace, SETTINGS, use_interval=True)
        self.assertEqual(a, b)
        self.assertEqual((b.lower, b.upper), (None, 0.))

    def test_late_stop_remains_positive_and_bad_evidence_is_rejected(self):
        late_stop = record(200., complete=False)
        self.assertFalse(timing_feedback(late_stop, SETTINGS, use_interval=True).lower_inclusive)
        for trace in (record(190., complete=True), record(205., complete=False),
                      record(130., complete=False, soc=.21),
                      record(130., complete=False, reason="physical_discharge_domain")):
            with self.assertRaises(ValueError):
                timing_feedback(trace, SETTINGS, use_interval=True)
