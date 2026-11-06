"""Pure numeric policy tests. No physical evaluator or GP/VAE fitting."""
from copy import deepcopy
import math
import unittest
import numpy as np
from scipy.special import ndtri

from ra_lsbo.experiments.information_value_stage3.planner import (
    Outcome, FantasyFitError, choose, one_step, outcomes, probability, nodes, shortlist)
from ra_lsbo.experiments.information_value_stage3.protocol import plan, compare, summarize, ARMS


def fixture(probabilities=(.4, .55, .05, .05)):
    n = len(probabilities)
    return {"constraints": {
        "region": {"mean": -ndtri(np.array(probabilities)), "variance": np.ones(n)},
        "resource": {"mean": -np.ones(n), "variance": np.zeros(n)}},
        "objective": {"mean": np.arange(n, dtype=float), "variance": np.ones(n)}}


class NoFit:
    def __init__(self, f):
        self.f, self.calls = f, []

    def condition(self, i, outcome, *, use_abort_interval):
        self.calls.append((i, outcome, use_abort_interval))
        return deepcopy(self.f)


class Informative(NoFit):
    def condition(self, i, outcome, *, use_abort_interval):
        f = super().condition(i, outcome, use_abort_interval=use_abort_interval)
        if i == 2 and outcome.aborted and use_abort_interval:
            f["constraints"]["region"]["mean"][0] = -ndtri(.99)
        return f


class Stage3PolicyTests(unittest.TestCase):
    def test_frozen_budgets(self):
        self.assertEqual(plan("ci")["total_calls"], 20)
        self.assertEqual(plan("smoke")["total_calls"], 80)
        self.assertEqual(plan("full")["total_calls"], 576)

    def test_smoke_ci_full_seeds_disjoint(self):
        values = [set(plan(p)["seeds"]) for p in ("ci", "smoke", "full")]
        self.assertFalse(values[0] & values[1] or values[0] & values[2] or values[1] & values[2])

    def test_abort_cannot_disclose_hidden_objective(self):
        with self.assertRaises(ValueError):
            Outcome(True, None, -1., 100.)

    def test_abort_cannot_disclose_exact_region(self):
        with self.assertRaises(ValueError):
            Outcome(True, .3, -1., None)

    def test_completion_with_resource_failure_retains_objective(self):
        o = Outcome(False, -.2, .4, 11.)
        self.assertFalse(o.feasible)
        self.assertEqual(o.packet()["objective"]["value"], 11.)

    def test_probability_and_cold_myopic(self):
        f = fixture()
        np.testing.assert_allclose(one_step(f, None), [.4, .55, .05, .05])

    def test_deterministic_head(self):
        np.testing.assert_array_equal(probability({"mean": np.array([-1., 0., 1.]), "variance": np.zeros(3)}), [1, 1, 0])

    def test_warm_expected_improvement(self):
        f = fixture()
        f["constraints"]["region"] = {"mean": -np.ones(4), "variance": np.zeros(4)}
        f["objective"] = {"mean": np.array([1., 3., 7., -1.]), "variance": np.zeros(4)}
        np.testing.assert_array_equal(one_step(f, 5.), [4, 2, 0, 6])

    def test_last_step_is_exact_same_myopic_for_all_new_arms(self):
        f = fixture()
        for best in (None, 2.):
            expected = int(np.argmax(one_step(f, best)))
            for arm in ARMS[1:]:
                factory = NoFit(f)
                index, meta = choose(f, best, 1, factory, arm=arm, seed=11, iteration=5)
                self.assertEqual(index, expected)
                self.assertEqual(meta["depth"], 1)
                self.assertFalse(factory.calls)

    def test_myopic_never_uses_fantasies(self):
        f = fixture()
        factory = NoFit(f)
        choose(f, None, 6, factory, arm="cei_myopic", seed=11, iteration=0)
        self.assertFalse(factory.calls)

    def test_failure_information_changes_two_step_choice(self):
        f = fixture()
        full, _ = choose(f, None, 2, Informative(f), arm="cei_two_step", seed=10, iteration=0)
        ablated, _ = choose(f, None, 2, Informative(f), arm="cei_two_step_no_abort_update", seed=10, iteration=0)
        self.assertEqual(full, 2)
        self.assertNotEqual(ablated, 2)

    def test_no_information_negative_control(self):
        f = fixture()
        a, ra = choose(f, None, 2, NoFit(f), arm="cei_two_step", seed=10, iteration=0)
        b, rb = choose(f, None, 2, NoFit(f), arm="cei_two_step_no_abort_update", seed=10, iteration=0)
        self.assertEqual(a, b)
        self.assertEqual(ra["scores"], rb["scores"])

    def test_no_repeated_second_candidate(self):
        f = fixture()
        _, r = choose(f, 2., 2, NoFit(f), arm="cei_two_step", seed=11, iteration=0)
        for row in r["fantasies"]:
            self.assertNotEqual(row["first_index"], row["second_index"])

    def test_fantasy_weights_sum_to_one(self):
        f = fixture()
        for i in range(4):
            for warm in (False, True):
                rows = list(outcomes(f, i, nodes(13, 0), warm))
                self.assertAlmostEqual(sum(w for w, _ in rows), 1.)
                self.assertTrue(all(o.region is None and o.objective is None for _, o in rows if o.aborted))

    def test_cold_does_not_invent_objective(self):
        f = fixture()
        f["objective"] = None
        rows = list(outcomes(f, 0, nodes(13, 0), False))
        self.assertTrue(all(o.objective is None for _, o in rows))
        choose(f, None, 2, NoFit(f), arm="cei_two_step", seed=10, iteration=0)

    def test_fantasy_failure_falls_back_entirely(self):
        class Bad:
            def condition(self, *args, **kwargs):
                raise FantasyFitError("deliberate numerical test")
        f = fixture()
        index, r = choose(f, 2., 2, Bad(), arm="cei_two_step", seed=10, iteration=0)
        self.assertEqual(index, int(np.argmax(one_step(f, 2.))))
        self.assertEqual(r["fallback"], "fantasy_numerical_failure_full_myopic")

    def test_programming_error_is_not_a_numerical_fallback(self):
        class Bad:
            def condition(self, *args, **kwargs):
                raise KeyError("bug")
        with self.assertRaises(KeyError):
            choose(fixture(), 2., 2, Bad(), arm="cei_two_step", seed=10, iteration=0)

    def test_shortlist_contains_myopic_winner(self):
        f = fixture(tuple(np.linspace(.05, .95, 20)))
        base = one_step(f, None)
        self.assertIn(int(np.argmax(base)), shortlist(f, base))
        self.assertEqual(len(shortlist(f, base)), 4)

    def test_node_stream_reproducible(self):
        np.testing.assert_array_equal(nodes(22, 2), nodes(22, 2))
        self.assertFalse(np.array_equal(nodes(22, 2), nodes(22, 3)))

    def test_source_predictions_not_mutated(self):
        f = fixture()
        original = deepcopy(f)
        choose(f, 2., 2, NoFit(f), arm="cei_two_step", seed=10, iteration=0)
        np.testing.assert_array_equal(f["constraints"]["region"]["mean"], original["constraints"]["region"]["mean"])

    def test_missing_objective_not_penalty(self):
        self.assertEqual(compare(None, 200000.), -1)
        self.assertEqual(compare(None, None), 0)
        with self.assertRaises(ValueError):
            compare(float('nan'), None)

    def test_bad_profiles_and_zero_remaining_rejected(self):
        with self.assertRaises(ValueError):
            plan("guess")
        with self.assertRaises(ValueError):
            choose(fixture(), None, 0, NoFit(fixture()), arm="cei_two_step", seed=1, iteration=0)


if __name__ == "__main__":
    unittest.main()
