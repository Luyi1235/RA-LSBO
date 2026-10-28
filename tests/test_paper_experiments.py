from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core import methods, metrics, problems, protocol, runner
from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft, candidate_pool
from ra_lsbo.risk_core.observations import OutputObservation

BASE = protocol.BASE
MATH_CASES = tuple(case for case in problems.CASES if not case.startswith("aircraft_"))


@pytest.fixture(scope="module", autouse=True)
def evaluation_accounting():
    counts = Counter()
    original = problems.MathematicalCase.evaluate

    def counted(problem, candidate):
        counts["mathematical_evaluations"] += 1
        record = original(problem, candidate)
        counts["objective_" + record.observation.objective.kind] += 1
        return record

    def forbidden(*args, **kwargs):
        pytest.fail("Physical aircraft mission calls are forbidden in this unit suite")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(problems.MathematicalCase, "evaluate", counted)
        patch.setattr(SimplifiedElectricAircraft, "evaluate", forbidden)
        yield counts
    print("\nPaper unit-test calls: " + repr(dict(sorted(counts.items()))) +
          "; physical aircraft/RCAIDE/optimization calls: 0")


@pytest.fixture
def mathematical():
    return problems.MathematicalCase(BASE)


def synthetic_record(problem, *, objective=10., margin=-.1, status="completed", **diagnostics):
    details = {"termination_reason": "unit_test_fixture", **diagnostics}
    if status == "hidden_failure":
        details["failure_cause"] = "mock_failure"
    return problem.record(problem.anchors()[0], objective, {"soc": margin}, details, status=status)


@pytest.mark.parametrize("case", MATH_CASES)
@pytest.mark.parametrize("architecture", ["basic", "augmented"])
def test_mathematical_variants_match_independent_formulas(case, architecture):
    problem = problems.make_problem(BASE, case)
    family, difficulty = case.split("_")
    candidate = deepcopy(problem.anchors()[0])
    candidate["discrete"].update(architecture=architecture, module=3)
    candidate["continuous"]["trim"] = .4
    if architecture == "augmented":
        candidate["continuous"]["x"] -= .035
    record = problem.evaluate(candidate)
    center_x, center_y = problem.settings["center"]
    if family == "branin":
        position_x, position_y = 15*center_x-5, 15*center_y
        expected = ((position_y-5.1/(4*math.pi**2)*position_x**2+
                     5/math.pi*position_x-6)**2 +
                    10*(1-1/(8*math.pi))*math.cos(position_x)+10)
        if architecture == "augmented":
            expected += 2*(.4-.25*3)**2
    else:
        coordinates = [4*center_x-2, 4*center_y-2]
        if architecture == "augmented":
            coordinates.append(4*.4-2)
        expected = sum(100*(right-left**2)**2+(1-left)**2
                       for left, right in zip(coordinates, coordinates[1:]))
    assert record.observation.objective.kind == "exact"
    assert record.observation.objective.value == pytest.approx(expected)
    assert record.observation.constraint("region").value == pytest.approx(-1.)
    expected_resource = .4+.12*3-1.05 if architecture == "augmented" else -.2
    assert record.observation.constraint("resource").value == pytest.approx(expected_resource)
    assert record.usable_feasible
    assert record.observation.design_id == digest(problem.canonical(candidate))
    assert problem.settings["difficulty"] == difficulty
    assert problem.settings["upstream_replica"] is False


@pytest.mark.parametrize("case", MATH_CASES)
def test_anchors_are_exact_usable_and_replayable(case):
    problem = problems.make_problem(BASE, case)
    anchors = problem.anchors()
    assert len(anchors) == len({digest(raw) for raw in anchors}) == 2
    assert anchors == problems.make_problem(BASE, case).anchors()
    for raw in anchors:
        record = problem.evaluate(raw)
        assert problem.canonical(raw) == raw
        assert record.usable_feasible
        assert record.observation.constraint("region").value == pytest.approx(-1.)


@pytest.mark.parametrize("case", MATH_CASES)
def test_region_abort_preserves_strict_bound_and_missing_objective(case):
    problem = problems.make_problem(BASE, case)
    raw = deepcopy(problem.anchors()[0])
    raw["continuous"].update(x=0., y=0.)
    record = problem.evaluate(raw)
    region = record.observation.constraint("region")
    assert record.observation.objective.kind == "missing"
    assert record.observation.objective.value is None
    assert record.observation.objective.reason == "region_stage_aborted"
    assert region.kind == "interval"
    assert region.lower == 0. and region.upper is None
    assert region.lower_inclusive is False
    assert region.constraint_satisfaction() is False
    assert record.observation.constraint("resource").kind == "exact"
    assert record.physical_feasible is False
    assert record.usable_feasible is False
    assert record.observation.status == "completed"


@pytest.mark.parametrize("family", ["branin", "rosenbrock"])
def test_resource_violation_keeps_exact_objective(family):
    problem = problems.MathematicalCase(BASE, family)
    raw = deepcopy(problem.anchors()[1])
    raw["discrete"]["module"] = 3
    raw["continuous"].update(x=problem.settings["center"][0]-.035, trim=1.)
    record = problem.evaluate(raw)
    assert record.observation.objective.kind == "exact"
    assert record.observation.constraint("region").constraint_satisfaction() is True
    assert record.observation.constraint("resource").value == pytest.approx(.31)
    assert record.physical_feasible is False


@pytest.mark.parametrize("family", ["branin", "rosenbrock"])
def test_inactive_mathematical_coordinates_have_no_effect(family):
    problem = problems.MathematicalCase(BASE, family)
    raw = problem.anchors()[0]
    changed = deepcopy(raw)
    changed["continuous"]["trim"] = .99
    changed["discrete"]["module"] = 3
    assert problem.space.mask(raw["discrete"]).tolist() == [1., 1., 0.]
    assert problem.space.mask(problem.anchors()[1]["discrete"]).tolist() == [1., 1., 1.]
    assert problem.evaluate(raw) == problem.evaluate(changed)


@pytest.mark.parametrize("family", ["branin", "rosenbrock"])
def test_sparse_region_is_smaller_than_ordinary(family):
    ordinary = problems.MathematicalCase(BASE, family, "ordinary")
    sparse = problems.MathematicalCase(BASE, family, "sparse")
    raw = ordinary.anchors()[0]
    raw["continuous"]["x"] += .25
    assert ordinary.evaluate(raw).usable_feasible
    assert sparse.evaluate(raw).physical_feasible is False
    assert ordinary.contract_id != sparse.contract_id


@pytest.mark.parametrize("difficulty,distance", [("ordinary", 100000.), ("sparse", 140000.)])
def test_aircraft_has_nine_variables_and_branch_masks(difficulty, distance):
    problem = problems.AircraftCase(BASE, difficulty)
    assert {variable.name for variable in problem.space.continuous} == {
        "wing_area", "aspect_ratio", "tip_diameter", "dep_diameter", "prop_spacing", "cruise_airspeed"}
    assert {variable.name for variable in problem.space.discrete} == {
        "prop_layout", "propeller_count", "parallel_strings"}
    assert len(problem.space.variables) == 9
    by_name = {variable.name: variable for variable in problem.space.variables}
    assert by_name["cruise_airspeed"].bounds == (44., 58.)
    assert by_name["parallel_strings"].values == (32, 40, 48)
    for raw in problem.anchors():
        active = dict(zip((variable.name for variable in problem.space.continuous),
                          problem.space.mask(raw["discrete"])))
        distributed = raw["discrete"]["prop_layout"] == "distributed"
        assert active == {"wing_area": 1., "aspect_ratio": 1., "cruise_airspeed": 1.,
                          "tip_diameter": float(not distributed),
                          "dep_diameter": float(distributed), "prop_spacing": float(distributed)}
        assert raw == problem.canonical(raw)
        assert raw["continuous"]["cruise_airspeed"] == 50.
        assert raw["discrete"]["parallel_strings"] == 40
    assert problem.settings["nominal_cruise_distance_m"] == distance


@pytest.mark.parametrize("missing", ["cruise_airspeed", "parallel_strings"])
def test_aircraft_requires_complete_nine_variable_schema(missing):
    problem = problems.AircraftCase(BASE)
    raw = problem.anchors()[0]
    del raw["continuous" if missing == "cruise_airspeed" else "discrete"][missing]
    with pytest.raises(ValueError):
        problem.canonical(raw)


@pytest.mark.parametrize("airspeed", [43.9, 58.1, float("nan")])
def test_aircraft_rejects_invalid_active_airspeed(airspeed):
    problem = problems.AircraftCase(BASE)
    raw = problem.anchors()[0]
    raw["continuous"]["cruise_airspeed"] = airspeed
    with pytest.raises(ValueError):
        problem.canonical(raw)


@pytest.mark.parametrize("factory,arguments", [
    (problems.MathematicalCase, {"family": "unknown"}),
    (problems.MathematicalCase, {"difficulty": "unknown"}),
    (problems.AircraftCase, {"difficulty": "unknown"}),
    (problems.AircraftCase, {"full_trajectory": 1}),
])
def test_factories_reject_unsupported_options(factory, arguments):
    with pytest.raises(ValueError):
        factory(BASE, **arguments)


def test_registry_and_candidate_pool_are_reproducible_without_evaluation():
    contracts = set()
    for case in problems.CASES:
        problem = problems.make_problem(BASE, case)
        contracts.add(problem.contract_id)
        anchors = {digest(raw) for raw in problem.anchors()}
        first = candidate_pool(problem, 2601026, 5, anchors)
        assert first == candidate_pool(problem, 2601026, 5, anchors)
        assert len({digest(raw) for raw in first[0]}) == 5
        for raw in first[0]:
            assert raw == problem.canonical(raw)
            assert digest(raw) not in anchors
            assert all(value <= 0 for value in problem.precheck(raw).values())
    assert len(contracts) == 6
    with pytest.raises(ValueError):
        problems.make_problem(BASE, "unregistered")


def test_flight_scenarios_reproduce_without_touching_global_rng():
    state_before = np.random.get_state()
    first = problems.flight_scenarios(4321, 8)
    state_after = np.random.get_state()
    assert first == problems.flight_scenarios(4321, 8)
    assert first[:3] == problems.flight_scenarios(4321, 3)
    assert first != problems.flight_scenarios(4322, 8)
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    for row in first:
        assert -4 <= row.headwind_m_s <= 8
        assert -40 <= row.payload_delta_kg <= 60
        assert .9 <= row.capacity_retention <= 1.
        assert .9 <= row.initial_soc <= .98


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_flight_scenarios_reject_invalid_counts(count):
    with pytest.raises(ValueError):
        problems.flight_scenarios(1, count)


@pytest.mark.parametrize("field,value", [
    ("headwind_m_s", -4.01), ("headwind_m_s", 8.01),
    ("payload_delta_kg", -40.01), ("payload_delta_kg", 60.01),
    ("capacity_retention", .899), ("capacity_retention", 1.001),
    ("initial_soc", .899), ("initial_soc", .981),
    ("headwind_m_s", float("nan")), ("initial_soc", float("inf")),
])
def test_scenario_support_is_enforced(field, value):
    with pytest.raises(ValueError):
        problems.FlightScenario(**{field: value})


@pytest.mark.parametrize("full_trajectory", [False, True])
def test_aircraft_scenario_transforms_and_state_isolation(monkeypatch, full_trajectory):
    scenario = problems.FlightScenario(6., 25., .93, .92)
    problem = problems.AircraftCase(BASE, "sparse", scenario, full_trajectory)
    other = problems.AircraftCase(BASE, "ordinary")
    nominal, other_nominal = deepcopy(problem.settings), deepcopy(other.settings)
    base_ah = problem.battery.CELL_AH
    snapshots = []

    def parent_evaluate(instance, raw):
        snapshots.append((deepcopy(instance.settings), instance.battery.TOTAL_CELLS,
                          instance.battery.CELL_AH))
        assert other.battery.TOTAL_CELLS == 5120
        assert other.battery.CELL_AH == base_ah
        return instance.record(raw, 123., {name: -.1 for name in instance.required_constraints},
                               {"trace": [{"soc": .15}], "termination_reason": "complete"})

    parent = Mock(side_effect=parent_evaluate)
    monkeypatch.setattr(SimplifiedElectricAircraft, "evaluate", lambda instance, raw: parent(instance, raw))
    raw = problem.anchors()[0]
    raw["continuous"]["cruise_airspeed"] = 56.
    raw["discrete"]["parallel_strings"] = 48
    first = problem.evaluate(raw)
    assert problem.evaluate(raw) == first
    assert parent.call_count == 2
    assert problem.battery is not other.battery
    for settings, cells, cell_ah in snapshots:
        assert cells == 128*48
        assert cell_ah == pytest.approx(base_ah*.93)
        assert settings["fixed_mass_kg"] == pytest.approx(1050.+25.+(128*48-5120)*.048/.75)
        assert settings["cruise_speed_m_s"] == 56.
        assert settings["cruise_distance_m"] == pytest.approx(140000.*56./50.)
        assert settings["initial_soc"] == .92
        assert settings["minimum_soc"] == (0. if full_trajectory else .2)
    assert problem.settings == nominal
    assert other.settings == other_nominal
    assert problem.battery.TOTAL_CELLS == 5120
    assert problem.battery.CELL_AH == base_ah
    assert first.diagnostics["scenario"] == asdict(scenario)
    assert first.diagnostics["total_cells"] == 6144
    assert first.diagnostics["final_soc"] == .15
    assert first.diagnostics["final_soc_exact"] is True
    assert first.observation.constraint("soc").value == pytest.approx(.05 if full_trajectory else -.1)


def test_aircraft_restores_state_when_mock_parent_raises(monkeypatch):
    problem = problems.AircraftCase(BASE, scenario=problems.FlightScenario(8., 60., .9, .9))
    nominal, base_ah = deepcopy(problem.settings), problem.battery.CELL_AH
    raw = problem.anchors()[0]
    raw["discrete"]["parallel_strings"] = 32
    parent = Mock(side_effect=RuntimeError("mock parent failure"))
    monkeypatch.setattr(SimplifiedElectricAircraft, "evaluate", parent)
    with pytest.raises(RuntimeError, match="mock parent failure"):
        problem.evaluate(raw)
    parent.assert_called_once_with(raw)
    assert problem.settings == nominal
    assert problem.battery.CELL_AH == base_ah
    assert problem.battery.TOTAL_CELLS == 5120


@pytest.mark.parametrize("trace", [[], [{"soc": .05}]])
def test_full_trajectory_censors_final_soc_on_incomplete_mission(monkeypatch, trace):
    problem = problems.AircraftCase(BASE, full_trajectory=True)

    def parent_evaluate(instance, raw):
        return instance.record(raw, OutputObservation("interval", "prefix", lower=15.),
            {"soc": OutputObservation("interval", "soc_prefix", lower=.1)},
            {"trace": trace, "failure_cause": "mock_numerical_failure"}, status="hidden_failure")

    monkeypatch.setattr(SimplifiedElectricAircraft, "evaluate", parent_evaluate)
    record = problem.evaluate(problem.anchors()[0])
    assert record.observation.objective.kind == "interval"
    assert record.observation.objective.lower == 15.
    assert record.observation.constraint("soc").kind == "missing"
    assert record.diagnostics["final_soc_exact"] is False
    assert record.diagnostics["final_soc"] == (.05 if trace else None)
    assert record.physical_feasible is None
    assert record.usable_feasible is False


@pytest.mark.parametrize("values,alpha,expected", [
    ([1., 2., 3., 4.], 0., 2.5), ([1., 2., 3., 4.], .5, 3.5),
    ([1., 2., 3., 4.], .625, 11/3), ([1., 2., 3., 4.], .95, 4.),
    ([-5., -1.], .5, -1.), ([2.], .999, 2.),
])
def test_cvar_uses_fractional_upper_tail_mass(values, alpha, expected):
    assert metrics.cvar(values, alpha) == pytest.approx(expected)
    assert metrics.cvar(list(reversed(values)), alpha) == pytest.approx(expected)


@pytest.mark.parametrize("values,alpha", [([], .95), ([float("nan")], .5),
    ([float("inf")], .5), ([1.], 1.), ([1.], -.1)])
def test_cvar_rejects_invalid_samples(values, alpha):
    with pytest.raises(ValueError):
        metrics.cvar(values, alpha)


def test_probability_scores_exclude_unknown_and_keep_boundary_bins():
    result = metrics.probability_scores([0., .1, .9, 1., .99], [False, True, False, True, None])
    assert result["n"] == 4 and result["unknown"] == 1
    assert result["brier"] == pytest.approx((.81+.81)/4)
    assert result["positive_brier"] == pytest.approx(.81/2)
    assert result["negative_brier"] == pytest.approx(.81/2)
    assert result["log_loss"] == pytest.approx(-math.log(.1)/2, abs=1e-10)
    assert result["high_confidence_errors"] == 1
    assert [row["n"] for row in result["bins"]] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert result["bins"][9]["observed"] == .5
    unknown = metrics.probability_scores([.2], [None])
    assert unknown["n"] == 0 and unknown["unknown"] == 1
    assert unknown["brier"] is None and unknown["bins"] == []


@pytest.mark.parametrize("probabilities,labels", [([.1], []), ([1.1], [False]),
    ([float("nan")], [None]), ([.1], ["unknown"])])
def test_probability_scores_validate_all_inputs(probabilities, labels):
    with pytest.raises(ValueError):
        metrics.probability_scores(probabilities, labels)


def test_binomial_interval_exact_edges():
    assert metrics.binomial_interval(0, 0) == [None, None]
    assert metrics.binomial_interval(0, 4) == pytest.approx([0., 1-.025**.25])
    assert metrics.binomial_interval(4, 4) == pytest.approx([.025**.25, 1.])
    assert metrics.binomial_interval(2, 4) == pytest.approx([.0675859865, .9324140135])


@pytest.mark.parametrize("successes,count,confidence", [(2, 1, .95), (-1, 2, .95),
    (True, 2, .95), (1, 2, 1.), (1, 2, 0.)])
def test_binomial_interval_rejects_invalid_counts(successes, count, confidence):
    with pytest.raises(ValueError):
        metrics.binomial_interval(successes, count, confidence)


def test_flight_risk_keeps_unknown_final_soc_censored(mathematical):
    rows = [synthetic_record(mathematical, objective=10., final_soc=.3, final_soc_exact=True),
            synthetic_record(mathematical, objective=20., margin=.1, final_soc=.1, final_soc_exact=True),
            synthetic_record(mathematical, objective=OutputObservation("interval", "prefix", lower=99.),
                margin=OutputObservation("missing", "soc", reason="not_complete"),
                final_soc=.01, final_soc_exact=False, status="hidden_failure")]
    result = metrics.flight_risk(rows, alpha=.5)
    assert result["soc_violations"] == result["soc_unknown"] == 1
    assert result["soc_failure_fraction_bounds"] == pytest.approx([1/3, 2/3])
    assert result["soc_shortfall_cvar_bounds"] == pytest.approx([1/15, 1/6])
    assert result["soc_failure_probability_95_conservative"] == pytest.approx([
        metrics.binomial_interval(1, 3)[0], metrics.binomial_interval(2, 3)[1]])
    assert result["completed_mission_count"] == 2
    assert result["completed_only_mean_energy_Wh"] == 15.
    assert result["usable_fraction"] == pytest.approx(1/3)
    assert result["hidden_failures"] == 1


def test_flight_risk_all_unknown_and_empty(mathematical):
    row = synthetic_record(mathematical, objective=OutputObservation("missing", "f", reason="aborted"),
        margin=OutputObservation("missing", "g", reason="aborted"), final_soc=-100., final_soc_exact=False)
    result = metrics.flight_risk([row])
    assert result["soc_failure_fraction_bounds"] == [0., 1.]
    assert result["soc_failure_probability_95_conservative"] == [0., 1.]
    assert result["soc_shortfall_cvar_bounds"] == pytest.approx([0., .2])
    assert result["completed_only_mean_energy_Wh"] is None
    assert metrics.flight_risk([]) == {"status": "no_nominal_feasible_incumbent", "n": 0}


def test_trajectory_stops_without_inventing_feasible_values(mathematical):
    failed = synthetic_record(mathematical, objective=OutputObservation("missing", "f", reason="abort"),
        margin=OutputObservation("missing", "g", reason="abort"), status="hidden_failure", numerical_failure=True)
    feasible = synthetic_record(mathematical, objective=7.)
    result = metrics.trajectory_metrics([failed], [failed, feasible], 4, stopped=True)
    assert result["initial_kind"] == "cold"
    assert result["first_feasible_total_call"] == 3
    assert result["first_feasible_online_call"] == 2
    assert [row["best_feasible"] for row in result["curve"]] == [None, None, 7., 7., 7.]
    assert [row["actual_calls"] for row in result["curve"]] == [1, 2, 3, 3, 3]
    assert result["hidden_failures"] == result["numerical_failures"] == 1
    assert result["stopped"] is True
    censored = metrics.trajectory_metrics([], [failed], 2)
    assert censored["first_feasible_censored"] is True
    assert censored["best_feasible"] is None
    warm = metrics.trajectory_metrics([feasible], [], 1)
    assert warm["first_feasible_online_call"] == 0
    assert warm["initial_kind"] == "warm"
    with pytest.raises(ValueError):
        metrics.trajectory_metrics([], [failed, feasible], 1)


@pytest.mark.parametrize("method,expected", [
    ("ra_lsbo", (True, True, True)), ("onehot_local", (False, True, True)),
    ("no_intervals", (True, False, True)), ("no_local", (True, True, False)),
    ("conventional_eic", (False, False, False)),
])
def test_method_ablations_change_only_declared_switches(method, expected):
    settings = methods.settings_for(method, protocol.PROFILES["smoke"])
    reference = methods.settings_for("ra_lsbo", protocol.PROFILES["smoke"])
    assert (settings.latent, settings.use_intervals, settings.local) == expected
    settings_dict, reference_dict = asdict(settings), asdict(reference)
    for name in ("latent", "use_intervals", "local"):
        settings_dict.pop(name)
        reference_dict.pop(name)
    assert settings_dict == reference_dict
    assert settings.average_kernel is False
    assert settings.probability_exploitation is True


@pytest.mark.parametrize("method", ["no_intervals", "conventional_eic", "ra_lsbo"])
def test_posterior_ablation_preserves_exact_and_missing_evidence(monkeypatch, mathematical, method):
    record = mathematical.record(mathematical.anchors()[0], OutputObservation("interval", "f", lower=2.),
        {"bound": OutputObservation("interval", "g", lower=0., lower_inclusive=False),
         "exact": -.1, "missing": OutputObservation("missing", "h", reason="not_run")}, {})
    constructor = Mock(return_value=object())
    monkeypatch.setattr(methods, "RiskPosterior", constructor)
    methods.posterior_for(mathematical, BASE, [record], 42, method, protocol.PROFILES["smoke"])
    training = constructor.call_args.args[2]
    assert training[0].observation.constraint("exact") == record.observation.constraint("exact")
    assert training[0].observation.constraint("missing") == record.observation.constraint("missing")
    expected = "interval" if method == "ra_lsbo" else "missing"
    assert training[0].observation.objective.kind == expected
    assert training[0].observation.constraint("bound").kind == expected
    assert record.observation.constraint("bound").kind == "interval"


@pytest.mark.parametrize("method", ["original_lsbo", "random", "unknown"])
def test_posterior_methods_reject_separate_runners(method):
    with pytest.raises(ValueError):
        methods.settings_for(method, protocol.PROFILES["smoke"])
    with pytest.raises(ValueError):
        methods.optimizer_for(None, BASE, 1, method, protocol.PROFILES["smoke"])


@pytest.mark.parametrize("profile_name", ["smoke", "pilot", "confirm"])
def test_complete_budget_accounts_all_roles(profile_name):
    profile = protocol.PROFILES[profile_name]
    result = protocol.budget_for(profile)
    cases, seeds, arms = len(profile["cases"]), len(profile["seeds"]), len(methods.ARMS)
    aircraft = sum(case.startswith("aircraft_") for case in profile["cases"])
    assert result["qualification_anchors"] == 2*cases
    assert result["independent_panel"] == cases*profile["panel"]
    assert result["shared_initial"] == cases*seeds*sum(
        profile["initial"]-(policy == "anchor_plus_random") for policy in profile["initial_policies"])
    assert result["online"] == cases*seeds*len(profile["initial_policies"])*arms*profile["horizon"]
    assert result["held_out_flight_scenarios"] == aircraft*seeds*len(profile["initial_policies"])*arms*profile["scenarios"]
    assert result["maximum_model_calls"] == sum(value for key, value in result.items()
        if key not in {"maximum_model_calls", "maximum_rcaide_mda_calls"})
    assert result["maximum_rcaide_mda_calls"] == 0
    reduced = protocol.budget_for(profile, ("ra_lsbo",))
    assert reduced["online"]*arms == result["online"]
    assert reduced["shared_initial"] == result["shared_initial"]


def minimal_plan(mathematical):
    profile = deepcopy(protocol.PROFILES["smoke"])
    return {"schema": "ra_lsbo_experiments_v2", "suite": "combined",
            "profile": profile, "arms": methods.ARMS,
            "budget": protocol.budget_for(profile),
            "sources": {str(Path(__file__).resolve()): protocol.sha(__file__)},
            "tasks": {"branin_sparse": {"contract": mathematical.contract_id}}}


def test_protocol_integrity_without_pending_document_dependency(mathematical):
    plan = minimal_plan(mathematical)
    assert protocol.verify_plan(plan) is None
    for key, replacement in [("schema", "unregistered"),
                             ("budget", {**plan["budget"], "maximum_model_calls": 0}),
                             ("sources", {str(Path(__file__).resolve()): "wrong"}),
                             ("tasks", {"branin_sparse": {"contract": "wrong"}})]:
        changed = {**plan, key: replacement}
        with pytest.raises(ValueError):
            protocol.verify_plan(changed)


@pytest.mark.parametrize("suite,expected_prefix", [
    ("mathematical", ("branin_", "rosenbrock_")),
    ("simplified_aircraft", ("aircraft_",)),
])
def test_suite_plans_are_isolated_and_within_approved_smoke_budget(suite, expected_prefix):
    plan = protocol.build_plan("smoke", suite)
    cases = plan["profile"]["cases"]
    assert len(cases) == 4 if suite == "mathematical" else len(cases) == 2
    assert all(case.startswith(expected_prefix) for case in cases)
    assert plan["suite"] == suite
    assert plan["budget"]["maximum_model_calls"] <= 500
    assert set(plan["tasks"]) == set(cases)


@pytest.fixture
def fake_ledger(monkeypatch, mathematical):
    events = []
    monkeypatch.setattr(runner, "append", lambda path, value: events.append(deepcopy(value)))
    monkeypatch.setattr(Path, "exists", lambda path: False)
    candidate = mathematical.anchors()[0]
    record = synthetic_record(mathematical)
    problem = SimpleNamespace(contract_id=mathematical.contract_id, canonical=Mock(side_effect=deepcopy),
        precheck=Mock(return_value={}), evaluate=Mock(return_value=record))
    return runner.Ledger("unit_test_in_memory_only", 1), problem, candidate, events


def test_ledger_settlement_and_duplicate_or_budget_refusal(fake_ledger):
    ledger, problem, candidate, events = fake_ledger
    record = ledger.evaluate(problem, candidate, "first", "unit")
    assert record.candidate == candidate
    assert ledger.count == 1 and ledger.pending is None
    assert ledger.roles == {"unit": 1}
    assert [event["event"] for event in events] == ["started", "settled"]
    assert events[0]["call_id"] == events[1]["call_id"]
    assert events[1]["record_digest"] == digest(record.to_dict())
    ledger.maximum = 2
    with pytest.raises(RuntimeError):
        ledger.evaluate(problem, candidate, "first", "unit")
    ledger.maximum = 1
    with pytest.raises(RuntimeError):
        ledger.evaluate(problem, candidate, "second", "unit")
    assert problem.evaluate.call_count == 1 and len(events) == 2


def test_ledger_pending_failure_is_charged_and_cannot_retry(fake_ledger):
    ledger, problem, candidate, events = fake_ledger
    problem.evaluate.side_effect = RuntimeError("mock crash")
    with pytest.raises(RuntimeError, match="mock crash"):
        ledger.evaluate(problem, candidate, "first", "unit")
    assert ledger.count == 1 and ledger.pending is not None
    assert ledger.roles == {}
    assert len(events) == 1 and events[0]["event"] == "started"
    ledger.maximum = 10
    with pytest.raises(RuntimeError, match="Unsettled"):
        ledger.evaluate(problem, candidate, "second", "unit")
    assert problem.evaluate.call_count == 1


@pytest.mark.parametrize("reason", ["noncanonical", "screen", "contract", "candidate"])
def test_ledger_rejects_invalid_input_or_mismatched_evidence(fake_ledger, reason):
    ledger, problem, candidate, events = fake_ledger
    if reason == "noncanonical":
        problem.canonical.return_value = {}
        problem.canonical.side_effect = None
    elif reason == "screen":
        problem.precheck.return_value = {"packing": .1}
    elif reason == "contract":
        problem.contract_id = "wrong_contract"
    else:
        problem.evaluate.return_value = SimpleNamespace(candidate={})
    with pytest.raises(ValueError if reason in {"noncanonical", "screen"} else RuntimeError):
        ledger.evaluate(problem, candidate, "first", "unit")
    if reason in {"noncanonical", "screen"}:
        assert ledger.count == 0 and events == []
        problem.evaluate.assert_not_called()
    else:
        assert ledger.count == 1 and ledger.pending is not None
        assert len(events) == 1


def test_ledger_refuses_existing_journal(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda path: True)
    with pytest.raises(RuntimeError, match="Existing journal"):
        runner.Ledger("unit_test_in_memory_only", 1)


