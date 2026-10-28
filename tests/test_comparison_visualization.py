from ra_lsbo.experiments.comparison.visualization import make_figures, trajectory_summary


def row(method, values, flight_risk=None):
    result = {"case": "aircraft_sparse", "method": method,
              "best_feasible": values[-1],
              "curve": [{"online_budget": index, "best_feasible": value}
                        for index, value in enumerate(values)]}
    if flight_risk is not None:
        result["flight_risk"] = flight_risk
    return result


def test_trajectory_counts_missing_feasible_runs():
    rows = [row("ra_lsbo", [None, 9., 8.]), row("ra_lsbo", [None, None, 10.])]
    budgets, median, fraction, counts = trajectory_summary(rows)
    assert budgets.tolist() == [0, 1, 2]
    assert fraction.tolist() == [0., .5, 1.]
    assert counts.tolist() == [0, 1, 2]
    assert median[1:].tolist() == [9., 9.]


def test_figures_handle_no_feasible_and_censored_risk(tmp_path):
    result = {"summaries": [
        row("ra_lsbo", [None, None], {"status": "no_nominal_feasible_incumbent", "n": 0}),
        row("random", [None, 8.], {"status": "evaluated", "n": 8,
                                   "soc_failure_fraction_bounds": [0., .5]}),
    ]}
    figures = make_figures(result, tmp_path)
    assert len(figures) == 3
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures)


def test_report_embeds_generated_figures(monkeypatch, tmp_path):
    from ra_lsbo.experiments.comparison import runner

    sample = row("ra_lsbo", [None, 8.])
    sample.update(seed=9701, has_feasible=True, explicit_violations={})
    plan = {"suite": "mathematical", "profile": {"name": "smoke"}}
    result = {"summaries": [sample], "charged_model_calls": 1, "maximum_model_calls": 1}
    monkeypatch.setattr(runner, "load_plan", lambda folder: plan)
    monkeypatch.setattr(runner, "read", lambda path: {"passed": True} if path.name == "audit.json" else result)
    # This unit test isolates figure embedding. Real audit reconstruction and
    # rejection of stale seals are covered by test_comparison_reliability.py.
    audit_calls = []
    def audit(folder, *, write=True):
        audit_calls.append((folder, write))
        return {"passed": True}
    monkeypatch.setattr(runner, "audit", audit)
    output = runner.report(tmp_path)
    assert audit_calls == [(tmp_path, False)]
    text = output.read_text(encoding="utf-8")
    assert "figures/aircraft_sparse_convergence.png" in text
    assert (tmp_path / "figures" / "aircraft_sparse_convergence.png").is_file()
