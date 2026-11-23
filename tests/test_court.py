import numpy as np
import pandas as pd

from src import evaluate_court as experiment
from src.court import METHODS, evidence_inputs, fit_court
from src.data import FEATURES


def test_evidence_follows_vote_direction():
    p = np.array([[0.8, 0.3, 0.6, 0.2]])
    context = np.array([[900, 500, 7, 1, -24]])
    reasons = np.full((1, 4, 5), 0.1)
    blue = evidence_inputs(p, context, reasons, 0)[0]
    red = evidence_inputs(p, context, reasons, 1)[0]
    assert np.allclose(blue[:3], [0.3, 1, 1])
    assert np.allclose(red[:3], [0.2, -1, 1])
    assert np.array_equal(blue[3:], -red[3:])


def test_judge_can_overrule_the_sides_and_weights_are_probabilities():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.1, 0.9, (100, 4))
    y = np.tile([0, 1], 50)
    context = np.column_stack(((2 * y - 1) * 1000, rng.normal(size=(100, 4))))
    court = fit_court(p, context, np.zeros((100, 4, 5)), y)
    test_p = np.array([[0.49, 0.49, 0.49, 0.49], [0.8, 0.2, 0.5, 0.1]])
    results, trace = court.evaluate(test_p, context[[1, 0]], np.zeros((2, 4, 5)))
    assert set(results) == set(METHODS)
    assert all(np.all((0 <= v) & (v <= 1)) for v in results.values())
    assert np.allclose(trace["gating_weights"].sum(axis=1), 1)
    assert np.isclose(court.weights.sum(), 1)
    assert trace["blue_evidence"][0] == 0
    assert results["evidence_only"][0] < 0.5 < results["judge_context"][0]
    assert np.allclose(results["judge_stacking"], (results["evidence_only"] + results["stacking"]) / 2)


def test_selection_ignores_audit_and_historical_test_labels(tmp_path, monkeypatch):
    dates = list(pd.date_range("2026-01-05", periods=48)) + [pd.Timestamp("2026-09-01")]
    base = pd.DataFrame({"date": dates, "game_id": [str(i) for i in range(len(dates))], "league": "LCK",
                         "team_name": "Blue", "opponent_team_name": "Red", "result": np.arange(len(dates)) % 2})
    for feature in FEATURES:
        base[feature] = np.arange(len(dates)) + 1
    original = base.copy()
    monkeypatch.setattr(experiment, "load_matches", lambda _: (base.copy(), base.copy(), {}))
    monkeypatch.setattr(experiment, "feature_rows", lambda rows, _: [
        {"base": row, "elo": {"elo_diff": 0}} for row in rows.to_dict("records")
    ])

    def fake_experts(data, records, labels, train, valid, legacy):
        assert data.iloc[train].date.max() < data.iloc[valid].date.min()
        assert set(data.iloc[train].series_group).isdisjoint(data.iloc[valid].series_group)
        assert len(set(train) & set(valid)) == 0
        rng = np.random.default_rng(int(valid[0]))
        frame = pd.DataFrame(rng.uniform(0.2, 0.8, (len(valid), 5)),
                             index=valid, columns=[*experiment.EXPERTS, experiment.REFERENCE])
        return frame, np.zeros((len(valid), 4, 5))

    monkeypatch.setattr(experiment, "expert_predictions", fake_experts)
    source = tmp_path / "data.csv.gz"
    source.write_bytes(b"synthetic")
    first = experiment.run(source)
    start = pd.Timestamp(first["folds"][-1]["validation"]["start"])
    base.loc[base.date >= start, "result"] = 1 - original.loc[base.date >= start, "result"]
    second = experiment.run(source)
    assert first["selected_method"] == second["selected_method"]
    assert first["selection_metrics"] == second["selection_metrics"]
    assert first["fixed_weights"] == second["fixed_weights"]
    assert first["audit_metrics"] != second["audit_metrics"]


def test_calibration_bins_include_both_ends_and_ci_for_identical_forecasts():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    summary = experiment.calibration(y, p)
    assert [b["n"] for b in summary["bins"]] == [1, 1]
    assert summary["ece_10_bins"] == 0
    results = experiment.comparison(y, p, p, pd.Series(["A", "B"]))
    assert results["accuracy_ci95"] == results["brier_ci95"] == results["log_loss_ci95"] == [0, 0]


def test_expert_explanation_measures_neutralized_inputs_without_mutation():
    inputs = pd.DataFrame({"golddiffat10": [1000], "killsat10": [4], "deathsat10": [2]})

    def predict(frame):
        return (0.5 + frame.golddiffat10 / 10000 + (frame.killsat10 - frame.deathsat10) / 100).to_numpy()

    reasons = experiment.evidence_deltas(predict, inputs, predict(inputs))
    assert np.allclose(reasons, [[0.1, 0, 0, 0.02, 0]])
    assert inputs.iloc[0].tolist() == [1000, 4, 2]
