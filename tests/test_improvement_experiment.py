import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES
from src import evaluate_improvements as experiment
from src.evaluate_improvements import Candidate, best_threshold, fit, paired_accuracy, predict_local
from src.improvement_features import History, feature_rows, match_features


def match(date, result):
    return {**dict.fromkeys(FEATURES, 0), "date": pd.Timestamp(date), "result": result,
            "team_name": "Blue Team", "opponent_team_name": "Red Team", "goldat10": 15000,
            "blue_picks": ("Ahri",), "red_picks": ("Orianna",)}


def test_history_ignores_current_and_future_outcomes():
    matches = pd.DataFrame([match(f"2026-01-0{day}", 1) for day in range(1, 5)])
    features = feature_rows(matches, matches)
    changed = matches.copy()
    changed.loc[1:, "result"] = 0
    changed_features = feature_rows(matches, changed)
    assert features[:3] == changed_features[:3]
    assert features[0]["elo"]["elo_diff"] == features[1]["elo"]["elo_diff"] == 0
    assert features[2]["elo"]["elo_diff"] > 0
    assert features[2]["draft_strength"]["draft_strength_diff"] > 0
    assert features[3]["elo"] != changed_features[3]["elo"]
    assert features[3]["draft_strength"] != changed_features[3]["draft_strength"]
    assert feature_rows(matches.iloc[::-1], matches) == features[::-1]


def test_signed_draft_and_cold_start():
    row = match("2026-01-01", 1)
    row["blue_picks"] = ("Lux", "Ahri", "Lux", None)
    families = match_features(row, History())
    assert families["draft_picks"] == {"pick=ahri": 1, "pick=lux": 1, "pick=orianna": -1}
    assert families["draft_strength"] == {"draft_strength_diff": 0, "draft_strength_mean": 0.5}
    assert families["elo"] == {"elo_diff": 0, "elo_mean": 1500}
    assert "result" not in {key for family in families.values() for key in family}


def test_threshold_and_series_bootstrap_use_paired_classifications():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.35, 0.45, 0.49])
    threshold = best_threshold(y, p)
    assert np.array_equal(p >= threshold, y)
    report = paired_accuracy(y, p >= 0.5, p >= threshold, pd.Series(["A", "A", "B", "B"]))
    assert report["accuracy_delta"] == 0.5
    assert report["ci95"] == [0, 1]
    assert report["groups"] == 2


def test_rolling_threshold_is_frozen_before_its_validation_fold(monkeypatch):
    monkeypatch.setattr(experiment, "fit", lambda *_: None)
    p = np.array([0.2, 0.35, 0.45, 0.49, 0.1, 0.9])
    monkeypatch.setattr(experiment, "predict", lambda model, candidate, records, indices: p[indices])
    folds = [(np.array([0]), np.arange(4)), (np.arange(4), np.array([4, 5]))]
    candidate = Candidate(optimize_threshold=True)
    y = np.array([0, 0, 1, 1, 0, 1])
    results = experiment.development_predictions(candidate, [], y, folds, [], np.array([]))
    reversed_latest = experiment.development_predictions(candidate, [],  np.r_[y[:4], y[4:][::-1]], folds, [], np.array([]))
    assert results[2] == reversed_latest[2] == [0.5, 0.45]


def test_local_model_uses_only_its_requested_inputs(tmp_path):
    rows = [match("2026-01-01", result) | {"golddiffat10": result * 2000 - 1000} for result in (0, 1)]
    features = [match_features(row, History()) for row in rows]
    candidate = Candidate()
    model = fit(candidate, features, np.array([0, 1]), np.array([0, 1]), [], np.array([]))
    bundle = {"model": model, "families": candidate.families, "name": "test", "train_end": "2026-01-01", "threshold": 0.5}
    result = predict_local(bundle, rows[1] | {"date": "2026-01-02"}, tmp_path / "unused.csv.gz")
    assert 0.5 < result["blue_probability"] < 1
    assert result["winner"] == "Blue"
    with pytest.raises(ValueError, match="eğitim bitişinden sonra"):
        predict_local(bundle, rows[1], tmp_path / "unused.csv.gz")
