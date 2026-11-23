import json

import pandas as pd
import pytest

from src.data import FEATURES, chrono_split, clean, load_raw
from src.predict import load_model, predict_blue_win
from src.train import GROUP_PATH, METRICS_PATH, evaluate, group_performance, paired_bootstrap


@pytest.fixture(scope="module")
def cleaned():
    return clean(load_raw())


def test_raw_shape():
    df = load_raw()
    assert df.shape[1] == 74
    assert df["game_id"].nunique() == 9951


def test_one_row_per_game_and_no_zero_at10(cleaned):
    data, report = cleaned
    assert data["game_id"].is_unique
    assert report["dropped_all_zero_at10"] == 175
    assert not (data[FEATURES] == 0).all(axis=1).any()
    assert data[FEATURES].notna().all().all()


def test_no_leakage_columns(cleaned):
    for col in ["kills", "towers", "gamelength", "game_id", "goldat15"]:
        assert col not in FEATURES


def test_chronological_split_no_overlap(cleaned):
    data, _ = cleaned
    train, test = chrono_split(data)
    assert train["date"].max() < test["date"].min()
    assert set(train["game_id"]).isdisjoint(test["game_id"])
    assert len(train) + len(test) == len(data)


def test_evaluate_known_values():
    m = evaluate(pd.Series([1, 1, 0, 0]), pd.Series([0.9, 0.2, 0.1, 0.6]).to_numpy())
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["accuracy"] == 0.5 and m["specificity"] == 0.5


def test_model_predictions_are_sensible():
    bundle = load_model()
    even = dict(golddiffat10=0, xpdiffat10=0, csdiffat10=0, killsat10=2, deathsat10=2)
    ahead = dict(golddiffat10=3000, xpdiffat10=1500, csdiffat10=30, killsat10=5, deathsat10=1)
    behind = {k: (-v if "diff" in k else v) for k, v in ahead.items()} | {"killsat10": 1, "deathsat10": 5}
    p_even, p_ahead, p_behind = (predict_blue_win(bundle, s) for s in (even, ahead, behind))
    assert 0 <= p_behind < p_even < p_ahead <= 1
    assert p_ahead > 0.8 and p_behind < 0.2


def test_predict_rejects_missing_feature():
    with pytest.raises(ValueError):
        predict_blue_win(load_model(), {"golddiffat10": 100})


def test_metrics_file_consistent():
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert set(m["test"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert m["best_model"] == max(m["cv"], key=lambda n: m["cv"][n]["cv_roc_auc"])
    assert m["test"][m["best_model"]]["roc_auc"] > 0.7


def test_features_are_only_10_minute_stats():
    assert all(f.endswith("at10") for f in FEATURES)
    assert "series" not in FEATURES and "game_id" not in FEATURES


def test_valid_zeros_are_kept(cleaned):
    data, _ = cleaned
    assert (data["golddiffat10"] == 0).sum() > 0      # gerçekten eşit altın
    assert (data["killsat10"] == 0).sum() > 1000      # 10. dk'da kill almamak normal


def test_zero_at10_rows_match_on_both_sides():
    from src.data import AT10_COLS
    df = load_raw()
    zero = (df[AT10_COLS] == 0).all(axis=1)
    per_game = zero.groupby(df["game_id"]).sum()
    assert set(per_game.unique()) <= {0, 2}           # ya iki taraf da eksik ya hiçbiri
    assert (df.loc[zero, "kills"] + df.loc[zero, "teamdeaths"] > 0).all()  # maç oynanmış


def test_chrono_split_keeps_equal_timestamps_together():
    d = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-02"] * 4 + ["2026-01-03"] * 4)})
    train, test = chrono_split(d, test_size=0.5)       # kesim 01-02 grubunun ortasına düşer
    assert train["date"].max() < test["date"].min()
    assert len(train) == 2                              # 01-02 maçlarının tamamı teste gider


def test_timeseries_cv_folds_are_chronological(cleaned):
    from sklearn.model_selection import TimeSeriesSplit
    data, _ = cleaned
    train, _ = chrono_split(data)
    assert train["date"].is_monotonic_increasing
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(train):
        assert train["date"].iloc[tr_idx].max() <= train["date"].iloc[va_idx].min()


def test_paired_bootstrap_zero_diff_when_identical():
    import numpy as np
    y = np.array([0, 1] * 50)
    score = np.linspace(-1, 1, 100)
    p = (score + 1) / 2
    r = paired_bootstrap(y, p, (p >= 0.5).astype(int), score, n=50)
    assert abs(r["accuracy_diff"]) < 1e-12 and abs(r["roc_auc_diff"]) < 1e-12


def test_group_performance_flags_small_groups():
    t = pd.DataFrame({"patch": [16.1] * 4 + [16.2] * 2, "league": ["A"] * 6,
                      "golddiffat10": [1, -1, 1, -1, 1, -1], "result": [1, 0, 1, 0, 1, 0]})
    g = group_performance(t, pd.Series([.9, .1, .8, .2, .7, .3]).to_numpy(), min_n=4)
    assert g.set_index("group").loc["16.10", "reliable"] and not g.set_index("group").loc["16.20", "reliable"]


def test_calibration_beats_constant_reference_and_reports_exist():
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    c = m["calibration"]
    assert c["brier_model"] < c["brier_constant"] and c["logloss_model"] < c["logloss_constant"]
    assert "best_vs_sign_bootstrap" in m["baselines"]
    assert GROUP_PATH.exists()
