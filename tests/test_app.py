import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.predict import load_model, predict_blue_win
from src import train

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

# (Blue kill, Red kill, altın, XP, CS) – arayüzdeki form sırası
SCENARIOS = [(0, 0, 0, 0, 0), (5, 1, 3000, 1500, 30), (1, 6, -4000, -2500, -50), (16, 0, 7500, 6500, 160)]


def _pct(metric):
    return float(metric.value.strip("%"))


def test_app_runs_with_defaults():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["🎯 Tahmin", "📊 Performans", "ℹ️ Hakkında"]


@pytest.mark.parametrize("inputs", SCENARIOS)
def test_app_prediction_matches_model(inputs):
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    for widget, value in zip(at.number_input, inputs):
        widget.set_value(value)
    at.button[0].click().run()   # formun "Tahmin et" butonu
    assert not at.exception

    blue, red = at.metric[0], at.metric[1]
    assert "Blue" in blue.label and "Red" in red.label
    k, d, g, x, c = inputs
    expected = predict_blue_win(load_model(), dict(killsat10=k, deathsat10=d, golddiffat10=g, xpdiffat10=x, csdiffat10=c))
    assert _pct(blue) == pytest.approx(round(expected * 100, 1), abs=0.051)
    assert _pct(blue) + _pct(red) == pytest.approx(100, abs=0.11)
    winner = next(m.value for m in at.markdown if "Tahmini kazanan" in m.value)
    assert ("Blue" in winner) == (expected >= 0.5)


def test_historical_comparison_reads_report_and_handles_missing_file(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_bytes(train.METRICS_PATH.read_bytes())
    monkeypatch.setattr(train, "METRICS_PATH", metrics_path)
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert any("Tarihsel değerlendirme raporu henüz mevcut değil" in item.value for item in at.caption)

    report = {
        "models": {
            "gold_only": {"test": {"brier": 0.2, "roc_auc": 0.72, "log_loss": 0.6}},
            "five_features": {"test": {"brier": 0.19, "roc_auc": 0.75, "log_loss": 0.59}},
        },
        "split": {"test": {"games": 20}},
        "brier_comparison": {"gold_only_minus_five_features": 0.01, "ci95": [-0.02, 0.03]},
        "interpretation": "Örnek tarihsel yorum",
    }
    report_path = tmp_path / "esports" / "esports_metrics.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    at.run()
    assert not at.exception
    table = at.tabs[1].dataframe[-1].value.set_index("Model")
    assert table.loc["Beş girdi"].to_dict() == {"Brier ↓": 0.19, "ROC-AUC ↑": 0.75, "Log-loss ↓": 0.59}
    assert any("[-0.02000, +0.03000]" in item.value for item in at.caption)
    assert any(item.value == report["interpretation"] for item in at.markdown)
