import pytest
from streamlit.testing.v1 import AppTest

from src.predict import load_model, predict_blue_win

# (Blue kill, Red kill, altın, XP, CS) – arayüzdeki form sırası
SCENARIOS = [(0, 0, 0, 0, 0), (5, 1, 3000, 1500, 30), (1, 6, -4000, -2500, -50), (16, 0, 7500, 6500, 160)]


def _pct(metric):
    return float(metric.value.strip("%"))


def test_app_runs_with_defaults():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["🎯 Tahmin", "📊 Performans", "ℹ️ Hakkında"]


@pytest.mark.parametrize("inputs", SCENARIOS)
def test_app_prediction_matches_model(inputs):
    at = AppTest.from_file("../app.py", default_timeout=60).run()
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
