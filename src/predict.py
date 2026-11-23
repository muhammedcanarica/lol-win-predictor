"""Kaydedilmiş modeli yükleyip Blue galibiyet olasılığı üretir."""
import joblib
import pandas as pd

from src.train import MODEL_PATH


def load_model(path=MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError("Model bulunamadı. Önce `python -m src.train` çalıştırın.")
    return joblib.load(path)


def predict_blue_win(bundle: dict, stats: dict) -> float:
    """stats: FEATURES anahtarlarını içeren sözlük (Blue bakış açısı). Blue kazanma olasılığını döndürür."""
    missing = set(bundle["features"]) - set(stats)
    if missing:
        raise ValueError(f"Eksik özellik(ler): {sorted(missing)}")
    X = pd.DataFrame([{f: float(stats[f]) for f in bundle["features"]}])
    return float(bundle["pipeline"].predict_proba(X)[0, 1])


if __name__ == "__main__":
    b = load_model()
    for s in [dict(golddiffat10=0, xpdiffat10=0, csdiffat10=0, killsat10=2, deathsat10=2),
              dict(golddiffat10=3000, xpdiffat10=1500, csdiffat10=30, killsat10=5, deathsat10=1),
              dict(golddiffat10=-3000, xpdiffat10=-1500, csdiffat10=-30, killsat10=1, deathsat10=5)]:
        print(s, f"-> Blue kazanma olasılığı: {predict_blue_win(b, s):.3f}")
