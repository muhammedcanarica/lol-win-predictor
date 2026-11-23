"""İlk teslim ve espor değerlendirmesi için ortak lojistik regresyon tanımı."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGISTIC_C_GRID = {"model__C": [0.001, 0.01, 0.1, 1, 10]}


def logistic_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
