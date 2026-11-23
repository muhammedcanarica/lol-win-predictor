"""İlk teslimin tarihsel test döneminde iki olasılık modelini karşılaştırır."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.data import FEATURES, TARGET, clean, load_raw
from src.logistic import LOGISTIC_C_GRID, logistic_pipeline

SNAPSHOT_START = pd.Timestamp("2026-01-01")
SNAPSHOT_END = pd.Timestamp("2026-11-21 17:02:48")
TEST_START = pd.Timestamp("2026-08-25 17:19:21")
N_SPLITS = 5
MIN_GROUP_N = 100
BOOTSTRAP_SAMPLES = 2000
SEED = 42
GOLD = "golddiffat10"
MODEL_FEATURES = {"gold_only": [GOLD], "five_features": FEATURES}


def add_series_groups(data: pd.DataFrame) -> pd.DataFrame:
    names = data[["league", "team_name", "opponent_team_name"]].astype("string")
    names = names.apply(lambda column: column.str.strip()).fillna("")
    days = data["date"].dt.normalize()
    keys = []
    missing = []
    for i, (day, (league, team, opponent)) in enumerate(zip(days, names.itertuples(index=False, name=None))):
        incomplete = not (league and team and opponent)
        missing.append(incomplete)
        keys.append(("game", i) if incomplete else (league, day, *sorted((team, opponent))))
    return data.assign(series_group=keys, missing_series_identity=missing)


def purge_series(train: pd.DataFrame, later: pd.DataFrame) -> pd.DataFrame:
    return train.loc[~train["series_group"].isin(later["series_group"])]


def period(data: pd.DataFrame) -> dict:
    return {
        "games": len(data),
        "start": str(data["date"].min()),
        "end": str(data["date"].max()),
        "missing_series_identity_games": int(data["missing_series_identity"].sum()),
    }


def validation_folds(train: pd.DataFrame) -> tuple[list, list]:
    # GridSearchCV, DataFrame etiketleri yerine satır konumlarını bekler.
    train = train.reset_index(drop=True)
    dates = pd.Index(train["date"].unique()).sort_values()
    if len(dates) <= N_SPLITS:
        raise ValueError(f"Doğrulama için en az {N_SPLITS + 1} farklı eğitim zaman damgası gerekiyor.")
    folds, details = [], []
    for number, (earlier, later) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(dates), start=1):
        fit = train.loc[train["date"].isin(dates[earlier])]
        valid = train.loc[train["date"].isin(dates[later])]
        purged = purge_series(fit, valid)
        if purged[TARGET].nunique() != 2:
            raise ValueError(f"{number}. doğrulama katında eğitim için iki sonuç sınıfı gerekiyor.")
        folds.append((purged.index.to_numpy(), valid.index.to_numpy()))
        details.append({
            "fold": number, "train": period(purged), "validation": period(valid),
            "purged_train_games": len(fit) - len(purged),
        })
    return folds, details


def probability_metrics(y, probability) -> dict:
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
    }


def paired_brier_bootstrap(y, p_gold, p_five, groups) -> dict:
    # Her çekilişte seri içindeki bütün maçlar ve iki modelin aynı maçları birlikte kalır.
    codes, unique_groups = pd.factorize(groups, sort=False)
    difference = (p_gold - y) ** 2 - (p_five - y) ** 2
    group_sums = np.bincount(codes, weights=difference)
    group_sizes = np.bincount(codes)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(len(unique_groups), size=len(unique_groups))
        draws[i] = group_sums[sampled].sum() / group_sizes[sampled].sum()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "gold_only_minus_five_features": float(difference.mean()),
        "ci95": [float(low), float(high)],
        "series_groups": len(unique_groups),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "five_features_improved": bool(low > 0),
    }


def group_performance(test: pd.DataFrame, probabilities: dict) -> pd.DataFrame:
    table = test.assign(**probabilities)
    rows = []
    for dimension in ("patch", "league"):
        for label, group in table.groupby(dimension, dropna=False):
            y = group[TARGET]
            row = {
                "group_by": dimension, "group": None if pd.isna(label) else label,
                "n": len(group), "descriptive_only": len(group) < MIN_GROUP_N,
                "observed_blue_win_rate": float(y.mean()),
            }
            for model in MODEL_FEATURES:
                p = group[model]
                row[f"{model}_brier"] = float(brier_score_loss(y, p))
                row[f"{model}_mean_prediction"] = float(p.mean())
                row[f"{model}_roc_auc"] = float(roc_auc_score(y, p)) if y.nunique() == 2 else None
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_raw(raw_path: Path) -> tuple[dict, pd.DataFrame]:
    raw = load_raw(raw_path, patch_as_text=True)
    with raw_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    data, cleaning = clean(raw, keep_team_names=True)
    window = data.loc[data["date"].between(SNAPSHOT_START, SNAPSHOT_END)]
    data_in_window = add_series_groups(window)
    earlier = data_in_window.loc[data_in_window["date"] < TEST_START]
    test = data_in_window.loc[data_in_window["date"] >= TEST_START]
    if earlier.empty or test.empty:
        raise ValueError("Sabit 2026 tarih aralığında hem eğitim hem test maçı bulunmalı.")
    if not data_in_window[TARGET].isin([0, 1]).all():
        raise ValueError("Maç sonucu 0 veya 1 olmalı.")
    train = purge_series(earlier, test)
    folds, fold_details = validation_folds(train)

    y_train, y_test = train[TARGET], test[TARGET].to_numpy()
    models, probabilities = {}, {}
    for name, features in MODEL_FEATURES.items():
        search = GridSearchCV(logistic_pipeline(), LOGISTIC_C_GRID, cv=folds,
                              scoring="neg_brier_score", error_score="raise")
        search.fit(train[features], y_train)
        p = search.predict_proba(test[features])[:, 1]
        probabilities[name] = p
        models[name] = {
            "features": features,
            "best_c": search.best_params_["model__C"],
            "mean_cv_brier": float(-search.best_score_),
            "test": probability_metrics(y_test, p),
        }

    blue_rate = float(y_train.mean())
    comparison = paired_brier_bootstrap(y_test, probabilities["gold_only"],
                                        probabilities["five_features"], test["series_group"])
    metrics = {
        "snapshot": {
            "raw_sha256": digest, "window_start": str(SNAPSHOT_START), "window_end": str(SNAPSHOT_END),
            "cleaned_games_outside_window": len(data) - len(window), "historical_test": True,
        },
        "cleaning": cleaning,
        "split": {
            "test_start": str(TEST_START), "train": period(train), "test": period(test),
            "purged_train_games": len(earlier) - len(train), "folds": fold_details,
            "series_definition": "Aynı lig, takvim günü ve sırasız takım çifti. Eksik kimlikte maç ayrı sayılır.",
            "series_limitations": "Aynı gün oynanan ayrı seriler birleşebilir, gece yarısını aşan seriler ayrılabilir.",
        },
        "selection_metric": "mean_cv_brier",
        "models": models,
        "references": {
            "gold_sign_accuracy": float(((test[GOLD].to_numpy() > 0) == y_test).mean()),
            "constant_blue_probability": blue_rate,
            "constant_test": probability_metrics(y_test, np.full(len(test), blue_rate)),
        },
        "brier_comparison": comparison,
        "interpretation": (
            "Beş girdi bu tarihsel test döneminde Brier skorunu iyileştirdi."
            if comparison["five_features_improved"] else
            "Güven aralığı bu tarihsel test döneminde beş girdinin Brier skorunu iyileştirdiğini göstermedi."
        ),
    }
    return metrics, group_performance(test, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="ChainCC maç verisi (.csv.gz)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Değerlendirme raporlarının dizini")
    args = parser.parse_args()
    try:
        metrics, groups = evaluate_raw(args.raw)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Değerlendirme hatası: {error}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "esports_metrics.json"
    group_path = args.output_dir / "esports_group_performance.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    groups.to_csv(group_path, index=False, lineterminator="\n")
    print(f"{metrics_path}\n{group_path}\n{metrics['interpretation']}")


if __name__ == "__main__":
    main()
