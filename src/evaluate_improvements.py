"""İlk 2026 test dönemini aday seçiminden ayıran 10. dakika deneyleri."""
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, RAW_PATH, ROOT, TARGET
from src.evaluate_esports import (
    TEST_START, add_series_groups, paired_brier_bootstrap, period, probability_metrics, purge_series, validation_folds,
)
from src.improvement_features import (
    CHAMPION_PRIOR_GAMES, ELO_K, HISTORY_DELAY, STARTING_ELO, History, feature_rows, load_matches, match_features,
)

BASELINE = "five_features"
PRIOR_PATH = RAW_PATH.with_name("chaincc-teams-2025.csv.gz")
REPORT_PATH = ROOT / "reports" / "improvements" / "metrics.json"
MODEL_PATH = ROOT / "data" / "processed" / "exploratory_model.joblib"
FAMILIES = ("elo", "draft_picks", "draft_strength", "extra_at10")
BOOTSTRAP_SAMPLES = 2000


@dataclass(frozen=True)
class Candidate:
    families: tuple = ("base",)
    pooled: bool = False
    optimize_threshold: bool = False


def inputs(row: dict, families: tuple) -> dict:
    return {key: value for family in families for key, value in row[family].items()}


def fit(candidate, records, y, fold, prior_records, prior_y):
    features = [inputs(records[i], candidate.families) for i in fold]
    labels = y[fold]
    if candidate.pooled:
        features = [inputs(row, candidate.families) for row in prior_records] + features
        labels = np.concatenate((prior_y, labels))
    model = make_pipeline(DictVectorizer(), StandardScaler(with_mean=False), LogisticRegression(C=1, max_iter=1000))
    return model.fit(features, labels)


def predict(model, candidate, records, fold):
    return model.predict_proba([inputs(records[i], candidate.families) for i in fold])[:, 1]


def best_threshold(y, p) -> float:
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    score = tpr * y.mean() + (1 - fpr) * (1 - y.mean())
    # Varsayılan eşik de adaydır. Eşit doğrulukta 0,5'e en yakın eşik seçilir.
    thresholds[0] = 1.0
    score[0] = accuracy_score(y, p >= 1.0)
    thresholds = np.append(thresholds, 0.5)
    score = np.append(score, accuracy_score(y, p >= 0.5))
    best = np.flatnonzero(np.isclose(score, score.max(), atol=1e-12, rtol=0))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def metric_table(y, p, prediction) -> dict:
    return {"accuracy": float(accuracy_score(y, prediction)), "f1": float(f1_score(y, prediction, zero_division=0)),
            **probability_metrics(y, p)}


def paired_delta(delta, groups) -> dict:
    codes, clusters = pd.factorize(groups, sort=False)
    sizes = np.bincount(codes)
    sums = np.bincount(codes, weights=delta)
    rng = np.random.default_rng(42)
    samples = rng.integers(len(clusters), size=(BOOTSTRAP_SAMPLES, len(clusters)))
    distribution = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return {"delta": float(delta.mean()), "ci95": np.percentile(distribution, [2.5, 97.5]).tolist(),
            "groups": len(clusters), "bootstrap_samples": BOOTSTRAP_SAMPLES}


def paired_accuracy(y, baseline, challenger, groups) -> dict:
    delta = (challenger == y).astype(int) - (baseline == y).astype(int)
    result = paired_delta(delta, groups)
    return {"accuracy_delta": result.pop("delta"), **result}


def development_predictions(candidate, records, y, folds, prior_records, prior_y):
    all_y, probabilities, classifications, thresholds = [], [], [], []
    for train, valid in folds:
        model = fit(candidate, records, y, train, prior_records, prior_y)
        p = predict(model, candidate, records, valid)
        threshold = (best_threshold(np.concatenate(all_y), np.concatenate(probabilities))
                     if candidate.optimize_threshold and all_y else 0.5)
        all_y.append(y[valid])
        probabilities.append(p)
        classifications.append(p >= threshold)
        thresholds.append(threshold)
    return np.concatenate(probabilities), np.concatenate(classifications), thresholds


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def tree_models(legacy) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # İlk teslimin parametreleri karşılaştırma için sabit. Aday seçiminde kullanılmaz.
    models = {"Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=2),
              "XGBoost": XGBClassifier(random_state=42, n_jobs=2, eval_metric="logloss", subsample=0.8)}
    for name, model in models.items():
        params = {key.removeprefix("model__"): value for key, value in legacy["cv"][name]["best_params"].items()}
        model.set_params(**params)
    return models


def tree_scores(data, y, train, audit, legacy) -> dict:
    scores = {}
    for name, model in tree_models(legacy).items():
        model.fit(data.iloc[train][FEATURES], y[train])
        p = model.predict_proba(data.iloc[audit][FEATURES])[:, 1]
        scores[name] = {**metric_table(y[audit], p, p >= 0.5), "threshold": 0.5}
    return scores


def run(raw: Path, prior: Path | None, *, tree_references=False) -> tuple[dict, dict]:
    matches, history, cleaning = load_matches(raw)
    grouped = add_series_groups(matches)
    old_test = grouped.loc[grouped.date >= TEST_START]
    before_test = grouped.loc[grouped.date < TEST_START]
    development = purge_series(before_test, old_test).reset_index(drop=True)
    records = feature_rows(development, history.loc[history.date < TEST_START])
    y = development[TARGET].to_numpy()
    folds, details = validation_folds(development)
    dev_folds, (fit_rows, audit_rows) = folds[:-1], folds[-1]
    # Son katın serileri, önceki katlardaki geliştirme ve model seçiminden de çıkarılır.
    audit_series = development.iloc[audit_rows].series_group
    allowed = ~development.series_group.isin(audit_series).to_numpy()
    dev_folds = [(train[allowed[train]], valid[allowed[valid]]) for train, valid in dev_folds]
    dev_indices = np.concatenate([valid for _, valid in dev_folds])
    dev_y = y[dev_indices]
    dev_series = development.iloc[dev_indices].series_group
    prior_records, prior_y, prior_info = [], np.array([], dtype=int), None
    if prior is not None:
        prior_matches, prior_history, prior_cleaning = load_matches(prior)
        if prior_history.date.max() >= development.date.min() or not set(prior_history.game_id).isdisjoint(history.game_id):
            raise ValueError("Önceki sezon 2026 ile zamansal olarak ayrık olmalı ve maç kimliği paylaşmamalı.")
        prior_records = feature_rows(prior_matches, prior_history)
        prior_y = prior_matches[TARGET].to_numpy()
        prior_info = {"sha256": file_hash(prior), "cleaning": prior_cleaning,
                      "start": str(prior_matches.date.min()), "end": str(prior_matches.date.max())}

    candidates = {BASELINE: Candidate(), **{family: Candidate(("base", family)) for family in FAMILIES},
                  "threshold": Candidate(optimize_threshold=True)}
    if prior_records:
        candidates["2025_pooled"] = Candidate(pooled=True)
    outcomes, scores = {}, {}

    def score(name, candidate):
        p, pred, thresholds = development_predictions(candidate, records, y, dev_folds, prior_records, prior_y)
        outcomes[name] = p, pred
        comparison = paired_accuracy(dev_y, outcomes[BASELINE][1], pred, dev_series)
        scores[name] = {**metric_table(dev_y, p, pred), "thresholds": thresholds, "vs_five_features": comparison}

    for name, candidate in candidates.items():
        score(name, candidate)
    gold_candidate = Candidate(("gold_only",))
    score("gold_only", gold_candidate)
    supported = [name for name in (*FAMILIES, "2025_pooled")
                 if name in scores and scores[name]["vs_five_features"]["ci95"][0] > 0]
    if len(supported) > 1:
        candidates["supported_combination"] = Candidate(("base", *(name for name in supported if name in FAMILIES)),
                                                        pooled="2025_pooled" in supported)
        score("supported_combination", candidates["supported_combination"])
    if supported and scores["threshold"]["vs_five_features"]["ci95"][0] > 0:
        best_supported = max((name for name in (*supported, "supported_combination") if name in scores),
                             key=lambda name: scores[name]["accuracy"])
        selected = candidates[best_supported]
        candidates["supported_with_threshold"] = Candidate(selected.families, selected.pooled, True)
        score("supported_with_threshold", candidates["supported_with_threshold"])

    finalist = max((name for name in candidates if name != BASELINE),
                   key=lambda name: (scores[name]["accuracy"], -scores[name]["brier"]))
    # Ek tanı. Finalist dondurulduktan sonra yalnız geliştirme katlarında ölçülür.
    for name in supported:
        candidate = candidates[name]
        score(f"{name}_with_threshold_sensitivity", Candidate(candidate.families, candidate.pooled, True))
    audit_y = y[audit_rows]
    audit_series = development.iloc[audit_rows].series_group
    audit, predictions, probabilities = {}, {}, {}
    selected_model = None
    for name in (BASELINE, finalist, "gold_only"):
        candidate = candidates.get(name, gold_candidate)
        model = fit(candidate, records, y, fit_rows, prior_records, prior_y)
        p = predict(model, candidate, records, audit_rows)
        probabilities[name] = p
        threshold = best_threshold(dev_y, outcomes[name][0]) if candidate.optimize_threshold else 0.5
        predictions[name] = p >= threshold
        audit[name] = {**metric_table(audit_y, p, predictions[name]), "threshold": threshold}
        if name == finalist:
            selected_model = {"model": model, "families": candidate.families, "threshold": threshold, "name": name,
                              "train_end": str(development.iloc[fit_rows].date.max())}

    legacy = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    historical = json.loads((ROOT / "reports/esports/esports_metrics.json").read_text(encoding="utf-8"))
    brier = paired_brier_bootstrap(audit_y, probabilities[BASELINE], probabilities[finalist], audit_series)
    report = {
        "status": "exploratory", "raw_sha256": file_hash(raw), "cleaning": cleaning, "prior_season": prior_info,
        "versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "protocol": {"elo_start": STARTING_ELO, "elo_k": ELO_K, "history_delay_days": HISTORY_DELAY.days,
                     "champion_prior_games": CHAMPION_PRIOR_GAMES, "logistic_c": 1, "bootstrap_seed": 42,
                     "primary_selection_metric": "accuracy", "prior_season_features_reset": True},
        "excluded_test_start": str(TEST_START), "excluded_test_games": len(old_test),
        "purged_at_excluded_test_boundary": len(before_test) - len(development),
        "development_folds": [{"train": period(development.iloc[t]), "validation": period(development.iloc[v]),
                               "total_removed_from_training": (
                                   details[i]["purged_train_games"] + len(folds[i][0]) - len(t)
                               )} for i, (t, v) in enumerate(dev_folds)],
        "development": scores, "supported_for_combination": supported, "finalist": finalist,
        "audit_fold": details[-1], "audit": audit,
        "audit_finalist_vs_five_features": paired_accuracy(audit_y, predictions[BASELINE], predictions[finalist], audit_series),
        "audit_accuracy_day_bootstrap": paired_accuracy(audit_y, predictions[BASELINE], predictions[finalist],
                                                         development.iloc[audit_rows].date.dt.normalize()),
        "audit_brier_baseline_minus_finalist": {"delta": brier["gold_only_minus_five_features"], "ci95": brier["ci95"]},
        "audit_tree_references": tree_scores(development, y, fit_rows, audit_rows, legacy) if tree_references else {},
        "published_references_on_excluded_test": {
            "initial_models": legacy["test"],
            "historical_logistic_comparison": {name: model["test"] for name, model in historical["models"].items()},
        },
        "promotion": False,
    }
    return report, selected_model


def predict_local(bundle: dict, stats: dict, raw: Path) -> dict:
    date = pd.Timestamp(stats["date"])
    if date <= pd.Timestamp(bundle["train_end"]):
        raise ValueError("Tahmin tarihi modelin eğitim bitişinden sonra olmalı.")
    for family, required in (("elo", ("team_name", "opponent_team_name")),
                             ("draft_picks", ("blue_picks", "red_picks")),
                             ("draft_strength", ("blue_picks", "red_picks")), ("extra_at10", ("goldat10",))):
        if family in bundle["families"] and any(key not in stats for key in required):
            raise ValueError(f"{family} için gereken girdiler: {', '.join(required)}")
    history = History()
    if set(bundle["families"]) & {"elo", "draft_strength"}:
        _, past, _ = load_matches(raw)
        known = past.loc[past.date.dt.normalize() <= date.normalize() - HISTORY_DELAY]
        for _, day in known.groupby(known.date.dt.normalize(), sort=True):
            history.update(day)
    row = inputs(match_features(stats, history), bundle["families"])
    p = float(bundle["model"].predict_proba([row])[0, 1])
    return {"model": bundle["name"], "blue_probability": p, "threshold": bundle["threshold"],
            "winner": "Blue" if p >= bundle["threshold"] else "Red", "status": "exploratory"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--prior", type=Path, default=PRIOR_PATH, help="Model eğitimine eklenecek 2025 dosyası")
    parser.add_argument("--no-prior", action="store_true")
    parser.add_argument("--tree-references", action="store_true", help="Son dilimde ilk teslimin RF ve XGBoost ayarlarını kıyasla")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--predict-json", type=Path, help="Yerel deney modeli için 10. dakika girdileri")
    args = parser.parse_args()
    if args.predict_json:
        stats = json.loads(args.predict_json.read_text(encoding="utf-8"))
        print(json.dumps(predict_local(joblib.load(args.model), stats, args.raw), ensure_ascii=False))
        return
    report, bundle = run(args.raw, None if args.no_prior else args.prior, tree_references=args.tree_references)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    joblib.dump(bundle, args.model)
    print(args.report)


if __name__ == "__main__":
    main()
