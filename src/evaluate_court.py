"""Mahkeme yöntemlerini zaman sıralı, keşifsel bir deneyde karşılaştırır."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.court import CONTEXT, SELECTABLE, context_row, fit_court, fit_weights
from src.data import FEATURES, RAW_PATH, ROOT, TARGET
from src.evaluate_esports import TEST_START, add_series_groups, paired_brier_bootstrap, period, purge_series, validation_folds
from src.evaluate_improvements import Candidate, file_hash, fit, inputs, metric_table, paired_accuracy, paired_delta, tree_models
from src.improvement_features import HISTORY_DELAY, feature_rows, load_matches

EXPERTS = ("five_features", "elo", "random_forest", "xgboost")
REFERENCE = "gold_only"
REPORT_PATH = ROOT / "reports" / "court" / "metrics.json"


def split_folds(data):
    folds, _ = validation_folds(data)
    disjoint = []
    future_series = set()
    for train, validation in reversed(folds):
        clean_validation = validation[~data.iloc[validation].series_group.isin(future_series)]
        future_series.update(data.iloc[validation].series_group)
        clean_train = train[~data.iloc[train].series_group.isin(future_series)]
        if not len(clean_validation) or data.iloc[clean_train][TARGET].nunique() != 2:
            raise ValueError("Seri ayrımından sonra doğrulama maçı ve eğitimde iki sonuç sınıfı gerekiyor.")
        disjoint.append((clean_train, clean_validation))
    return disjoint[::-1]


def evidence_deltas(predictor, features, original):
    """Her sütun, uzman tahminindeki tek karşılaştırma farkının sıfırlanmasına verilen yanıt."""
    deltas = np.zeros((len(features), len(CONTEXT)))
    for i, feature in enumerate(CONTEXT):
        if feature == "kill_diff":
            equal_kills = (features["killsat10"] + features["deathsat10"]) / 2
            neutral = features.assign(killsat10=equal_kills, deathsat10=equal_kills)
        elif feature in features:
            neutral = features.assign(**{feature: 0})
        else:
            continue
        deltas[:, i] = original - predictor(neutral)
    return deltas


def expert_predictions(data, records, y, train, valid, legacy):
    probabilities = {}
    explanations = {}
    for name, families in (("five_features", ("base",)), ("elo", ("base", "elo")), (REFERENCE, ("gold_only",))):
        candidate = Candidate(families)
        model = fit(candidate, records, y, train, [], np.array([], dtype=int))
        features = pd.DataFrame([inputs(records[i], families) for i in valid])

        def probability(frame):
            return model.predict_proba(frame.to_dict("records"))[:, 1]

        probabilities[name] = probability(features)
        if name in EXPERTS:
            explanations[name] = evidence_deltas(probability, features, probabilities[name])
    for name, model in tree_models(legacy).items():
        # Ağaçların tahminleri aynı sırada toplansın.
        model.set_params(n_jobs=1)
        model.fit(data.iloc[train][FEATURES], y[train])
        features = data.iloc[valid][FEATURES]

        def probability(frame):
            return model.predict_proba(frame)[:, 1]

        key = name.lower().replace(" ", "_")
        probabilities[key] = probability(features)
        explanations[key] = evidence_deltas(probability, features, probabilities[key])
    return pd.DataFrame(probabilities, index=valid), np.stack([explanations[n] for n in EXPERTS], axis=1)


def calibration(y, p):
    indices = np.minimum((p * 10).astype(int), 9)
    bins = []
    for i in range(10):
        rows = indices == i
        if rows.any():
            bins.append({"start": i / 10, "end": (i + 1) / 10, "n": int(rows.sum()),
                         "predicted": float(p[rows].mean()), "observed": float(y[rows].mean())})
    return {"mean_prediction": float(p.mean()), "blue_win_rate": float(y.mean()),
            "ece_10_bins": sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in bins) / len(y), "bins": bins}


def comparison(y, reference, candidate, groups):
    acc = paired_accuracy(y, reference >= 0.5, candidate >= 0.5, groups)
    brier = paired_brier_bootstrap(y, reference, candidate, groups)
    epsilon = np.finfo(float).eps
    correct_ref = np.clip(np.where(y, reference, 1 - reference), epsilon, 1 - epsilon)
    correct_candidate = np.clip(np.where(y, candidate, 1 - candidate), epsilon, 1 - epsilon)
    log_loss = paired_delta(np.log(correct_candidate) - np.log(correct_ref), groups)
    return {"accuracy_candidate_minus_reference": acc["accuracy_delta"], "accuracy_ci95": acc["ci95"],
            "brier_reference_minus_candidate": brier["gold_only_minus_five_features"], "brier_ci95": brier["ci95"],
            "log_loss_reference_minus_candidate": log_loss["delta"], "log_loss_ci95": log_loss["ci95"],
            "series_groups": acc["groups"], "bootstrap_samples": acc["bootstrap_samples"]}


def diversity(frame, y):
    correct = frame.ge(0.5).eq(y, axis=0)
    pairs = []
    for a, b in combinations(frame, 2):
        correlation = frame[a].corr(frame[b]) if frame[a].nunique() > 1 and frame[b].nunique() > 1 else None
        pairs.append({"experts": [a, b], "probability_pearson": correlation,
                      "vote_disagreement": float(correct[a].ne(correct[b]).mean()),
                      "both_wrong": int((~correct[a] & ~correct[b]).sum()),
                      "first_only_correct": int((correct[a] & ~correct[b]).sum()),
                      "second_only_correct": int((~correct[a] & correct[b]).sum())})
    experts = correct[list(EXPERTS)]
    return {"pairs": pairs, "unanimously_wrong": int((~experts.any(axis=1)).sum()),
            "all_experts_correct": int(experts.all(axis=1).sum()),
            "only_correct": {n: int((experts[n] & (experts.sum(axis=1) == 1)).sum()) for n in EXPERTS},
            "gold_only_rescues_when_all_four_wrong": int((correct[REFERENCE] & ~experts.any(axis=1)).sum())}


def summarize_cases(data, y, probabilities, p, trace, context, reasons):
    votes = p >= 0.5
    count = votes.sum(axis=1)
    unanimous = np.isin(count, [0, len(EXPERTS)])
    ties = count == len(EXPERTS) / 2
    elo_correct = (p[:, EXPERTS.index("elo")] >= 0.5) == y
    scenarios = {"unanimous": unanimous, "split_3_1": ~unanimous & ~ties, "tie_2_2": ties,
                 "close_gold_under_1000": np.abs(context[:, 0]) < 1000,
                 "gold_vs_elo_disagree": context[:, 0] * context[:, -1] < 0}
    samples = {"all_experts_wrong": np.flatnonzero(~(votes == y[:, None]).any(axis=1))}
    changes = {}
    for name, probability in probabilities.items():
        correct = (probability >= 0.5) == y
        gained, lost = correct & ~elo_correct, ~correct & elo_correct
        changes[name] = {"rescued_vs_elo": int(gained.sum()), "spoiled_vs_elo": int(lost.sum()),
                         "rescued_all_expert_errors": int((correct & ~(votes == y[:, None]).any(axis=1)).sum()),
                         "majority_overruled": int(((probability >= 0.5) != (count > len(EXPERTS) / 2))[~ties].sum())}
        samples[f"{name}_rescued"] = np.flatnonzero(gained)
        samples[f"{name}_spoiled"] = np.flatnonzero(lost)

    def describe(i):
        return {"game_id": str(data.iloc[i]["game_id"]), "date": str(data.iloc[i]["date"]), "blue_win": int(y[i]),
                "evidence_inputs": dict(zip(CONTEXT, context[i].tolist())),
                "blue_side": [name for name, vote in zip(EXPERTS, votes[i]) if vote],
                "red_side": [name for name, vote in zip(EXPERTS, votes[i]) if not vote],
                "expert_probabilities": dict(zip(EXPERTS, p[i].tolist())),
                "expert_original_minus_neutralized_probability": {
                    name: dict(zip(CONTEXT, reasons[i, j].tolist())) for j, name in enumerate(EXPERTS)
                },
                "evidence_coefficients": dict(zip(EXPERTS, trace["reliability"][i].tolist())),
                "blue_evidence": float(trace["blue_evidence"][i]), "red_evidence": float(trace["red_evidence"][i]),
                "judge_context_independent": float(trace["context_judge"][i]),
                "court_probabilities": {n: float(v[i]) for n, v in probabilities.items()}}

    scenario_results = {}
    for name, mask in scenarios.items():
        scenario_results[name] = {
            "games": int(mask.sum()), "elo_correct": int(elo_correct[mask].sum()),
            "expert_correct": dict(zip(EXPERTS, (votes[mask] == y[mask, None]).sum(axis=0).tolist())),
            "mean_gating_weights": (dict(zip(EXPERTS, trace["gating_weights"][mask].mean(axis=0).tolist()))
                                    if mask.any() else None),
            "court_correct": {n: int(((v[mask] >= 0.5) == y[mask]).sum()) for n, v in probabilities.items()},
        }
    return {"scenarios": scenario_results,
            "changes": changes, "examples": {n: [describe(i) for i in indices[:2]] for n, indices in samples.items()}}


def run(raw):
    matches, history, _ = load_matches(raw)
    matches = add_series_groups(matches)
    excluded = matches.loc[matches.date >= TEST_START]
    development = purge_series(matches.loc[matches.date < TEST_START], excluded).reset_index(drop=True)
    records = feature_rows(development, history.loc[history.date < TEST_START])
    context = np.array([context_row(r) for r in records])
    y = development[TARGET].to_numpy()
    folds = split_folds(development)
    legacy = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    oof = []
    explanations = []
    selection = []
    additional_expert_weights = []
    details = []
    for i, (train, valid) in enumerate(folds):
        details.append({"fold": i + 1, "expert_train": period(development.iloc[train]),
                        "validation": period(development.iloc[valid]),
                        "role": ("meta_train" if i < 2 else "selection" if i < 4 else "exploratory_audit")})
        if i == len(folds) - 1:
            break
        held, reasons = expert_predictions(development, records, y, train, valid, legacy)
        if i >= 2:
            meta = pd.concat(oof)
            model = fit_court(meta[list(EXPERTS)].to_numpy(), context[meta.index], np.concatenate(explanations), y[meta.index])
            methods, _ = model.evaluate(held[list(EXPERTS)].to_numpy(), context[valid], reasons)
            extra = [*EXPERTS, REFERENCE]
            weights = fit_weights(meta[extra].to_numpy(), y[meta.index])
            additional_expert_weights.append(dict(zip(extra, weights.tolist())))
            selection.append(held.assign(**methods, weighted_with_gold_sensitivity=held[extra].to_numpy() @ weights))
        oof.append(held)
        explanations.append(reasons)

    select = pd.concat(selection)
    select_metrics = {n: metric_table(y[select.index], select[n].to_numpy(), select[n].to_numpy() >= 0.5) for n in select}
    winner = min(SELECTABLE, key=lambda n: select_metrics[n]["log_loss"])
    # Seçim katı sonuçlarıyla yöntem donduruldu. Son katın sonuçları seçimi değiştiremez.
    meta = pd.concat(oof)
    meta_reasons = np.concatenate(explanations)
    court = fit_court(meta[list(EXPERTS)].to_numpy(), context[meta.index], meta_reasons, y[meta.index])
    keep = ["five_features", "random_forest", "xgboost"]
    keep_indices = [EXPERTS.index(name) for name in keep]
    without_elo = fit_court(meta[keep].to_numpy(), context[meta.index, :-1],
                            meta_reasons[:, keep_indices, :-1], y[meta.index])
    train, valid = folds[-1]
    held, reasons = expert_predictions(development, records, y, train, valid, legacy)
    p = held[list(EXPERTS)].to_numpy()
    outcomes, trace = court.evaluate(p, context[valid], reasons)
    ablation, _ = without_elo.evaluate(held[keep].to_numpy(), context[valid, :-1], reasons[:, keep_indices, :-1])
    baseline = held["elo"].to_numpy()
    y_audit = y[valid]
    data_audit = development.iloc[valid]
    groups = data_audit.series_group
    full = held.assign(**outcomes)
    audit = {n: {**metric_table(y_audit, full[n].to_numpy(), full[n].to_numpy() >= 0.5),
                 "calibration": calibration(y_audit, full[n].to_numpy()),
                 "vs_elo": comparison(y_audit, baseline, full[n].to_numpy(), groups)} for n in full}
    return {
        "status": "exploratory", "promotion": False, "raw_sha256": file_hash(raw),
        "excluded_test_start": str(TEST_START), "excluded_test_games": len(excluded),
        "experts": list(EXPERTS), "selected_method": winner, "selection_metric": "log_loss",
        "history_delay_days": HISTORY_DELAY.days, "folds": details,
        "selection_metrics": select_metrics, "selection_diversity": diversity(select[[*EXPERTS, REFERENCE]], y[select.index]),
        "gold_sensitivity_selection_weights": additional_expert_weights,
        "audit_metrics": audit, "audit_diversity": diversity(held[[*EXPERTS, REFERENCE]], y_audit),
        "fixed_weights": dict(zip(EXPERTS, court.weights.tolist())),
        "mean_audit_gating_weights": dict(zip(EXPERTS, trace["gating_weights"].mean(axis=0).tolist())),
        "evidence_evaluator_features": ["confidence", "blue_vote", "other_supporting_votes",
                                        *(f"aligned_{name}" for name in CONTEXT),
                                        *(f"aligned_model_response_{name}" for name in CONTEXT)],
        "standardized_evidence_coefficients": {n: model[-1].coef_[0].tolist()
                                               for n, model in zip(EXPERTS, court.reliability)},
        "selected_without_elo": {**metric_table(y_audit, ablation[winner], ablation[winner] >= 0.5),
                                "effect_of_elo": comparison(y_audit, ablation[winner], outcomes[winner], groups)},
        "audit_cases": summarize_cases(data_audit, y_audit, outcomes, p, trace, context[valid], reasons),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run(args.raw)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
