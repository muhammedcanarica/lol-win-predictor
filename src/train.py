"""Üç modeli eğitir (GridSearchCV + zaman serisi CV), en iyiyi CV ile seçer, testte değerlendirir.

Çalıştırma: python -m src.train
Çıktılar: models/best_model.joblib, reports/metrics.json, reports/figures/*.png
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import FEATURES, PROCESSED_PATH, ROOT, TARGET, chrono_split, clean, load_raw
from src.logistic import LOGISTIC_C_GRID, logistic_pipeline

SEED = 42
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"
GROUP_PATH = ROOT / "reports" / "group_performance.csv"
MIN_GROUP_N = 100  # bundan küçük patch/lig gruplarından sonuç çıkarılmaz

# Ölçekleme Pipeline içinde: her CV katında yalnızca eğitim katına fit edilir (sızıntı yok).
CANDIDATES = {
    "Logistic Regression": (logistic_pipeline(), LOGISTIC_C_GRID),
    "Random Forest": (
        Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
        {"model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [1, 20, 50]},
    ),
    "XGBoost": (
        Pipeline([("model", XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss", subsample=0.8))]),
        {"model__n_estimators": [200, 400], "model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1]},
    ),
}


def evaluate(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def paired_bootstrap(y, p_model, sign_pred, golddiff, n=2000) -> dict:
    """Aynı test maçlarını yeniden örnekleyerek model − altın-işareti farkının %95 güven aralığı."""
    rng = np.random.default_rng(SEED)
    d_acc, d_auc = [], []
    pred = (p_model >= 0.5).astype(int)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d_acc.append((pred[i] == y[i]).mean() - (sign_pred[i] == y[i]).mean())
        d_auc.append(roc_auc_score(y[i], p_model[i]) - roc_auc_score(y[i], golddiff[i]))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"accuracy_diff": float((pred == y).mean() - (sign_pred == y).mean()), "accuracy_diff_ci95": ci(d_acc),
            "roc_auc_diff": float(roc_auc_score(y, p_model) - roc_auc_score(y, golddiff)), "roc_auc_diff_ci95": ci(d_auc)}


def group_performance(test: pd.DataFrame, proba, min_n=MIN_GROUP_N) -> pd.DataFrame:
    """Test setinde patch ve lig bazında performans. n < min_n gruplar 'reliable=False' işaretlenir."""
    t = test.assign(p=proba, pred=(proba >= 0.5).astype(int), sign=(test["golddiffat10"] > 0).astype(int))
    rows = []
    for col in ["patch", "league"]:
        for g, d in t.groupby(col):
            both = d[TARGET].nunique() == 2
            rows.append({"group_by": col, "group": f"{g:.2f}" if col == "patch" else g, "n": len(d),
                         "blue_win_rate": d[TARGET].mean(), "accuracy": (d.pred == d[TARGET]).mean(),
                         "sign_accuracy": (d.sign == d[TARGET]).mean(),
                         "roc_auc": roc_auc_score(d[TARGET], d.p) if both else np.nan,
                         "reliable": len(d) >= min_n})
    return pd.DataFrame(rows).sort_values(["group_by", "n"], ascending=[True, False])


def main():
    data, cleaning = clean(load_raw())
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)

    train, test = chrono_split(data)
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    split = {
        "train_games": len(train), "test_games": len(test),
        "train_period": [str(train.date.min()), str(train.date.max())],
        "test_period": [str(test.date.min()), str(test.date.max())],
        "train_blue_win_rate": round(float(y_tr.mean()), 4), "test_blue_win_rate": round(float(y_te.mean()), 4),
        "test_games_whose_series_has_train_games": int(test["series"].isin(train["series"]).sum()),
    }
    print("Temizleme:", cleaning, "\nBölme:", split)

    cv = TimeSeriesSplit(n_splits=5)  # doğrulama katları her zaman eğitim katlarından sonra gelir
    fitted, cv_results, test_results = {}, {}, {}
    for name, (pipe, grid) in CANDIDATES.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        fitted[name] = gs.best_estimator_
        i = gs.best_index_
        cv_results[name] = {"best_params": gs.best_params_, "cv_roc_auc": gs.cv_results_["mean_test_score"][i],
                            "cv_roc_auc_std": gs.cv_results_["std_test_score"][i]}
        print(f"{name}: CV AUC={cv_results[name]['cv_roc_auc']:.4f} {gs.best_params_}")

    # Seçim yalnızca CV (doğrulama) skoruna göre; test seti seçimde kullanılmaz.
    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_roc_auc"])
    for name, model in fitted.items():
        test_results[name] = evaluate(y_te, model.predict_proba(X_te)[:, 1])

    best = fitted[best_name]
    p_best = best.predict_proba(X_te)[:, 1]
    sign_pred = (X_te["golddiffat10"] > 0).astype(int)
    baselines = {
        "always_blue_accuracy": float(y_te.mean()),
        "golddiff_sign_accuracy": float((sign_pred == y_te).mean()),
        # Altın farkının kendisi skor olarak: tek özellikle elde edilebilecek sıralama kalitesi
        "golddiff_only_roc_auc": float(roc_auc_score(y_te, X_te["golddiffat10"])),
        "best_vs_sign_bootstrap": paired_bootstrap(y_te.to_numpy(), p_best, sign_pred.to_numpy(),
                                                   X_te["golddiffat10"].to_numpy()),
    }
    # Sabit olasılık referansı: her maça eğitimdeki Blue galibiyet oranını vermek (bilgisiz ama kalibre)
    base_rate = float(y_tr.mean())
    const = np.full(len(y_te), base_rate)
    calibration = {
        "constant_prob": round(base_rate, 4),
        "brier_constant": brier_score_loss(y_te, const), "brier_model": test_results[best_name]["brier"],
        "logloss_constant": log_loss(y_te, const), "logloss_model": log_loss(y_te, p_best),
        "brier_skill_score": 1 - test_results[best_name]["brier"] / brier_score_loss(y_te, const),
        "mean_predicted": float(p_best.mean()), "observed_rate": float(y_te.mean()),
    }
    groups = group_performance(test, p_best)
    groups.to_csv(GROUP_PATH, index=False)
    pi = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    importance = dict(sorted(zip(FEATURES, pi.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best, "name": best_name, "features": FEATURES}, MODEL_PATH)

    metrics = {"cleaning": cleaning, "split": split, "cv": cv_results, "test": test_results,
               "best_model": best_name, "baselines": baselines, "calibration": calibration,
               "feature_importance": importance,
               "groups_reliable": groups[groups.reliable].drop(columns="reliable").round(4).to_dict("records")}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    plot_all(fitted, X_te, y_te, test_results, cv_results, best_name, importance)
    print(f"En iyi model: {best_name}\nTest:", {k: round(v, 4) for k, v in test_results[best_name].items()
                                                 if k != "confusion_matrix"})


def plot_all(fitted, X_te, y_te, test_results, cv_results, best_name, importance):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = list(fitted)

    # 1) Model karşılaştırma
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    tbl = pd.DataFrame({n: [test_results[n][k] for k in keys] for n in names}, index=keys)
    ax = tbl.plot.bar(figsize=(10, 5), rot=0)
    ax.set_ylim(0.5, 0.9); ax.set_title("Test seti metrikleri (eşik 0.5)"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG_DIR / "model_comparison.png", dpi=120); plt.close()

    # 2) ROC eğrileri
    plt.figure(figsize=(6, 6))
    for n in names:
        fpr, tpr, _ = roc_curve(y_te, fitted[n].predict_proba(X_te)[:, 1])
        plt.plot(fpr, tpr, label=f"{n} (AUC={test_results[n]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("Yanlış pozitif oranı"); plt.ylabel("Doğru pozitif oranı")
    plt.title("ROC eğrileri (test)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=120); plt.close()

    # 3) Karışıklık matrisi (seçilen model)
    ConfusionMatrixDisplay(np.array(test_results[best_name]["confusion_matrix"]),
                           display_labels=["Red kazandı", "Blue kazandı"]).plot(cmap="Blues", colorbar=False)
    plt.title(f"Karışıklık matrisi – {best_name}"); plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=120); plt.close()

    # 4) Kalibrasyon
    plt.figure(figsize=(6, 6))
    for n in names:
        frac, mean_pred = calibration_curve(y_te, fitted[n].predict_proba(X_te)[:, 1], n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac, "o-", label=f"{n} (Brier={test_results[n]['brier']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("Tahmin edilen olasılık"); plt.ylabel("Gerçek Blue galibiyet oranı")
    plt.title("Kalibrasyon eğrisi (test)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration.png", dpi=120); plt.close()

    # 5) Özellik önemi
    s = pd.Series(importance).sort_values()
    s.plot.barh(figsize=(7, 4), color="#c89b3c"); plt.xlabel("ROC-AUC düşüşü (permütasyon)")
    plt.title(f"Özellik önemi – {best_name}"); plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=120); plt.close()

    # 6) CV skorları
    cvs = pd.Series({n: cv_results[n]["cv_roc_auc"] for n in names})
    err = [cv_results[n]["cv_roc_auc_std"] for n in names]
    ax = cvs.plot.bar(yerr=err, rot=0, figsize=(7, 4), color=["#0a6ebd", "#2e7d32", "#c89b3c"])
    ax.set_ylim(0.7, 0.85); ax.set_ylabel("Ortalama CV ROC-AUC"); ax.set_title("Doğrulama (TimeSeriesSplit, 5 kat)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "cv_scores.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
