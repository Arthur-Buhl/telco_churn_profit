"""Entraînement de bout en bout.

    python -m churn.train --trials 50

1. Découpage stratifié train/test (test gelé).
2. Pour chaque modèle : tuning Optuna en CV stratifiée 5 plis (objectif PR-AUC).
3. Sélection du meilleur modèle + tests appariés (Wilcoxon) contre les autres.
4. Calibration (Platt) et choix du seuil qui maximise le profit sur les prédictions
   out-of-fold du train — jamais sur le test.
5. Évaluation unique sur le test, sauvegarde du modèle et des métadonnées.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from churn import __version__
from churn.config import BUSINESS, METADATA_PATH, MODEL_PATH, N_FOLDS, SEED, BusinessAssumptions
from churn.data import FEATURE_COLS, download, load, train_test
from churn.evaluate import (
    bootstrap_ci,
    classification_metrics,
    customer_value,
    optimal_threshold,
    paired_cv_test,
    strategy_comparison,
)
from churn.models import MODEL_NAMES, SEARCH_SPACES, build_pipeline

log = logging.getLogger("churn.train")

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"}


def make_cv(seed: int = SEED, n_splits: int = N_FOLDS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cv_scores(name: str, params: dict, X, y, cv) -> dict[str, list[float]]:
    res = cross_validate(build_pipeline(name, params), X, y, cv=cv, scoring=SCORING, n_jobs=1)
    return {
        "pr_auc": res["test_pr_auc"].tolist(),
        "roc_auc": res["test_roc_auc"].tolist(),
        "brier": (-res["test_brier"]).tolist(),
    }


def tune(name: str, X, y, cv, n_trials: int) -> dict[str, Any]:
    """Recherche bayésienne (TPE) des hyperparamètres ; renvoie les meilleurs paramètres."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SEARCH_SPACES[name]

    def objective(trial) -> float:
        pipe = build_pipeline(name, space(trial))
        res = cross_validate(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
        return float(res["test_score"].mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    # FixedTrial reconstruit aussi les paramètres fixes de l'espace (ex. subsample_freq).
    return space(optuna.trial.FixedTrial(study.best_params))


def calibrated(name: str, params: dict, cv) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(build_pipeline(name, params), method="sigmoid", cv=cv)


def build_artifact(
    name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
    assumptions: BusinessAssumptions = BUSINESS,
) -> dict[str, Any]:
    """Entraîne le modèle final (calibré) et choisit le seuil métier sur l'out-of-fold du train."""
    cv = cv or make_cv()
    oof = cross_val_predict(calibrated(name, params, cv), X, y, cv=cv, method="predict_proba")[:, 1]
    threshold, oof_profit = optimal_threshold(
        y, oof, customer_value(X["MonthlyCharges"], assumptions), assumptions
    )
    model = calibrated(name, params, cv).fit(X, y)
    explainer_pipeline = build_pipeline(name, params).fit(X, y)
    background = explainer_pipeline[:-1].transform(X.sample(min(100, len(X)), random_state=SEED))
    return {
        "model_name": name,
        "params": params,
        "model": model,
        "explainer_pipeline": explainer_pipeline,
        "background": background,
        "threshold": threshold,
        "oof_profit": oof_profit,
        "oof_metrics": classification_metrics(y, oof, threshold),
        "assumptions": assumptions.to_dict(),
        "feature_columns": FEATURE_COLS,
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run(n_trials: int = 50, models: list[str] | None = None) -> dict[str, Any]:
    models = models or MODEL_NAMES
    df = load(download())
    X_train, X_test, y_train, y_test = train_test(df)
    cv = make_cv()
    log.info("Train %d / test %d - taux de churn %.3f", len(X_train), len(X_test), y_train.mean())

    results: dict[str, Any] = {
        "dummy": {"params": {}, "cv": cv_scores("dummy", {}, X_train, y_train, cv)}
    }
    for name in models:
        params = tune(name, X_train, y_train, cv, n_trials) if n_trials > 0 else {}
        results[name] = {"params": params, "cv": cv_scores(name, params, X_train, y_train, cv)}
        log.info("%-9s PR-AUC CV = %.4f", name, np.mean(results[name]["cv"]["pr_auc"]))

    best = max(models, key=lambda m: np.mean(results[m]["cv"]["pr_auc"]))
    comparisons = {
        other: paired_cv_test(results[best]["cv"]["pr_auc"], results[other]["cv"]["pr_auc"])
        for other in results
        if other != best
    }
    log.info("Meilleur modèle : %s", best)

    artifact = build_artifact(best, results[best]["params"], X_train, y_train, cv)
    proba_test = artifact["model"].predict_proba(X_test)[:, 1]
    value_test = customer_value(X_test["MonthlyCharges"])
    thr = artifact["threshold"]
    roc, roc_lo, roc_hi = bootstrap_ci(y_test, proba_test)
    from sklearn.metrics import average_precision_score

    pr, pr_lo, pr_hi = bootstrap_ci(y_test, proba_test, metric=average_precision_score)

    metadata = {
        "model_name": best,
        "version": artifact["version"],
        "trained_at": artifact["trained_at"],
        "sklearn_version": sklearn.__version__,
        "threshold": thr,
        "assumptions": artifact["assumptions"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv": {m: {"params": r["params"], **r["cv"]} for m, r in results.items()},
        "cv_paired_tests_vs_best": comparisons,
        "oof_train": {"profit": artifact["oof_profit"], **artifact["oof_metrics"]},
        "test": {
            "at_business_threshold": classification_metrics(y_test, proba_test, thr),
            "at_0.5": classification_metrics(y_test, proba_test, 0.5),
            "roc_auc_ci95": [roc_lo, roc_hi],
            "pr_auc_ci95": [pr_lo, pr_hi],
            "strategies": strategy_comparison(y_test, proba_test, value_test, thr).to_dict(
                "records"
            ),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False), "utf-8")
    log.info(
        "Test : ROC-AUC %.3f [%.3f-%.3f], PR-AUC %.3f, seuil %.2f", roc, roc_lo, roc_hi, pr, thr
    )
    log.info("Modèle sauvegardé : %s", MODEL_PATH)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne le modèle de churn.")
    parser.add_argument(
        "--trials", type=int, default=50, help="essais Optuna par modèle (0 = défauts)"
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.trials, args.models)


if __name__ == "__main__":
    main()
