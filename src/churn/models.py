"""Fabrique de pipelines et espaces de recherche d'hyperparamètres."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import SEED
from churn.features import FeatureEngineer, make_preprocessor

MODEL_NAMES = ["logreg", "xgboost", "lightgbm"]


def make_estimator(name: str, params: dict[str, Any] | None = None):
    params = dict(params or {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=SEED, **params)
    if name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1, **params
        )
    if name == "lightgbm":
        return LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **params)
    raise ValueError(f"Modèle inconnu : {name}")


def build_pipeline(name: str, params: dict[str, Any] | None = None) -> Pipeline:
    """Pipeline complet : variables métier -> encodage -> modèle.

    La standardisation n'est utile qu'au modèle linéaire ; les arbres y sont invariants.
    """
    pipe = Pipeline(
        [
            ("features", FeatureEngineer()),
            ("preprocess", make_preprocessor(scale=name in {"logreg", "dummy"})),
            ("model", make_estimator(name, params)),
        ]
    )
    return pipe.set_output(transform="pandas")


def _logreg_space(trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _xgboost_space(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
    }


def _lightgbm_space(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


SEARCH_SPACES: dict[str, Callable] = {
    "logreg": _logreg_space,
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
}
