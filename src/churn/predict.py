"""Inférence : probabilité de churn, décision métier et explication locale (SHAP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churn.config import MODEL_PATH, BusinessAssumptions
from churn.data import clean
from churn.evaluate import customer_value, expected_action_gain
from churn.features import pretty_feature_name


class ChurnPredictor:
    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.assumptions = BusinessAssumptions(**artifact["assumptions"])
        self.feature_columns: list[str] = artifact["feature_columns"]
        pipeline = artifact["explainer_pipeline"]
        self._preprocess = pipeline[:-1]
        self._estimator = pipeline[-1]
        self._explainer = None

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> ChurnPredictor:
        return cls(joblib.load(path))

    @property
    def explainer(self):
        # Construit à la demande : shap est lent à importer.
        if self._explainer is None:
            import shap

            if hasattr(self._estimator, "coef_"):
                self._explainer = shap.LinearExplainer(self._estimator, self.artifact["background"])
            else:
                self._explainer = shap.TreeExplainer(self._estimator)
        return self._explainer

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return clean(df)[self.feature_columns]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare(df))[:, 1]

    def explain(self, df: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, Any]]]:
        """Top-k contributions SHAP (en log-odds du modèle non calibré) par client."""
        prepared = self._prepare(df)
        raw = self._preprocess[0].transform(prepared)  # variables métier avant standardisation
        Xt = self._preprocess.transform(prepared)
        values = self.explainer.shap_values(Xt)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        names = list(Xt.columns)
        out = []
        for i, row in enumerate(values):
            order = np.argsort(-np.abs(row))[:top_k]
            out.append(
                [
                    {
                        "feature": _describe(names[j], Xt.iloc[i, j], raw.iloc[i]),
                        "impact": round(float(row[j]), 4),
                        "effect": "augmente le risque" if row[j] > 0 else "diminue le risque",
                    }
                    for j in order
                ]
            )
        return out

    def predict(self, df: pd.DataFrame, explain: bool = True) -> list[dict[str, Any]]:
        proba = self.predict_proba(df)
        gain = expected_action_gain(
            proba, customer_value(df["MonthlyCharges"], self.assumptions), self.assumptions
        )
        factors = self.explain(df) if explain else [[] for _ in range(len(df))]
        return [
            {
                "churn_probability": round(float(p), 4),
                "will_churn": bool(p >= self.threshold),
                "threshold": self.threshold,
                "recommended_action": "retention_offer" if p >= self.threshold else "none",
                "expected_gain_if_targeted": round(float(g), 2),
                "top_factors": f,
            }
            for p, g, f in zip(proba, gain, factors, strict=True)
        ]

    def info(self) -> dict[str, Any]:
        a = self.artifact
        return {
            "model_name": a["model_name"],
            "version": a["version"],
            "trained_at": a["trained_at"],
            "threshold": self.threshold,
            "assumptions": a["assumptions"],
            "params": a["params"],
        }


def _describe(name: str, encoded: float, raw_row: pd.Series) -> str:
    """Nom lisible : `tenure = 3`, `Contract=Month-to-month`, ou `≠` pour une modalité absente."""
    label = pretty_feature_name(name)
    if name.startswith("num__") and label in raw_row.index:
        value = raw_row[label]
        return f"{label} = {round(float(value), 2):g}"
    if name.startswith("cat__") and "=" in label and float(encoded) == 0:
        return label.replace("=", " ≠ ", 1)
    return label
