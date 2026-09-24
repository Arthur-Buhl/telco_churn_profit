"""Ingénierie des variables, encapsulée dans des transformers scikit-learn.

Tout ce qui apprend quelque chose des données (moyennes, encodages, échelles) est fait
dans `fit`, donc uniquement sur les plis d'entraînement : pas de fuite vers la validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Ajoute des variables métier au DataFrame brut nettoyé.

    Seule partie apprise : le tarif mensuel moyen par type de contrat, utilisé pour mesurer
    si un client paie plus cher que ses pairs (`charge_vs_peers`).
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.contract_mean_charge_ = X.groupby("Contract")["MonthlyCharges"].mean().to_dict()
        self.global_mean_charge_ = float(X["MonthlyCharges"].mean())
        # `_engineer` et non `transform` : ce dernier est enveloppé par set_output et lirait
        # feature_names_out_ avant sa création.
        self.feature_names_out_ = np.asarray(self._engineer(X.head(1)).columns, dtype=object)
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._engineer(X)

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # "No internet service" / "No phone service" redondants avec *Service.
        for col in [*ADDON_SERVICES, "MultipleLines"]:
            X[col] = X[col].replace({"No internet service": "No", "No phone service": "No"})

        has_internet = X["InternetService"] != "No"
        X["nb_services"] = (
            (X["PhoneService"] == "Yes").astype(int)
            + (X["MultipleLines"] == "Yes").astype(int)
            + has_internet.astype(int)
            + (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        )
        X["nb_addons"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)

        tenure = X["tenure"].astype(float)
        X["avg_monthly_charge"] = np.where(
            tenure > 0, X["TotalCharges"] / tenure.clip(lower=1), X["MonthlyCharges"]
        )
        # > 0 : le client paie aujourd'hui plus que sa moyenne historique (hausse tarifaire).
        X["charge_increase"] = X["MonthlyCharges"] - X["avg_monthly_charge"]
        peers = X["Contract"].map(self.contract_mean_charge_).fillna(self.global_mean_charge_)
        X["charge_vs_peers"] = X["MonthlyCharges"] - peers

        X["tenure_bucket"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
        X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
        X["auto_payment"] = X["PaymentMethod"].str.contains("automatic").astype(int)
        X["has_security_support"] = (
            (X["OnlineSecurity"] == "Yes") & (X["TechSupport"] == "Yes")
        ).astype(int)
        X["fiber_no_support"] = (
            (X["InternetService"] == "Fiber optic") & (X["TechSupport"] == "No")
        ).astype(int)
        return X


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Encodage one-hot des catégorielles + standardisation des numériques (pour la LogReg)."""
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        verbose_feature_names_out=True,
    )


def pretty_feature_name(name: str) -> str:
    """`cat__Contract_Month-to-month` -> `Contract=Month-to-month` ; `num__tenure` -> `tenure`."""
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("cat__"):
        base = name[5:]
        for col in _KNOWN_CATEGORICALS:
            if base.startswith(col + "_"):
                return f"{col}={base[len(col) + 1 :]}"
        return base
    return name


_KNOWN_CATEGORICALS = sorted(
    [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        *ADDON_SERVICES,
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_bucket",
    ],
    key=len,
    reverse=True,
)
