"""Chargement, nettoyage et découpage du jeu de données Telco Customer Churn."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import DATA_URL, ID_COL, RAW_CSV, SEED, TARGET, TEST_SIZE

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def download(path: Path = RAW_CSV, url: str = DATA_URL, force: bool = False) -> Path:
    """Télécharge le CSV IBM s'il est absent."""
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage sans état (applicable ligne à ligne, donc sans risque de fuite).

    - `TotalCharges` est une chaîne avec des blancs pour les clients à `tenure == 0` :
      ils n'ont encore rien payé, on impute donc 0 (et non la moyenne).
    - La cible `Churn` Yes/No devient 1/0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def load(path: Path = RAW_CSV) -> pd.DataFrame:
    """Charge et nettoie le jeu complet."""
    return clean(pd.read_csv(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS].copy(), df[TARGET].copy()


def train_test(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Découpage stratifié et reproductible. Le test reste gelé jusqu'à l'évaluation finale."""
    X, y = split_xy(df.drop(columns=[ID_COL], errors="ignore"))
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
