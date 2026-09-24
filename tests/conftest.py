"""Fixtures : données synthétiques au format Telco, pour que les tests n'exigent pas le CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.api.schemas import EXAMPLE_CUSTOMER
from churn.predict import ChurnPredictor
from churn.train import build_artifact, make_cv

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Clients cohérents (pas d'add-on sans internet, etc.) churn lié au contrat/ancienneté."""
    rng = np.random.default_rng(seed)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.2, 0.25])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            **{
                col: np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
                for col in ADDONS
            },
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            # Format brut du CSV : chaîne, blanc pour les nouveaux clients.
            "TotalCharges": [
                " " if t == 0 else f"{t * m:.2f}" for t, m in zip(tenure, monthly, strict=True)
            ],
        }
    )
    logit = (
        -1.0
        + 1.8 * (contract == "Month-to-month")
        - 0.04 * tenure
        + 0.8 * (internet == "Fiber optic")
    )
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    from churn.data import clean

    return clean(raw_df)


@pytest.fixture(scope="session")
def xy(clean_df):
    from churn.data import split_xy

    return split_xy(clean_df)


@pytest.fixture(scope="session")
def artifact(xy):
    X, y = xy
    return build_artifact("logreg", {"C": 1.0}, X, y, cv=make_cv(n_splits=3))


@pytest.fixture(scope="session")
def predictor(artifact) -> ChurnPredictor:
    return ChurnPredictor(artifact)


@pytest.fixture
def customer() -> dict:
    return dict(EXAMPLE_CUSTOMER)
