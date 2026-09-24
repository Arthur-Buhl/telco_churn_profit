"""Schémas Pydantic de l'API : validation stricte des entrées client."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 256.5,
}


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Ancienneté en mois")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, le=1000, description="Facture mensuelle (€)")
    TotalCharges: float = Field(ge=0, description="Total facturé depuis l'arrivée (€)")


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class Factor(BaseModel):
    feature: str
    impact: float = Field(description="Contribution SHAP (log-odds)")
    effect: str


class PredictionOut(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    will_churn: bool
    threshold: float
    recommended_action: Literal["retention_offer", "none"]
    expected_gain_if_targeted: float = Field(
        description="p × acceptation × valeur − coût offre (€)"
    )
    top_factors: list[Factor]


class BatchOut(BaseModel):
    predictions: list[PredictionOut]
    n_targeted: int
    total_expected_gain: float


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
