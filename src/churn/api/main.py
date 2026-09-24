"""API FastAPI de scoring du churn : `uvicorn churn.api.main:app --reload`."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn import __version__
from churn.api.schemas import (
    BatchOut,
    BatchRequest,
    CustomerFeatures,
    HealthOut,
    PredictionOut,
)
from churn.config import MODEL_PATH
from churn.predict import ChurnPredictor

log = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "predictor", None) is None:
        try:
            app.state.predictor = ChurnPredictor.load(MODEL_PATH)
            log.info("Modèle chargé depuis %s", MODEL_PATH)
        except FileNotFoundError:
            app.state.predictor = None
            log.warning("Aucun modèle trouvé à %s — lancer `python -m churn.train`.", MODEL_PATH)
    yield


app = FastAPI(
    title="Churn Prediction API",
    version=__version__,
    description=(
        "Score le risque de départ d'un client télécom et recommande (ou non) une offre de "
        "rétention selon un seuil optimisé sur le profit de la campagne."
    ),
    lifespan=lifespan,
)


def get_predictor(request: Request) -> ChurnPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return predictor


def _to_frame(customers: list[CustomerFeatures]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in customers])


@app.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    loaded = getattr(request.app.state, "predictor", None) is not None
    return HealthOut(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/model/info")
def model_info(request: Request) -> dict:
    return get_predictor(request).info()


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerFeatures, request: Request) -> PredictionOut:
    result = get_predictor(request).predict(_to_frame([customer]))[0]
    return PredictionOut(**result)


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchRequest, request: Request, explain: bool = False) -> BatchOut:
    results = get_predictor(request).predict(_to_frame(batch.customers), explain=explain)
    targeted = [r for r in results if r["recommended_action"] == "retention_offer"]
    return BatchOut(
        predictions=[PredictionOut(**r) for r in results],
        n_targeted=len(targeted),
        total_expected_gain=round(sum(r["expected_gain_if_targeted"] for r in targeted), 2),
    )
