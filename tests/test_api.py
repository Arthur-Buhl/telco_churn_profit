import pytest
from fastapi.testclient import TestClient

from churn.api.main import app


@pytest.fixture
def client(predictor):
    # Sans `with` : le lifespan (chargement du modèle sur disque) n'est pas déclenché.
    app.state.predictor = predictor
    yield TestClient(app)
    app.state.predictor = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_customer(client, customer):
    res = client.post("/predict", json=customer)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])
    assert body["recommended_action"] in {"retention_offer", "none"}
    assert len(body["top_factors"]) == 3


@pytest.mark.parametrize(
    "field, value",
    [("Contract", "Three year"), ("tenure", -1), ("MonthlyCharges", 0), ("SeniorCitizen", 2)],
)
def test_predict_rejects_invalid_values(client, customer, field, value):
    customer[field] = value
    assert client.post("/predict", json=customer).status_code == 422


def test_predict_rejects_missing_and_extra_fields(client, customer):
    missing = {k: v for k, v in customer.items() if k != "tenure"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**customer, "foo": 1}).status_code == 422


def test_batch(client, customer):
    loyal = {**customer, "Contract": "Two year", "tenure": 70, "TotalCharges": 5985.0}
    res = client.post("/predict/batch", json={"customers": [customer, loyal]})
    assert res.status_code == 200
    body = res.json()
    assert len(body["predictions"]) == 2
    assert body["n_targeted"] == sum(
        p["recommended_action"] == "retention_offer" for p in body["predictions"]
    )
    assert body["predictions"][0]["churn_probability"] > body["predictions"][1]["churn_probability"]


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_name"] == "logreg"
    assert 0 <= info["threshold"] <= 1


def test_503_without_model(customer):
    app.state.predictor = None
    client = TestClient(app)
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/predict", json=customer).status_code == 503
