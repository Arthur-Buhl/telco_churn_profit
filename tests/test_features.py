import numpy as np
import pytest

from churn.features import FeatureEngineer, pretty_feature_name
from churn.models import MODEL_NAMES, build_pipeline


def test_feature_engineer_adds_business_features(xy):
    X, _ = xy
    out = FeatureEngineer().fit(X).transform(X)
    for col in [
        "nb_services",
        "avg_monthly_charge",
        "charge_vs_peers",
        "tenure_bucket",
        "auto_payment",
    ]:
        assert col in out.columns
    assert not (out[["MultipleLines", "OnlineSecurity"]] == "No internet service").any().any()
    assert out["nb_services"].between(0, 9).all()
    assert np.isfinite(out["avg_monthly_charge"]).all()


def test_feature_engineer_learns_only_from_fit_data(xy):
    """Absence de fuite : les statistiques apprises ne dépendent que des données de fit."""
    X, _ = xy
    train, other = X.iloc[:200], X.iloc[200:].copy()
    fe = FeatureEngineer().fit(train)
    before = dict(fe.contract_mean_charge_)
    other["MonthlyCharges"] *= 10
    out = fe.transform(other)
    assert fe.contract_mean_charge_ == before
    expected = other["MonthlyCharges"] - other["Contract"].map(before)
    np.testing.assert_allclose(out["charge_vs_peers"], expected)


def test_unknown_category_does_not_crash(xy):
    X, y = xy
    pipe = build_pipeline("logreg").fit(X, y)
    odd = X.head(3).copy()
    odd["PaymentMethod"] = "Crypto"
    assert pipe.predict_proba(odd).shape == (3, 2)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_pipeline_trains_and_outputs_probabilities(xy, name):
    X, y = xy
    proba = build_pipeline(name).fit(X, y).predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pretty_feature_name():
    assert pretty_feature_name("num__tenure") == "tenure"
    assert pretty_feature_name("cat__Contract_Month-to-month") == "Contract=Month-to-month"
    assert pretty_feature_name("cat__PaymentMethod_Credit card (automatic)") == (
        "PaymentMethod=Credit card (automatic)"
    )
