import numpy as np
import pytest

from churn.config import BusinessAssumptions
from churn.evaluate import (
    campaign_profit,
    classification_metrics,
    customer_value,
    expected_action_gain,
    lift_table,
    optimal_threshold,
    profit_curve,
    strategy_comparison,
)

A = BusinessAssumptions(clv_months=10, offer_cost=20.0, acceptance_rate=0.5)


def test_campaign_profit_on_known_case():
    y = np.array([1, 1, 0, 0])
    value = np.array([100.0, 200.0, 100.0, 100.0])
    target = np.array([True, False, True, False])
    # TP : 0.5 × 100 − 20 = 30 ; FP : −20 ; non ciblés : 0
    assert campaign_profit(y, target, value, A) == pytest.approx(10.0)


def test_no_action_is_zero_and_oracle_is_best():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    value = customer_value(rng.uniform(20, 120, 200), A)
    proba = np.clip(y * 0.6 + rng.uniform(0, 0.4, 200), 0, 1)
    table = strategy_comparison(y, proba, value, threshold=0.5, assumptions=A).set_index(
        "strategie"
    )
    assert table.loc["Aucune action", "profit"] == 0
    assert table.loc["Oracle (churners connus)", "profit"] == table["profit"].max()


def test_optimal_threshold_maximises_profit_curve():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    proba = np.clip(0.3 * y + rng.uniform(0, 0.7, 500), 0, 1)
    value = np.full(500, 600.0)
    thr, profit = optimal_threshold(y, proba, value, A)
    curve = profit_curve(y, proba, value, A)
    assert profit == pytest.approx(curve["profit"].max())
    assert 0 <= thr <= 1


def test_expected_action_gain():
    gain = expected_action_gain([0.0, 1.0], [100.0, 100.0], A)
    np.testing.assert_allclose(gain, [-20.0, 30.0])


def test_classification_metrics_counts():
    m = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.6], threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_lift_table_first_decile_is_riskiest():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 1000)
    proba = np.clip(0.5 * y + rng.uniform(0, 0.5, 1000), 0, 1)
    lift = lift_table(y, proba)
    assert len(lift) == 10
    assert lift.loc[0, "lift"] > lift.loc[9, "lift"]
    assert lift["part_churners_cumulee"].iloc[-1] == pytest.approx(1.0)
