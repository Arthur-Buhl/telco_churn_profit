"""Métriques statistiques et traduction des prédictions en impact financier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import BUSINESS, BusinessAssumptions


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Métriques de ranking (indépendantes du seuil) + métriques au seuil choisi."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# --- Impact métier -----------------------------------------------------------------------------
#
# Référence : « aucune action » (profit = 0). Pour chaque client ciblé par la campagne :
#   - vrai positif  (churner ciblé)      : +acceptance_rate × valeur_client − offer_cost
#   - faux positif  (fidèle ciblé)       : −offer_cost (offre donnée pour rien)
#   - faux négatif  (churner non ciblé)  : 0 vs référence, mais manque à gagner = valeur perdue
#   - vrai négatif                       : 0
# valeur_client = MonthlyCharges × clv_months.


def customer_value(monthly_charges, assumptions: BusinessAssumptions = BUSINESS) -> np.ndarray:
    return np.asarray(monthly_charges, dtype=float) * assumptions.clv_months


def campaign_profit(
    y_true, target_mask, value, assumptions: BusinessAssumptions = BUSINESS
) -> float:
    """Profit net d'une campagne ciblant `target_mask`, relativement à ne rien faire."""
    y_true = np.asarray(y_true).astype(bool)
    target = np.asarray(target_mask).astype(bool)
    value = np.asarray(value, dtype=float)
    saved = assumptions.acceptance_rate * value[target & y_true].sum()
    return float(saved - assumptions.offer_cost * target.sum())


def profit_curve(
    y_true,
    proba,
    value,
    assumptions: BusinessAssumptions = BUSINESS,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Profit de la campagne pour chaque seuil de décision."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 201), 4)
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    rows = []
    for t in thresholds:
        target = proba >= t
        rows.append(
            {
                "threshold": float(t),
                "n_targeted": int(target.sum()),
                "tp": int((target & y_true).sum()),
                "fp": int((target & ~y_true).sum()),
                "fn": int((~target & y_true).sum()),
                "profit": campaign_profit(y_true, target, value, assumptions),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true, proba, value, assumptions: BusinessAssumptions = BUSINESS
) -> tuple[float, float]:
    """Seuil qui maximise le profit ; renvoie (seuil, profit)."""
    curve = profit_curve(y_true, proba, value, assumptions)
    best = curve.loc[curve["profit"].idxmax()]
    return float(best["threshold"]), float(best["profit"])


def expected_action_gain(proba, value, assumptions: BusinessAssumptions = BUSINESS) -> np.ndarray:
    """Gain espéré à cibler un client donné : p × acceptance × valeur − coût de l'offre."""
    return (
        np.asarray(proba) * assumptions.acceptance_rate * np.asarray(value, dtype=float)
        - assumptions.offer_cost
    )


def strategy_comparison(
    y_true, proba, value, threshold: float, assumptions: BusinessAssumptions = BUSINESS
) -> pd.DataFrame:
    """Compare le ciblage par modèle aux stratégies naïves et à l'oracle."""
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    n = len(y_true)
    strategies = {
        "Aucune action": np.zeros(n, dtype=bool),
        "Cibler tout le monde": np.ones(n, dtype=bool),
        "Modèle (seuil métier)": proba >= threshold,
        "Modèle (seuil 0.5)": proba >= 0.5,
        "Oracle (churners connus)": y_true,
    }
    rows = [
        {
            "strategie": name,
            "n_cibles": int(mask.sum()),
            "churners_cibles": int((mask & y_true).sum()),
            "profit": campaign_profit(y_true, mask, value, assumptions),
        }
        for name, mask in strategies.items()
    ]
    return pd.DataFrame(rows)


def lift_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Taux de churn et lift par décile de score (décile 1 = clients les plus risqués)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), n_bins, labels=False) + 1
    base = df["y"].mean()
    out = df.groupby("decile").agg(n=("y", "size"), churners=("y", "sum"), p_moy=("p", "mean"))
    out["taux_churn"] = out["churners"] / out["n"]
    out["lift"] = out["taux_churn"] / base
    out["part_churners_cumulee"] = out["churners"].cumsum() / df["y"].sum()
    return out.reset_index()


# --- Tests statistiques ------------------------------------------------------------------------


def mcnemar_test(y_true, pred_a, pred_b) -> dict[str, float]:
    """Test de McNemar : les deux modèles se trompent-ils sur des clients différents ?"""
    from statsmodels.stats.contingency_tables import mcnemar

    y_true = np.asarray(y_true)
    ok_a = np.asarray(pred_a) == y_true
    ok_b = np.asarray(pred_b) == y_true
    table = [
        [int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
        [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())],
    ]
    res = mcnemar(table, exact=False, correction=True)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "table": table}


def paired_cv_test(scores_a, scores_b) -> dict[str, float]:
    """Test de Wilcoxon apparié sur les scores par pli (mêmes plis pour les deux modèles)."""
    from scipy.stats import wilcoxon

    diff = np.asarray(scores_a) - np.asarray(scores_b)
    if np.allclose(diff, 0):
        return {"mean_diff": 0.0, "p_value": 1.0}
    res = wilcoxon(scores_a, scores_b)
    return {"mean_diff": float(diff.mean()), "p_value": float(res.pvalue)}


def bootstrap_ci(
    y_true, proba, metric=roc_auc_score, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Intervalle de confiance bootstrap (percentile) d'une métrique sur le test."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric(y_true[idx], proba[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, proba)), float(lo), float(hi)
