"""Configuration centrale : chemins, graine aléatoire et hypothèses métier."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(os.environ.get("CHURN_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", MODELS_DIR / "churn_model.joblib"))
METADATA_PATH = MODELS_DIR / "metadata.json"
FIGURES_DIR = ROOT / "reports" / "figures"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

TARGET = "Churn"
ID_COL = "customerID"


@dataclass(frozen=True)
class BusinessAssumptions:
    """Hypothèses de la campagne de rétention (modifiables, cf. analyse de sensibilité).

    - `clv_months` : revenu perdu quand un client part (valeur = MonthlyCharges × horizon).
    - `offer_cost` : coût unitaire de l'offre de rétention (remise, appel conseiller…).
    - `acceptance_rate` : probabilité qu'un futur churner ciblé reste grâce à l'offre.
    """

    clv_months: int = 12
    offer_cost: float = 50.0
    acceptance_rate: float = 0.30

    def to_dict(self) -> dict:
        return asdict(self)


BUSINESS = BusinessAssumptions()
