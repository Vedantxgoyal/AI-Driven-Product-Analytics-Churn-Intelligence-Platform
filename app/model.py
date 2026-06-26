"""
app/model.py - XGBoost + SHAP model wrapper
"""

import pickle
import pathlib
import logging
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
SHAP_PATH  = BASE_DIR / "models" / "shap_explainer.pkl"

FEATURE_COLS = [
    "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
    "PreferedOrderCat", "SatisfactionScore", "MaritalStatus",
    "NumberOfAddress", "Complain", "DaySinceLastOrder", "CashbackAmount"
]
CAT_COLS = ["PreferedOrderCat", "MaritalStatus"]

# Fit label encoders on known categories from training data
_CAT_VALUES = {
    "PreferedOrderCat": ["Laptop & Accessory", "Mobile Phone", "Fashion",
                          "Grocery", "Others"],
    "MaritalStatus":    ["Single", "Divorced", "Married"],
}
_ENCODERS: Dict[str, LabelEncoder] = {}
for col, vals in _CAT_VALUES.items():
    le = LabelEncoder()
    le.fit(vals)
    _ENCODERS[col] = le


def _encode_cat(col: str, val: str) -> int:
    le = _ENCODERS[col]
    if val not in le.classes_:
        logger.warning(f"Unseen category '{val}' for {col}, defaulting to 0")
        return 0
    return int(le.transform([val])[0])


def _build_frame(features: dict) -> pd.DataFrame:
    """Map API snake_case keys → training column names, encode cats."""
    row = {
        "Tenure":                    features["tenure"],
        "WarehouseToHome":           features["warehouse_to_home"],
        "NumberOfDeviceRegistered":  features["number_of_device_registered"],
        "PreferedOrderCat":          _encode_cat("PreferedOrderCat", features["preferred_order_cat"]),
        "SatisfactionScore":         features["satisfaction_score"],
        "MaritalStatus":             _encode_cat("MaritalStatus", features["marital_status"]),
        "NumberOfAddress":           features["number_of_address"],
        "Complain":                  features["complain"],
        "DaySinceLastOrder":         features["days_since_last_order"],
        "CashbackAmount":            features["cashback_amount"],
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


class ChurnModel:
    def __init__(self) -> None:
        self.model     = None
        self.explainer = None

    def load(self) -> bool:
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(SHAP_PATH, "rb") as f:
                self.explainer = pickle.load(f)
            logger.info("✓ XGBoost model + SHAP explainer loaded")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

    def is_loaded(self) -> bool:
        return self.model is not None

    def _risk(self, prob: float) -> Tuple[str, str]:
        if prob >= 0.7:
            return "HIGH",   "Immediate action: call customer, offer loyalty incentive"
        if prob >= 0.4:
            return "MEDIUM", "Proactive engagement: targeted offer, request feedback"
        return "LOW",        "Maintain engagement: regular check-ins, loyalty rewards"

    def predict_single(self, features: dict) -> Tuple[float, str, str]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        X = _build_frame(features)
        prob = float(self.model.predict_proba(X)[0, 1])
        risk, rec = self._risk(prob)
        return prob, risk, rec

    def explain_single(self, features: dict) -> Dict:
        """Return SHAP values for one customer."""
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not loaded")
        X = _build_frame(features)
        shap_vals = self.explainer.shap_values(X)

        # shap_vals shape: (1, n_features) for binary XGBoost TreeExplainer
        vals = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
        expected = float(self.explainer.expected_value
                         if not hasattr(self.explainer.expected_value, "__len__")
                         else self.explainer.expected_value[-1])

        contributions = {
            col: round(float(v), 4)
            for col, v in zip(FEATURE_COLS, vals)
        }
        top_drivers = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return {
            "expected_value":  round(expected, 4),
            "contributions":   contributions,
            "top_5_drivers":   [{"feature": k, "shap_value": v} for k, v in top_drivers],
        }

    def predict_batch(self, features_list: List[dict]) -> List[Tuple[float, str, str]]:
        return [self.predict_single(f) for f in features_list]


# ── Singleton ──────────────────────────────────────────────────────────────
_model_instance: ChurnModel | None = None


def get_model() -> ChurnModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = ChurnModel()
        _model_instance.load()
    return _model_instance


def initialize_model(*_) -> bool:
    global _model_instance
    _model_instance = ChurnModel()
    return _model_instance.load()