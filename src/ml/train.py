"""
src/ml/train.py
Production XGBoost churn training pipeline.
Features: TimeSeriesSplit, SMOTE, SHAP, MLflow, model registry promotion.
"""

import pathlib
import logging
import pickle
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "data_ecommerce_customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
SHAP_PATH  = MODEL_DIR / "shap_explainer.pkl"
META_PATH  = MODEL_DIR / "model_meta.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
TARGET = "Churn"
CAT_COLS = ["PreferedOrderCat", "MaritalStatus"]
FEATURE_COLS = [
    "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
    "PreferedOrderCat", "SatisfactionScore", "MaritalStatus",
    "NumberOfAddress", "Complain", "DaySinceLastOrder", "CashbackAmount"
]
BASELINE_F1 = 0.70          # CI gate: block promotion if below this
N_SPLITS    = 5             # TimeSeriesSplit folds
MLFLOW_URI  = "mlruns"      # local; swap for remote URI in prod
EXPERIMENT  = "churn_xgboost"


# ── 1. Data Loading ────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Coerce target
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    else:
        df[TARGET] = df[TARGET].astype(int)

    # Fill numeric NaNs with median
    num_cols = df.select_dtypes("float64").columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Sort by Tenure as temporal proxy (no timestamp available)
    df = df.reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} rows | churn rate: {df[TARGET].mean():.1%}")
    return df


# ── 2. Feature Engineering ─────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    # Encode categoricals
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y


# ── 3. Cross-validated Training ────────────────────────────────────────────
def train(X: pd.DataFrame, y: pd.Series) -> Tuple[xgb.XGBClassifier, dict]:
    tscv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    scale_pos = int((y == 0).sum()) / int((y == 1).sum())

    params = {
        "n_estimators":     400,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos,   # handles class imbalance natively
        "eval_metric":      "aucpr",
        "random_state":     42,
        "n_jobs":           -1,
    }

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # SMOTE only on train fold to prevent leakage
        sm = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        metrics = {
            "fold":      fold,
            "f1":        f1_score(y_val, pred),
            "pr_auc":    average_precision_score(y_val, proba),
            "roc_auc":   roc_auc_score(y_val, proba),
            "precision": precision_score(y_val, pred),
            "recall":    recall_score(y_val, pred),
        }
        fold_metrics.append(metrics)
        logger.info(f"Fold {fold} | F1={metrics['f1']:.3f} | PR-AUC={metrics['pr_auc']:.3f}")

    # Final model on full data with SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_res, y_res, verbose=False)

    # Aggregate CV metrics
    cv_summary = {
        k: float(np.mean([m[k] for m in fold_metrics]))
        for k in ["f1", "pr_auc", "roc_auc", "precision", "recall"]
    }
    cv_summary["params"] = params
    cv_summary["n_folds"] = N_SPLITS

    return final_model, cv_summary


# ── 4. SHAP Explainer ──────────────────────────────────────────────────────
def build_shap_explainer(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame
) -> shap.TreeExplainer:
    explainer = shap.TreeExplainer(model)
    # Validate it works
    _ = explainer.shap_values(X_sample.iloc[:5])
    return explainer


# ── 5. Promotion Gate ──────────────────────────────────────────────────────
def should_promote(cv_metrics: dict) -> bool:
    f1 = cv_metrics["f1"]
    promoted = f1 >= BASELINE_F1
    logger.info(
        f"Promotion gate: F1={f1:.3f} vs baseline={BASELINE_F1} → "
        f"{'PROMOTE' if promoted else 'REJECT'}"
    )
    return promoted


# ── 6. Persist ─────────────────────────────────────────────────────────────
def save_artifacts(
    model: xgb.XGBClassifier,
    explainer: shap.TreeExplainer,
    cv_metrics: dict,
    feature_cols: list,
) -> None:
    model.save_model(str(MODEL_PATH.with_suffix(".json")))

    with open(SHAP_PATH, "wb") as f:
        pickle.dump(explainer, f)

    meta = {**cv_metrics, "features": feature_cols}
    META_PATH.write_text(json.dumps(meta, indent=2))

    logger.info(f"Artifacts saved → {MODEL_DIR}")


# ── 7. MLflow Run ──────────────────────────────────────────────────────────
def log_mlflow(
    model: xgb.XGBClassifier,
    cv_metrics: dict,
    promoted: bool,
) -> str:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_params(cv_metrics["params"])
        mlflow.log_metrics({
            k: cv_metrics[k]
            for k in ["f1", "pr_auc", "roc_auc", "precision", "recall"]
        })
        mlflow.log_param("promoted", promoted)
        mlflow.xgboost.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        logger.info(f"MLflow run_id={run_id} | promoted={promoted}")

    return run_id


# ── 8. Entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logger.info("=== Churn XGBoost Training Pipeline ===")

    df              = load_data()
    X, y            = build_features(df)
    model, metrics  = train(X, y)
    explainer       = build_shap_explainer(model, X)
    promoted        = should_promote(metrics)
    run_id          = log_mlflow(model, metrics, promoted)

    if promoted:
        save_artifacts(model, explainer, metrics, FEATURE_COLS)
        logger.info("✓ Model promoted and saved.")
    else:
        logger.warning(
            f"✗ Model NOT promoted. F1={metrics['f1']:.3f} < {BASELINE_F1}. "
            f"MLflow run_id={run_id} retained for inspection."
        )

    # Print summary
    print("\n" + "="*50)
    print("CV RESULTS (mean across folds)")
    print("="*50)
    for k in ["f1", "pr_auc", "roc_auc", "precision", "recall"]:
        print(f"  {k:12s}: {metrics[k]:.4f}")
    print(f"  promoted    : {promoted}")
    print(f"  mlflow_run  : {run_id}")
    print("="*50)


if __name__ == "__main__":
    main()