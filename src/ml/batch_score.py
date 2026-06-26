"""
src/ml/batch_score.py
Daily batch scoring job — loads all customers, scores with XGBoost,
stores results to DuckDB, logs summary metrics.
Run: python src/ml/batch_score.py
"""

import logging
import pathlib
import pickle
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import duckdb
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

BASE_DIR    = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH   = BASE_DIR / "data" / "raw" / "data_ecommerce_customer_churn.csv"
MODEL_PATH  = BASE_DIR / "models" / "churn_model.pkl"
META_PATH   = BASE_DIR / "models" / "model_meta.json"
DB_PATH     = BASE_DIR / "data" / "batch_scores.duckdb"
REPORT_DIR  = BASE_DIR / "data" / "batch_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
    "PreferedOrderCat", "SatisfactionScore", "MaritalStatus",
    "NumberOfAddress", "Complain", "DaySinceLastOrder", "CashbackAmount"
]
CAT_COLS = ["PreferedOrderCat", "MaritalStatus"]

_CAT_VALUES = {
    "PreferedOrderCat": ["Laptop & Accessory", "Mobile Phone", "Mobile",
                          "Fashion", "Grocery", "Others"],
    "MaritalStatus":    ["Single", "Divorced", "Married"],
}


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Impute numeric nulls
    num_cols = df.select_dtypes("float64").columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    for col, vals in _CAT_VALUES.items():
        if col not in df.columns:
            continue
        le = LabelEncoder()
        le.fit(vals)
        df[col] = df[col].astype(str).apply(
            lambda v: le.transform([v])[0] if v in le.classes_ else 0
        )
    return df[FEATURE_COLS]


def _risk_level(prob: float) -> str:
    if prob >= 0.7:  return "HIGH"
    if prob >= 0.4:  return "MEDIUM"
    return "LOW"


def _init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS batch_scores (
            run_id          VARCHAR,
            scored_at       TIMESTAMP,
            customer_index  INTEGER,
            churn_probability DOUBLE,
            risk_level      VARCHAR,
            tenure          DOUBLE,
            satisfaction    DOUBLE,
            complain        INTEGER,
            days_inactive   DOUBLE,
            cashback        DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS batch_run_summary (
            run_id              VARCHAR PRIMARY KEY,
            scored_at           TIMESTAMP,
            total_customers     INTEGER,
            high_risk_count     INTEGER,
            medium_risk_count   INTEGER,
            low_risk_count      INTEGER,
            avg_churn_prob      DOUBLE,
            high_risk_pct       DOUBLE,
            model_f1            DOUBLE,
            model_pr_auc        DOUBLE
        )
    """)


def run_batch_scoring() -> dict:
    """
    Full batch scoring pipeline:
    1. Load data + model
    2. Score all customers
    3. Store to DuckDB
    4. Return summary metrics
    """
    run_id    = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    scored_at = datetime.now(timezone.utc)

    logger.info(f"=== Batch Scoring Start | {run_id} ===")

    # ── Load ──────────────────────────────────────────────────────────────
    df    = pd.read_csv(DATA_PATH)
    model = _load_model()
    meta  = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

    logger.info(f"Loaded {len(df):,} customers | model f1={meta.get('f1', 'N/A')}")

    # ── Score ─────────────────────────────────────────────────────────────
    X     = _encode_features(df)
    probs = model.predict_proba(X)[:, 1]
    risks = [_risk_level(p) for p in probs]

    results = df.copy()
    results["churn_probability"] = probs
    results["risk_level"]        = risks
    results["run_id"]            = run_id
    results["scored_at"]         = scored_at

    # ── Summary ───────────────────────────────────────────────────────────
    high   = int((results["risk_level"] == "HIGH").sum())
    medium = int((results["risk_level"] == "MEDIUM").sum())
    low    = int((results["risk_level"] == "LOW").sum())
    total  = len(results)
    avg_p  = float(np.mean(probs))

    summary = {
        "run_id":             run_id,
        "scored_at":          scored_at.isoformat(),
        "total_customers":    total,
        "high_risk_count":    high,
        "medium_risk_count":  medium,
        "low_risk_count":     low,
        "avg_churn_prob":     round(avg_p, 4),
        "high_risk_pct":      round(high / total * 100, 2),
        "model_f1":           round(meta.get("f1", 0.0), 4),
        "model_pr_auc":       round(meta.get("pr_auc", 0.0), 4),
    }

    # ── Store to DuckDB ───────────────────────────────────────────────────
    con = duckdb.connect(str(DB_PATH))
    _init_db(con)

    score_rows = pd.DataFrame({
        "run_id":             run_id,
        "scored_at":          scored_at,
        "customer_index":     results.index,
        "churn_probability":  probs,
        "risk_level":         risks,
        "tenure":             df.get("Tenure", pd.Series(dtype=float)),
        "satisfaction":       df.get("SatisfactionScore", pd.Series(dtype=float)),
        "complain":           df.get("Complain", pd.Series(dtype=int)),
        "days_inactive":      df.get("DaySinceLastOrder", pd.Series(dtype=float)),
        "cashback":           df.get("CashbackAmount", pd.Series(dtype=float)),
    })

    con.execute("INSERT INTO batch_scores SELECT * FROM score_rows")
    con.execute("""
        INSERT OR REPLACE INTO batch_run_summary VALUES (
            $run_id, $scored_at, $total_customers, $high_risk_count,
            $medium_risk_count, $low_risk_count, $avg_churn_prob,
            $high_risk_pct, $model_f1, $model_pr_auc
        )
    """, summary)
    con.close()

    # ── Save CSV snapshot ─────────────────────────────────────────────────
    out_cols = ["run_id", "scored_at", "churn_probability", "risk_level",
                "Tenure", "SatisfactionScore", "Complain"]
    snap_path = REPORT_DIR / f"{run_id}.csv"
    results[[c for c in out_cols if c in results.columns]].to_csv(snap_path, index=False)

    logger.info(
        f"Scored {total:,} | HIGH={high} ({summary['high_risk_pct']}%) | "
        f"avg_prob={avg_p:.3f} | saved → {snap_path.name}"
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    summary = run_batch_scoring()

    print("\n" + "=" * 50)
    print("BATCH SCORING SUMMARY")
    print("=" * 50)
    print(f"  Run ID         : {summary['run_id']}")
    print(f"  Total          : {summary['total_customers']:,}")
    print(f"  High risk      : {summary['high_risk_count']:,} ({summary['high_risk_pct']}%)")
    print(f"  Medium risk    : {summary['medium_risk_count']:,}")
    print(f"  Low risk       : {summary['low_risk_count']:,}")
    print(f"  Avg churn prob : {summary['avg_churn_prob']:.4f}")
    print(f"  Model F1       : {summary['model_f1']}")
    print(f"  Model PR-AUC   : {summary['model_pr_auc']}")
    print("=" * 50)