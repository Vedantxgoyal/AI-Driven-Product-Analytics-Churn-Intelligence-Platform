"""
src/monitoring/drift.py
Feature + prediction drift detection using Evidently.
Run: python src/monitoring/drift.py
Called from: batch scoring job or scheduled monitor.
"""

import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Dict, Tuple

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetMissingValuesMetric

logger = logging.getLogger(__name__)

BASE_DIR    = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH   = BASE_DIR / "data" / "raw" / "data_ecommerce_customer_churn.csv"
DRIFT_DIR   = BASE_DIR / "data" / "drift_reports"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
    "SatisfactionScore", "NumberOfAddress", "Complain",
    "DaySinceLastOrder", "CashbackAmount",
]
TARGET_COL = "Churn"

# Drift threshold — flag if >30% of features drift
DRIFT_SHARE_THRESHOLD = 0.30


# ── Core ───────────────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce target, drop categoricals (Evidently handles numerics cleanly)."""
    df = df.copy()
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
        df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    cols = FEATURE_COLS + ([TARGET_COL] if TARGET_COL in df.columns else [])
    return df[cols].dropna()


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_tag: str = "drift",
) -> Tuple[bool, Dict]:
    """
    Compare current data against reference for feature + target drift.

    Args:
        reference:  Training / baseline dataset.
        current:    New / production dataset.
        report_tag: Label used in saved report filename.

    Returns:
        (drift_detected: bool, summary: dict)
    """
    ref = _prep(reference)
    cur = _prep(current)

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DatasetMissingValuesMetric(),
    ])
    report.run(reference_data=ref, current_data=cur)

    result     = report.as_dict()
    metrics    = result["metrics"]

    # ── Parse DataDriftPreset output ──
    drift_meta      = next(m for m in metrics if m["metric"] == "DatasetDriftMetric")
    drift_share     = float(drift_meta["result"]["share_of_drifted_columns"])
    drifted_cols    = int(drift_meta["result"]["number_of_drifted_columns"])
    total_cols      = int(drift_meta["result"]["number_of_columns"])
    dataset_drifted = bool(drift_meta["result"]["dataset_drift"])

    # ── Parse per-column drift ──
    col_drift_meta = next(m for m in metrics if m["metric"] == "DataDriftTable")
    col_details = {
        col: {
            "drifted":    info["drift_detected"],
            "p_value":    round(info.get("p_value", 0.0), 4),
            "statistic":  round(info.get("statistic", 0.0), 4),
        }
        for col, info in col_drift_meta["result"]["drift_by_columns"].items()
    }

    # ── Parse missing values ──
    missing_meta    = next(m for m in metrics if m["metric"] == "DatasetMissingValuesMetric")
    missing_current = float(missing_meta["result"]["current"]["share_of_missing_values"])

    drift_detected = drift_share >= DRIFT_SHARE_THRESHOLD

    summary = {
        "report_tag":            report_tag,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "drift_detected":        drift_detected,
        "dataset_drifted":       dataset_drifted,
        "drift_share":           round(drift_share, 4),
        "drifted_columns":       drifted_cols,
        "total_columns":         total_cols,
        "drift_threshold":       DRIFT_SHARE_THRESHOLD,
        "missing_value_share":   round(missing_current, 4),
        "reference_rows":        len(ref),
        "current_rows":          len(cur),
        "column_drift":          col_details,
    }

    # Save JSON report
    ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = DRIFT_DIR / f"{report_tag}_{ts}.json"
    report_path.write_text(json.dumps(summary, indent=2))

    # Save HTML report
    html_path = DRIFT_DIR / f"{report_tag}_{ts}.html"
    report.save_html(str(html_path))

    level = logging.WARNING if drift_detected else logging.INFO
    logger.log(level,
        f"Drift {'DETECTED' if drift_detected else 'OK'} | "
        f"{drifted_cols}/{total_cols} features drifted | "
        f"share={drift_share:.1%} | threshold={DRIFT_SHARE_THRESHOLD:.0%}"
    )
    for col, info in col_details.items():
        if info["drifted"]:
            logger.warning(f"  DRIFT: {col} | p={info['p_value']}")

    return drift_detected, summary


# ── Simulate drift for testing ─────────────────────────────────────────────

def _simulate_current(df: pd.DataFrame, drift_factor: float = 0.3) -> pd.DataFrame:
    """
    Inject synthetic drift into numeric columns for local testing.
    Not used in production — replace with real current-window data.
    """
    import numpy as np
    cur = df.copy()
    rng = np.random.default_rng(42)
    for col in FEATURE_COLS:
        if col in cur.columns:
            noise = rng.normal(0, cur[col].std() * drift_factor, size=len(cur))
            cur[col] = cur[col] + noise
    return cur


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df  = pd.read_csv(DATA_PATH)
    ref = df.iloc[:int(len(df) * 0.7)]   # 70% as reference (training window)
    cur = _simulate_current(df.iloc[int(len(df) * 0.7):])  # 30% with injected drift

    drift_detected, summary = detect_drift(ref, cur, report_tag="baseline_test")

    print("\n" + "=" * 50)
    print("DRIFT MONITORING REPORT")
    print("=" * 50)
    print(f"  Status         : {'⚠ DRIFT DETECTED' if drift_detected else '✓ NO DRIFT'}")
    print(f"  Drifted cols   : {summary['drifted_columns']}/{summary['total_columns']}")
    print(f"  Drift share    : {summary['drift_share']:.1%}")
    print(f"  Threshold      : {summary['drift_threshold']:.0%}")
    print(f"  Missing values : {summary['missing_value_share']:.1%}")
    print(f"  Reference rows : {summary['reference_rows']:,}")
    print(f"  Current rows   : {summary['current_rows']:,}")
    if any(v["drifted"] for v in summary["column_drift"].values()):
        print("\n  Drifted features:")
        for col, info in summary["column_drift"].items():
            if info["drifted"]:
                print(f"    - {col} | p={info['p_value']}")
    print("=" * 50)
    print(f"\n  HTML report → data/drift_reports/")