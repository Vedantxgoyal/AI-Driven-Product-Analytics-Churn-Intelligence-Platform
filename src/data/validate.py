"""
src/data/validate.py
Data quality validation — pure pandas, no external GE dependency.
Run: python src/data/validate.py
"""

import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "data_ecommerce_customer_churn.csv"
REPORT_DIR = BASE_DIR / "data" / "validation_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _run_expectations(df: pd.DataFrame) -> Tuple[bool, List[Dict], Dict]:
    failures: List[Dict] = []
    total = passed_n = 0

    def check(name: str, result: bool, observed=None, kwargs=None):
        nonlocal total, passed_n
        total += 1
        if result:
            passed_n += 1
        else:
            failures.append({"expectation": name, "observed": observed, "kwargs": kwargs or {}})

    # Row count
    check("row_count_100_to_1M", 100 <= len(df) <= 1_000_000, observed=len(df))

    # Required columns
    required = {
        "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
        "PreferedOrderCat", "SatisfactionScore", "MaritalStatus",
        "NumberOfAddress", "Complain", "DaySinceLastOrder",
        "CashbackAmount", "Churn",
    }
    missing = required - set(df.columns)
    check("required_columns_present", len(missing) == 0, observed=list(missing))

    if missing:
        return False, failures, {"total": total, "passed": passed_n, "failed": total - passed_n}

    # Nulls — Tenure excluded (imputed during preprocessing)
    for col in ["Churn", "SatisfactionScore", "Complain"]:
        n = int(df[col].isnull().sum())
        check(f"no_nulls_{col}", n == 0, observed=n)

    tenure_null_pct = float(df["Tenure"].isnull().mean())
    check("tenure_null_pct_under_10pct", tenure_null_pct < 0.10, observed=round(tenure_null_pct, 4))

    # Ranges
    for col, lo, hi in [
        ("Tenure",                   0,   120),
        ("SatisfactionScore",        1,   5),
        ("Complain",                 0,   1),
        ("CashbackAmount",           0,   10_000),
        ("DaySinceLastOrder",        0,   365),
        ("NumberOfDeviceRegistered", 1,   10),
        ("NumberOfAddress",          1,   100),
        ("WarehouseToHome",          1,   200),
    ]:
        s = df[col].dropna()
        v = int(((s < lo) | (s > hi)).sum())
        check(f"range_{col}", v == 0, observed=v, kwargs={"min": lo, "max": hi})

    # Categoricals
    for col, valid in [
        ("PreferedOrderCat", {"Laptop & Accessory", "Mobile Phone", "Mobile", "Fashion", "Grocery", "Others"}),
        ("MaritalStatus",    {"Single", "Divorced", "Married"}),
    ]:
        bad = set(df[col].dropna().unique()) - valid
        check(f"valid_categories_{col}", len(bad) == 0, observed=list(bad))

    # Churn rate sanity
    rate = float(df["Churn"].mean())
    check("churn_rate_5pct_to_60pct", 0.05 <= rate <= 0.60, observed=round(rate, 4))

    stats = {"total": total, "passed": passed_n, "failed": total - passed_n}
    return len(failures) == 0, failures, stats


def validate_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "churn_data",
    raise_on_failure: bool = False,
) -> Tuple[bool, Dict]:
    df = df.copy()
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    passed, failures, stats = _run_expectations(df)

    report = {
        "dataset":          dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed":           passed,
        "total_checks":     stats["total"],
        "passed_checks":    stats["passed"],
        "failed_checks":    stats["failed"],
        "success_rate_pct": round(stats["passed"] / stats["total"] * 100, 2),
        "failures":         failures,
        "row_count":        len(df),
        "churn_rate_pct":   round(float(df["Churn"].mean()) * 100, 2) if "Churn" in df.columns else None,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"{dataset_name}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.log(
        logging.INFO if passed else logging.WARNING,
        f"Validation {'PASSED' if passed else 'FAILED'} | "
        f"{stats['passed']}/{stats['total']} checks | "
        f"churn={report['churn_rate_pct']}%"
    )
    for f in failures:
        logger.warning(f"  FAIL: {f['expectation']} | observed={f['observed']}")

    if raise_on_failure and not passed:
        raise ValueError(f"Data validation failed: {stats['failed']} checks. Report: {report_path}")

    return passed, report


def validate_file(
    path: pathlib.Path = DATA_PATH,
    raise_on_failure: bool = False,
) -> Tuple[bool, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return validate_dataframe(pd.read_csv(path), dataset_name=path.stem,
                              raise_on_failure=raise_on_failure)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    passed, report = validate_file()

    print("\n" + "=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)
    print(f"  Status       : {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"  Rows         : {report['row_count']:,}")
    print(f"  Checks       : {report['passed_checks']}/{report['total_checks']}")
    print(f"  Success rate : {report['success_rate_pct']}%")
    print(f"  Churn rate   : {report['churn_rate_pct']}%")
    if report["failures"]:
        print("\n  Failed checks:")
        for f in report["failures"]:
            print(f"    - {f['expectation']} | observed={f['observed']}")
    print("=" * 50)