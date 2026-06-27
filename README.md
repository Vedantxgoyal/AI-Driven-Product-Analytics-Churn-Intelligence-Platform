# Churn Intelligence Platform

> Production-grade ML system for e-commerce customer churn prediction with real-time explainability, drift monitoring, and automated CI/CD.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Tests](https://img.shields.io/badge/Tests-61%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-purple)
![CI](https://github.com/Vedantxgoyal/AI-Driven-Product-Analytics-Churn-Intelligence-Platform/actions/workflows/Ci.yml/badge.svg)
---

## Overview

End-to-end ML platform predicting customer churn on an e-commerce dataset (3,941 customers, 17.1% churn rate). Built for production with temporal cross-validation, SHAP-based explanations, MLflow experiment tracking, Evidently drift detection, DuckDB batch scoring, and a React dashboard — all served from a single FastAPI application.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              React Dashboard (:8000/dashboard)       │
│   Overview │ Predict & Explain │ History │ Analytics │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│              FastAPI REST API (:8000)                │
│  /predict  /explain  /predict/batch  /analytics/*   │
└──────┬─────────────────────────────────┬────────────┘
       │                                 │
┌──────▼──────────┐           ┌──────────▼────────────┐
│   ML Layer      │           │   Data Layer           │
│  XGBoost model  │           │  SQLite (predictions)  │
│  SHAP explainer │           │  DuckDB (batch scores) │
│  MLflow registry│           │  CSV (raw data)        │
└─────────────────┘           └───────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│              Monitoring & Quality                    │
│  Great Expectations (data validation)                │
│  Evidently AI (feature + prediction drift)           │
└─────────────────────────────────────────────────────┘
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| F1 Score | **0.727** |
| PR-AUC | **0.828** |
| ROC-AUC | **0.951** |
| Recall | **0.895** |
| Precision | 0.613 |

**Training setup:** 5-fold Stratified K-Fold CV · SMOTE on train folds only · XGBoost with `scale_pos_weight` for class imbalance · MLflow experiment tracking · Automated promotion gate (F1 ≥ 0.70)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| API | FastAPI + Uvicorn |
| ML Model | XGBoost 2.0 |
| Explainability | SHAP (TreeExplainer) |
| Experiment Tracking | MLflow (SQLite backend) |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Analytics DB | DuckDB |
| ORM / Audit DB | SQLAlchemy + SQLite |
| Data Validation | Custom (pandas-based, 17 checks) |
| Drift Monitoring | Evidently AI 0.4.x |
| Frontend | React 18 + Recharts (no build step) |
| Testing | pytest (61 tests) |
| CI/CD | GitHub Actions |
| Containerisation | Docker (multi-stage) |

---

## Project Structure

```
AI CHURN/
├── app/                        # FastAPI application
│   ├── main.py                 # 16 endpoints + React dashboard
│   ├── model.py                # XGBoost + SHAP wrapper
│   ├── database.py             # SQLAlchemy ORM
│   └── schemas.py              # Pydantic request/response models
├── src/
│   ├── ml/
│   │   ├── train.py            # Training pipeline (XGBoost + SHAP + MLflow)
│   │   └── batch_score.py      # Daily batch scoring → DuckDB
│   ├── data/
│   │   ├── validate.py         # 17-check data quality gate
│   │   ├── duckdb_processor.py # OLAP feature computations
│   │   ├── hive_warehouse.py   # Medallion warehouse simulator
│   │   └── s3_simulator.py     # Data lake simulator
│   └── monitoring/
│       └── drift.py            # Evidently feature + prediction drift
├── frontend/
│   └── index.html              # React dashboard (single file, no build)
├── models/
│   ├── churn_model.pkl         # Trained XGBoost model
│   ├── shap_explainer.pkl      # SHAP TreeExplainer
│   └── model_meta.json         # CV metrics + feature list
├── data/
│   ├── raw/                    # Source CSV
│   ├── validation_reports/     # JSON quality reports
│   ├── drift_reports/          # Evidently HTML + JSON reports
│   └── batch_reports/          # Daily batch CSV snapshots
├── tests/
│   ├── conftest.py             # Session-scoped test client + StaticPool DB
│   ├── test_api.py             # 32 API tests
│   └── test_training.py        # 29 ML pipeline tests
├── .github/workflows/
│   └── ci.yml                  # Lint → Test → Model gate → Docker
├── Dockerfile                  # Multi-stage production build
├── pyproject.toml              # Dependencies + tool config
└── mlflow.db                   # MLflow experiment store
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Git

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/ai-churn.git
cd ai-churn
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install fastapi uvicorn sqlalchemy pandas numpy scikit-learn \
            xgboost shap mlflow imbalanced-learn duckdb \
            python-dotenv pydantic evidently pytest httpx
```

### 2. Train the model

```bash
python src/ml/train.py
```

Expected output:
```
CV RESULTS (mean across folds)
==================================================
  f1          : 0.7266
  pr_auc      : 0.8282
  roc_auc     : 0.9514
  promoted    : True
  mlflow_run  : <run_id>
==================================================
```

### 3. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the dashboard

```
http://localhost:8000/dashboard
```

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check (model + DB status) |
| `GET` | `/dashboard` | React dashboard |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/explain` | Prediction + SHAP explanation |
| `POST` | `/predict/batch` | Batch predictions (up to 1,000) |
| `GET` | `/predictions` | Prediction history (filterable) |
| `GET` | `/predictions/stats` | Aggregate statistics |
| `GET` | `/analytics/overview` | Full analytics report |
| `GET` | `/analytics/customer-counts` | Customer segmentation |
| `GET` | `/analytics/churn-by-satisfaction` | Churn by satisfaction score |
| `GET` | `/analytics/churn-by-tenure` | Churn by lifecycle segment |
| `GET` | `/analytics/churn-by-complaints` | Complaint impact analysis |
| `GET` | `/analytics/risk-segmentation` | Rule-based risk buckets |
| `GET` | `/docs` | Swagger UI |

### Example: Predict & Explain

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "tenure": 3,
      "warehouse_to_home": 10,
      "number_of_device_registered": 3,
      "preferred_order_cat": "Mobile Phone",
      "satisfaction_score": 1,
      "marital_status": "Single",
      "number_of_address": 2,
      "complain": 1,
      "days_since_last_order": 15,
      "cashback_amount": 150.0
    }
  }'
```

Response:
```json
{
  "churn_probability": 0.9578,
  "risk_level": "HIGH",
  "recommendation": "Immediate action: call customer, offer loyalty incentive",
  "explanation": {
    "expected_value": 1.7414,
    "top_5_drivers": [
      {"feature": "DaySinceLastOrder", "shap_value": -1.5604},
      {"feature": "Tenure",            "shap_value": -1.1664},
      {"feature": "Complain",          "shap_value":  0.9366},
      {"feature": "CashbackAmount",    "shap_value": -0.8793},
      {"feature": "SatisfactionScore", "shap_value": -0.7843}
    ]
  }
}
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ --import-mode=importlib -q

# API tests only
python -m pytest tests/test_api.py --import-mode=importlib -v

# Training pipeline tests only
python -m pytest tests/test_training.py --import-mode=importlib -v
```

Expected: **61 passed**

---

## ML Pipeline

### Training (`src/ml/train.py`)

```
load_data()
    └── CSV → DataFrame → coerce target → impute nulls

build_features()
    └── LabelEncode categoricals → select 10 feature columns

train()
    └── StratifiedKFold(n_splits=5)
        └── for each fold:
            ├── SMOTE on train split only (prevents leakage)
            ├── XGBoost with scale_pos_weight
            └── evaluate: F1, PR-AUC, ROC-AUC, Precision, Recall
    └── final model trained on full dataset + SMOTE

build_shap_explainer()
    └── TreeExplainer on final model

should_promote()
    └── mean CV F1 >= 0.70 → save artifacts

log_mlflow()
    └── params + metrics + model artifact → sqlite:///mlflow.db
```

### Batch Scoring (`src/ml/batch_score.py`)

```bash
python src/ml/batch_score.py
```

Scores all 3,941 customers, stores results in `data/batch_scores.duckdb` with run summary table. Output: CSV snapshot in `data/batch_reports/`.

### Data Validation (`src/data/validate.py`)

```bash
python src/data/validate.py
```

Runs 17 checks: row count, required columns, null checks, value ranges, categorical validity, churn rate sanity. Saves JSON report to `data/validation_reports/`.

### Drift Monitoring (`src/monitoring/drift.py`)

```bash
python src/monitoring/drift.py
```

Compares reference (70% of data) against current window using Evidently. Flags drift if >30% of features shift. Saves HTML + JSON reports to `data/drift_reports/`.

---

## Docker

```bash
# Build
docker build -t churn-intelligence:latest .

# Run
docker run -p 8000:8000 churn-intelligence:latest

# Access
open http://localhost:8000/dashboard
```

---

## CI/CD (GitHub Actions)

Pipeline on every push to `main` / PR:

```
1. Lint & Type Check    ruff + black --check
2. Data Validation      python src/data/validate.py
3. Tests + Coverage     pytest (61 tests)
4. Model Eval Gate      train → assert CV F1 >= 0.70 (blocks merge if fails)
5. Docker Build         multi-stage build (main branch only)
```

---

## MLflow

View experiment runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## Dataset

E-commerce customer churn dataset.

| Property | Value |
|----------|-------|
| Rows | 3,941 |
| Features | 10 |
| Target | Churn (binary) |
| Churn rate | 17.1% |
| Key features | Tenure, SatisfactionScore, Complain, DaySinceLastOrder, CashbackAmount |

---

## Features

| Feature | Type | Description |
|---------|------|-------------|
| Tenure | Float | Months as customer |
| WarehouseToHome | Float | Distance from warehouse (km) |
| NumberOfDeviceRegistered | Int | Devices on account |
| PreferedOrderCat | Categorical | Laptop / Mobile / Fashion / Grocery / Others |
| SatisfactionScore | Int | 1–5 rating |
| MaritalStatus | Categorical | Single / Married / Divorced |
| NumberOfAddress | Int | Saved delivery addresses |
| Complain | Binary | Filed complaint in last month |
| DaySinceLastOrder | Float | Days since most recent order |
| CashbackAmount | Float | Cashback received last month (₹) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Vedant Goyal**  
