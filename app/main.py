"""
app/main.py - Churn Intelligence Platform API v2.0
XGBoost + SHAP + Analytics + Static Dashboard
"""

import logging
import pathlib
import time
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc, text

from app.schemas import (
    PredictRequest, BatchPredictRequest,
    PredictionResult, PredictResponse, BatchPredictResponse,
    HealthCheckResponse, PredictionHistoryResponse, PredictionRecord
)
from app.database import get_db, PredictionRecord as PredictionDB, engine, Base
from app.model import initialize_model, get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Churn Intelligence Platform",
    description="XGBoost churn prediction with SHAP explainability",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Serve dashboard static files
FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Analytics Engine ───────────────────────────────────────────────────────

class AnalyticsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if self.df["Churn"].dtype == "object":
            self.df["Churn"] = (self.df["Churn"] == "Yes").astype(int)
        self.churn_df  = self.df[self.df["Churn"] == 1]
        self.active_df = self.df[self.df["Churn"] == 0]

    def overall_churn_rate(self) -> float:
        return len(self.churn_df) / len(self.df)

    def customer_count(self) -> Dict:
        return {"total": len(self.df), "active": len(self.active_df), "churned": len(self.churn_df)}

    def avg_tenure(self) -> Dict:
        return {
            "overall": float(self.df["Tenure"].mean()),
            "churned": float(self.churn_df["Tenure"].mean()),
            "active":  float(self.active_df["Tenure"].mean()),
        }

    def avg_satisfaction(self) -> Dict:
        return {
            "overall": float(self.df["SatisfactionScore"].mean()),
            "churned": float(self.churn_df["SatisfactionScore"].mean()),
            "active":  float(self.active_df["SatisfactionScore"].mean()),
        }

    def churn_by_satisfaction(self) -> Dict:
        return {
            int(s): float((g["Churn"] == 1).sum() / len(g))
            for s, g in self.df.groupby("SatisfactionScore")
        }

    def churn_by_tenure_segment(self) -> Dict:
        def seg(t):
            if t <= 3:  return "New (0-3m)"
            if t <= 12: return "Growth (4-12m)"
            return "Mature (13+m)"
        df = self.df.copy()
        df["seg"] = df["Tenure"].apply(seg)
        return {s: float((g["Churn"] == 1).sum() / len(g)) for s, g in df.groupby("seg")}

    def churn_by_complaint(self) -> Dict:
        result = {}
        for status, g in self.df.groupby("Complain"):
            churned = int((g["Churn"] == 1).sum())
            result[str(int(status))] = {
                "churn_rate": float(churned / len(g)),
                "count": int(len(g)),
                "churned_count": churned,
            }
        return result

    def risk_segmentation(self) -> Dict:
        df = self.df.copy()
        df["RiskScore"] = (
            (df["SatisfactionScore"] <= 2).astype(int) * 25 +
            (df["DaySinceLastOrder"] > 10).astype(int) * 25 +
            (df["Complain"] == 1).astype(int) * 20 +
            (df["Tenure"] < 3).astype(int) * 30
        )
        return {
            "high_risk":   int((df["RiskScore"] >= 75).sum()),
            "medium_risk": int(((df["RiskScore"] >= 40) & (df["RiskScore"] < 75)).sum()),
            "low_risk":    int((df["RiskScore"] < 40).sum()),
        }

    def get_full_report(self) -> Dict:
        return {
            "churn_rate":            self.overall_churn_rate(),
            "customer_counts":       self.customer_count(),
            "avg_tenure":            self.avg_tenure(),
            "avg_satisfaction":      self.avg_satisfaction(),
            "churn_by_satisfaction": self.churn_by_satisfaction(),
            "churn_by_tenure":       self.churn_by_tenure_segment(),
            "churn_by_complaint":    self.churn_by_complaint(),
            "risk_segmentation":     self.risk_segmentation(),
        }


analytics_engine = None
analytics_data   = None


def load_analytics_data() -> bool:
    global analytics_engine, analytics_data
    try:
        BASE_DIR  = pathlib.Path(__file__).resolve().parent.parent
        data_path = BASE_DIR / "data" / "raw" / "data_ecommerce_customer_churn.csv"
        if not data_path.exists():
            logger.warning(f"Data not found: {data_path}")
            return False
        analytics_data = pd.read_csv(data_path)
        float_cols = analytics_data.select_dtypes("float64").columns.tolist()
        analytics_data[float_cols] = analytics_data[float_cols].fillna(
            analytics_data[float_cols].median()
        )
        analytics_engine = AnalyticsEngine(analytics_data)
        logger.info(f"✓ Analytics loaded: {len(analytics_data):,} customers")
        return True
    except Exception as e:
        logger.error(f"Analytics load failed: {e}")
        return False


# ── Startup / Shutdown ─────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Churn Intelligence Platform v2.0")
    logger.info("=" * 60)
    load_analytics_data()
    if not initialize_model():
        logger.warning("⚠ Model not loaded — predictions unavailable")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down.")


# ── Helpers ────────────────────────────────────────────────────────────────

def _require_analytics():
    if analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics not available")


def _require_model():
    m = get_model()
    if not m.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    return m


def _store_prediction(db, features, prob, risk, rec):
    row = PredictionDB(
        churn_probability=prob, churn_risk_level=risk,
        recommendation=rec, is_high_risk=(risk == "HIGH"),
        tenure=features["tenure"],
        warehouse_to_home=features["warehouse_to_home"],
        number_of_device_registered=features["number_of_device_registered"],
        preferred_order_cat=features["preferred_order_cat"],
        satisfaction_score=features["satisfaction_score"],
        marital_status=features["marital_status"],
        number_of_address=features["number_of_address"],
        complain=features["complain"],
        days_since_last_order=features["days_since_last_order"],
        cashback_amount=features["cashback_amount"],
    )
    db.add(row)
    return row


# ── Info / Health / Dashboard ──────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name":      "Churn Intelligence Platform",
        "version":   "2.0.0",
        "docs":      "/docs",
        "dashboard": "/dashboard",
    }


@app.get("/dashboard", tags=["Info"], include_in_schema=False)
async def dashboard():
    """Serve React dashboard."""
    path = FRONTEND_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found. Place index.html in frontend/")
    return FileResponse(str(path))


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    model = get_model()
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    healthy = model.is_loaded() and db_ok and analytics_engine is not None
    return HealthCheckResponse(
        status="healthy" if healthy else "degraded",
        model_loaded=model.is_loaded(),
        database_connected=db_ok,
        version="2.0.0",
    )


# ── Analytics ──────────────────────────────────────────────────────────────

@app.get("/analytics/overview", tags=["Analytics"])
async def analytics_overview():
    _require_analytics()
    return {"success": True, "data": analytics_engine.get_full_report()}


@app.get("/analytics/customer-counts", tags=["Analytics"])
async def customer_counts():
    _require_analytics()
    counts = analytics_engine.customer_count()
    return {**counts, "churn_rate_percentage": round(analytics_engine.overall_churn_rate() * 100, 2)}


@app.get("/analytics/churn-by-satisfaction", tags=["Analytics"])
async def churn_by_satisfaction():
    _require_analytics()
    return {"success": True, "data": analytics_engine.churn_by_satisfaction()}


@app.get("/analytics/churn-by-tenure", tags=["Analytics"])
async def churn_by_tenure():
    _require_analytics()
    return {"success": True, "data": analytics_engine.churn_by_tenure_segment()}


@app.get("/analytics/churn-by-complaints", tags=["Analytics"])
async def churn_by_complaints():
    _require_analytics()
    data = analytics_engine.churn_by_complaint()
    nr, wr = data["0"]["churn_rate"], data["1"]["churn_rate"]
    return {
        "no_complaints":   {"churn_rate": round(nr * 100, 2), "count": data["0"]["count"]},
        "with_complaints": {"churn_rate": round(wr * 100, 2), "count": data["1"]["count"],
                            "multiplier": round(wr / nr, 2)},
    }


@app.get("/analytics/risk-segmentation", tags=["Analytics"])
async def risk_segmentation():
    _require_analytics()
    risk  = analytics_engine.risk_segmentation()
    total = sum(risk.values())
    return {k: {"count": v, "percentage": round(v / total * 100, 2)} for k, v in risk.items()}


# ── Predictions ────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    model    = _require_model()
    features = request.customer.dict(by_alias=False)
    prob, risk, rec = model.predict_single(features)
    row = _store_prediction(db, features, prob, risk, rec)
    db.commit()
    logger.info(f"Predict id={row.id} risk={risk} prob={prob:.2%}")
    return PredictResponse(
        success=True,
        data=PredictionResult(churn_probability=prob, churn_risk_level=risk, recommendation=rec),
    )


@app.post("/explain", tags=["Predictions"])
async def explain(request: PredictRequest):
    """SHAP explanation for single customer."""
    model    = _require_model()
    features = request.customer.dict(by_alias=False)
    prob, risk, rec = model.predict_single(features)
    explanation     = model.explain_single(features)
    return {
        "churn_probability": round(prob, 4),
        "risk_level":        risk,
        "recommendation":    rec,
        "explanation":       explanation,
    }


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictRequest, db: Session = Depends(get_db)):
    model = _require_model()
    t0    = time.time()
    preds, high_risk_count = [], 0
    for customer in request.customers:
        features = customer.dict(by_alias=False)
        prob, risk, rec = model.predict_single(features)
        preds.append(PredictionResult(
            churn_probability=prob, churn_risk_level=risk, recommendation=rec
        ))
        if risk == "HIGH":
            high_risk_count += 1
        _store_prediction(db, features, prob, risk, rec)
    db.commit()
    ms = round((time.time() - t0) * 1000, 2)
    logger.info(f"Batch {len(preds)} customers | {high_risk_count} high-risk | {ms}ms")
    return BatchPredictResponse(
        success=True, processed_count=len(preds),
        high_risk_count=high_risk_count, predictions=preds,
        processing_time_ms=ms,
    )


# ── History ────────────────────────────────────────────────────────────────

@app.get("/predictions", response_model=PredictionHistoryResponse, tags=["History"])
async def get_predictions(
    days: int = 7, high_risk_only: bool = False,
    limit: int = 100, db: Session = Depends(get_db)
):
    limit = min(limit, 1000)
    since = datetime.utcnow() - timedelta(days=days)
    q     = db.query(PredictionDB).filter(PredictionDB.created_at >= since)
    if high_risk_only:
        q = q.filter(PredictionDB.is_high_risk == True)
    rows  = q.order_by(desc(PredictionDB.created_at)).limit(limit).all()
    total = len(rows)
    avg   = sum(r.churn_probability for r in rows) / total if total else 0
    return PredictionHistoryResponse(
        total_predictions=total,
        high_risk_count=sum(1 for r in rows if r.is_high_risk),
        average_churn_probability=round(avg, 4),
        predictions=[PredictionRecord.model_validate(r) for r in rows],
    )


@app.get("/predictions/stats", tags=["Analytics"])
async def get_stats(db: Session = Depends(get_db)):
    rows = db.query(PredictionDB).all()
    if not rows:
        return {"total_predictions": 0, "high_risk_count": 0,
                "medium_risk_count": 0, "low_risk_count": 0,
                "average_churn_probability": 0, "high_risk_percentage": 0}
    total = len(rows)
    high  = sum(1 for r in rows if r.churn_risk_level == "HIGH")
    med   = sum(1 for r in rows if r.churn_risk_level == "MEDIUM")
    low   = sum(1 for r in rows if r.churn_risk_level == "LOW")
    avg   = sum(r.churn_probability for r in rows) / total
    return {
        "total_predictions":         total,
        "high_risk_count":           high,
        "medium_risk_count":         med,
        "low_risk_count":            low,
        "average_churn_probability": round(avg, 4),
        "high_risk_percentage":      round(high / total * 100, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)