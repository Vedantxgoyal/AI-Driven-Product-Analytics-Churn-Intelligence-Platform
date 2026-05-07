"""
main.py - FastAPI application integrating Phase 1 Analytics + REST API
Combines data pipeline, analytics, and ML predictions
"""

import logging
import time
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
import pandas as pd
import numpy as np

# Phase 1 Imports
from app.schemas import (
    CustomerFeatures, PredictRequest, BatchPredictRequest,
    PredictionResult, PredictResponse, BatchPredictResponse,
    HealthCheckResponse, PredictionHistoryResponse, PredictionRecord
)
from app.database import get_db, PredictionRecord as PredictionDB, engine, Base
from app.model import initialize_model, get_model

# ============ LOGGING ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CREATE TABLES ============
Base.metadata.create_all(bind=engine)

# ============ FASTAPI APP ============
app = FastAPI(
    title="Churn Intelligence Platform",
    description="AI-powered REST API combining Phase 1 Analytics + ML Predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============ CORS ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ PHASE 1 ANALYTICS LAYER ============

class AnalyticsEngine:
    """Compute business metrics for churn analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Handle both 0/1 and Yes/No formats
        if self.df['Churn'].dtype == 'object':
            self.df['Churn'] = (self.df['Churn'] == 'Yes').astype(int)
        
        self.churn_df = self.df[self.df['Churn'] == 1]
        self.active_df = self.df[self.df['Churn'] == 0]
    
    def overall_churn_rate(self) -> float:
        """Percentage of customers who churned"""
        return len(self.churn_df) / len(self.df)
    
    def customer_count(self) -> Dict[str, int]:
        """Customer counts"""
        return {
            'total': len(self.df),
            'active': len(self.active_df),
            'churned': len(self.churn_df)
        }
    
    def avg_tenure(self) -> Dict[str, float]:
        """Average months as customer"""
        return {
            'overall': float(self.df['Tenure'].mean()),
            'churned': float(self.churn_df['Tenure'].mean()),
            'active': float(self.active_df['Tenure'].mean())
        }
    
    def avg_satisfaction(self) -> Dict[str, float]:
        """Average satisfaction score (1-5)"""
        return {
            'overall': float(self.df['SatisfactionScore'].mean()),
            'churned': float(self.churn_df['SatisfactionScore'].mean()),
            'active': float(self.active_df['SatisfactionScore'].mean())
        }
    
    def churn_by_satisfaction(self) -> Dict[int, float]:
        """Churn rate by satisfaction score"""
        result = {}
        for score in sorted(self.df['SatisfactionScore'].unique()):
            score_df = self.df[self.df['SatisfactionScore'] == score]
            churn_rate = (score_df['Churn'] == 1).sum() / len(score_df)
            result[int(score)] = float(churn_rate)
        return result
    
    def churn_by_tenure_segment(self) -> Dict[str, float]:
        """Churn rate by customer lifecycle"""
        
        def categorize_tenure(t):
            if t <= 3:
                return 'New (0-3m)'
            elif t <= 12:
                return 'Growth (4-12m)'
            else:
                return 'Mature (13+m)'
        
        self.df['TenureSegment'] = self.df['Tenure'].apply(categorize_tenure)
        
        result = {}
        for segment in ['New (0-3m)', 'Growth (4-12m)', 'Mature (13+m)']:
            segment_df = self.df[self.df['TenureSegment'] == segment]
            if len(segment_df) > 0:
                churn_rate = (segment_df['Churn'] == 1).sum() / len(segment_df)
                result[segment] = float(churn_rate)
        
        return result
    
    def churn_by_complaint(self) -> Dict[str, Dict]:
        """Churn rate by complaint status"""
        result = {}
        
        for complain_status in [0, 1]:
            complaint_df = self.df[self.df['Complain'] == complain_status]
            churn_count = (complaint_df['Churn'] == 1).sum()
            
            result[str(complain_status)] = {
                'churn_rate': float(churn_count / len(complaint_df)) if len(complaint_df) > 0 else 0,
                'count': int(len(complaint_df)),
                'churned_count': int(churn_count)
            }
        
        return result
    
    def risk_segmentation(self) -> Dict[str, int]:
        """Segment customers by risk level"""
        
        self.df['RiskScore'] = 0
        
        # Low satisfaction (1-2): 25 points
        self.df.loc[self.df['SatisfactionScore'] <= 2, 'RiskScore'] += 25
        
        # High days inactive: 25 points
        self.df.loc[self.df['DaySinceLastOrder'] > 10, 'RiskScore'] += 25
        
        # Has complained: 20 points
        self.df.loc[self.df['Complain'] == 1, 'RiskScore'] += 20
        
        # New customer: 30 points
        self.df.loc[self.df['Tenure'] < 3, 'RiskScore'] += 30
        
        return {
            'high_risk (75+)': int((self.df['RiskScore'] >= 75).sum()),
            'medium_risk (40-74)': int(((self.df['RiskScore'] >= 40) & (self.df['RiskScore'] < 75)).sum()),
            'low_risk (<40)': int((self.df['RiskScore'] < 40).sum())
        }
    
    def get_full_report(self) -> Dict:
        """Generate complete analytics report"""
        
        return {
            'churn_rate': self.overall_churn_rate(),
            'customer_counts': self.customer_count(),
            'avg_tenure': self.avg_tenure(),
            'avg_satisfaction': self.avg_satisfaction(),
            'churn_by_satisfaction': self.churn_by_satisfaction(),
            'churn_by_tenure_segment': self.churn_by_tenure_segment(),
            'churn_by_complaint': self.churn_by_complaint(),
            'risk_segmentation': self.risk_segmentation()
        }


# Global analytics engine instance
analytics_engine = None
analytics_data = None


def load_analytics_data():
    """Load and initialize analytics on startup"""
    global analytics_engine, analytics_data
    
    try:
        # Load data
        data_path = r"D:\Projects\AI CHURN\churn_api\data\raw\data_ecommerce_customer_churn.csv"
        if not os.path.exists(data_path):
            logger.warning(f"⚠ Analytics data not found: {data_path}")
            return False
        
        analytics_data = pd.read_csv(data_path)
        
        # Handle missing values (Phase 1)
        for col in analytics_data.columns:
            if analytics_data[col].dtype in ['float64']:
                analytics_data[col].fillna(analytics_data[col].median(), inplace=True)
        
        # Initialize analytics
        analytics_engine = AnalyticsEngine(analytics_data)
        
        logger.info(f"✓ Analytics loaded: {len(analytics_data):,} customers")
        return True
    
    except Exception as e:
        logger.error(f"Failed to load analytics: {str(e)}")
        return False


# ============ STARTUP & SHUTDOWN ============
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("="*70)
    logger.info("🚀 Starting Churn Intelligence Platform")
    logger.info("="*70)
    
    # Load analytics
    logger.info("[1/2] Loading Phase 1 Analytics...")
    analytics_ok = load_analytics_data()
    if analytics_ok:
        logger.info("✓ Phase 1 Analytics ready")
    else:
        logger.warning("⚠ Analytics not available")
    
    # Load ML model
    logger.info("[2/2] Loading ML Model...")
    success = initialize_model()
    if success:
        logger.info("✓ ML Model loaded successfully")
    else:
        logger.warning("⚠ Model not loaded - predictions will fail")
    
    logger.info("="*70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Churn Intelligence Platform")


# ============ ROOT & INFO ENDPOINTS ============

@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Churn Intelligence Platform",
        "version": "1.0.0",
        "description": "Phase 1 Analytics + ML Predictions",
        "docs": "/docs",
        "health": "/health",
        "analytics": "/analytics/overview"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    model = get_model()
    model_loaded = model.is_loaded()
    analytics_loaded = analytics_engine is not None
    
    # Check database
    db_connected = False
    try:
        db.execute("SELECT 1")
        db_connected = True
    except:
        pass
    
    return HealthCheckResponse(
        status="healthy" if (model_loaded and db_connected and analytics_loaded) else "degraded",
        model_loaded=model_loaded,
        database_connected=db_connected,
        version="1.0.0"
    )


# ============ PHASE 1 ANALYTICS ENDPOINTS ============

@app.get("/analytics/overview", tags=["Analytics"])
async def analytics_overview():
    """Get Phase 1 Analytics overview"""
    
    if analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics not available")
    
    try:
        report = analytics_engine.get_full_report()
        return {
            "success": True,
            "data": report
        }
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/customer-counts", tags=["Analytics"])
async def customer_counts():
    """Get customer segmentation"""
    
    if analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics not available")
    
    counts = analytics_engine.customer_count()
    churn_rate = analytics_engine.overall_churn_rate()
    
    return {
        "total_customers": counts['total'],
        "active_customers": counts['active'],
        "churned_customers": counts['churned'],
        "churn_rate_percentage": round(churn_rate * 100, 2)
    }


@app.get("/analytics/churn-by-satisfaction", tags=["Analytics"])
async def churn_by_satisfaction():
    """Get churn rate by satisfaction score"""
    
    if analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics not available")
    
    data = analytics_engine.churn_by_satisfaction()
    return {
        "success": True,
        "data": data
    }


@app.get("/analytics/churn-by-tenure", tags=["Analytics"])
async def churn_by_tenure():
    """Get churn rate by customer lifecycle"""
    
    if analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics not available")
    
    data = analytics_engine.churn_by_tenure_segment()
    return {
        "success": True,
        "data": data
    }


@app.get("/analytics/churn-by-complaints", tags=["Analytics"])
async def churn_by_complaints():
    """Get churn impact of complaints"""
    
    if analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics not available")
    
    data = analytics_engine.churn_by_complaint()
    no_complaint_rate = data['0']['churn_rate']
    with_complaint_rate = data['1']['churn_rate']
    
    return {
        "no_complaints": {
            "churn_rate": round(no_complaint_rate * 100, 2),
            "customer_count": data['0']['count']
        },
        "with_complaints": {
            "churn_rate": round(with_complaint_rate * 100, 2),
            "customer_count": data['1']['count'],
            "multiplier": round(with_complaint_rate / no_complaint_rate, 2)
        }
    }


@app.get("/analytics/risk-segmentation", tags=["Analytics"])
async def risk_segmentation():
    """Get customer risk segmentation"""
    
    if analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics not available")
    
    risk = analytics_engine.risk_segmentation()
    total = sum(risk.values())
    
    return {
        "high_risk": {
            "count": risk['high_risk (75+)'],
            "percentage": round((risk['high_risk (75+)'] / total) * 100, 2)
        },
        "medium_risk": {
            "count": risk['medium_risk (40-74)'],
            "percentage": round((risk['medium_risk (40-74)'] / total) * 100, 2)
        },
        "low_risk": {
            "count": risk['low_risk (<40)'],
            "percentage": round((risk['low_risk (<40)'] / total) * 100, 2)
        }
    }


# ============ ML PREDICTION ENDPOINTS ============

@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
async def predict(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    """Predict churn for single customer"""
    
    model = get_model()
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert Pydantic to dict
        features = request.customer.dict(by_alias=False)
        
        # Predict
        churn_prob, risk_level, recommendation = model.predict_single(features)
        
        # Store prediction
        prediction_record = PredictionDB(
            churn_probability=churn_prob,
            churn_risk_level=risk_level,
            recommendation=recommendation,
            is_high_risk=(risk_level == "HIGH"),
            tenure=features['tenure'],
            warehouse_to_home=features['warehouse_to_home'],
            number_of_device_registered=features['number_of_device_registered'],
            preferred_order_cat=features['preferred_order_cat'],
            satisfaction_score=features['satisfaction_score'],
            marital_status=features['marital_status'],
            number_of_address=features['number_of_address'],
            complain=features['complain'],
            days_since_last_order=features['days_since_last_order'],
            cashback_amount=features['cashback_amount']
        )
        
        db.add(prediction_record)
        db.commit()
        
        logger.info(f"Prediction: ID={prediction_record.id}, Risk={risk_level}, Prob={churn_prob:.2%}")
        
        return PredictResponse(
            success=True,
            data=PredictionResult(
                churn_probability=churn_prob,
                churn_risk_level=risk_level,
                recommendation=recommendation
            )
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Predictions"])
async def predict_batch(
    request: BatchPredictRequest,
    db: Session = Depends(get_db)
):
    """Predict churn for multiple customers (1-1000)"""
    
    model = get_model()
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    
    start_time = time.time()
    
    try:
        predictions = []
        high_risk_count = 0
        
        for customer in request.customers:
            features = customer.dict(by_alias=False)
            
            # Predict
            churn_prob, risk_level, recommendation = model.predict_single(features)
            
            predictions.append(PredictionResult(
                churn_probability=churn_prob,
                churn_risk_level=risk_level,
                recommendation=recommendation
            ))
            
            # Store in database
            is_high_risk = (risk_level == "HIGH")
            if is_high_risk:
                high_risk_count += 1
            
            prediction_record = PredictionDB(
                churn_probability=churn_prob,
                churn_risk_level=risk_level,
                recommendation=recommendation,
                is_high_risk=is_high_risk,
                tenure=features['tenure'],
                warehouse_to_home=features['warehouse_to_home'],
                number_of_device_registered=features['number_of_device_registered'],
                preferred_order_cat=features['preferred_order_cat'],
                satisfaction_score=features['satisfaction_score'],
                marital_status=features['marital_status'],
                number_of_address=features['number_of_address'],
                complain=features['complain'],
                days_since_last_order=features['days_since_last_order'],
                cashback_amount=features['cashback_amount']
            )
            
            db.add(prediction_record)
        
        db.commit()
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch: {len(predictions)} customers, {high_risk_count} high-risk in {processing_time:.0f}ms")
        
        return BatchPredictResponse(
            success=True,
            processed_count=len(predictions),
            high_risk_count=high_risk_count,
            predictions=predictions,
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/predictions", response_model=PredictionHistoryResponse, tags=["History"])
async def get_predictions(
    days: int = 7,
    high_risk_only: bool = False,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Fetch prediction history"""
    
    if limit > 1000:
        limit = 1000
    
    # Query
    since_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(PredictionDB).filter(
        PredictionDB.created_at >= since_date
    )
    
    if high_risk_only:
        query = query.filter(PredictionDB.is_high_risk == True)
    
    predictions = query.order_by(desc(PredictionDB.created_at)).limit(limit).all()
    
    total_count = len(predictions)
    high_risk_count = sum(1 for p in predictions if p.is_high_risk)
    avg_churn = sum(p.churn_probability for p in predictions) / total_count if total_count > 0 else 0
    
    logger.info(f"Fetched {total_count} predictions (last {days} days)")
    
    return PredictionHistoryResponse(
        total_predictions=total_count,
        high_risk_count=high_risk_count,
        average_churn_probability=round(avg_churn, 4),
        predictions=[
            PredictionRecord.model_validate(p) for p in predictions
        ]
    )


@app.get("/predictions/stats", tags=["Analytics"])
async def get_stats(db: Session = Depends(get_db)):
    """Get prediction statistics"""
    
    predictions = db.query(PredictionDB).all()
    
    if not predictions:
        return {
            "total_predictions": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "average_churn_probability": 0,
            "high_risk_percentage": 0
        }
    
    total = len(predictions)
    high_risk = sum(1 for p in predictions if p.churn_risk_level == "HIGH")
    medium_risk = sum(1 for p in predictions if p.churn_risk_level == "MEDIUM")
    low_risk = sum(1 for p in predictions if p.churn_risk_level == "LOW")
    avg_churn = sum(p.churn_probability for p in predictions) / total
    
    return {
        "total_predictions": total,
        "high_risk_count": high_risk,
        "medium_risk_count": medium_risk,
        "low_risk_count": low_risk,
        "average_churn_probability": round(avg_churn, 4),
        "high_risk_percentage": round((high_risk / total) * 100, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)