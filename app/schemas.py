"""
schemas.py - Pydantic models for API input/output validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# ============ INPUT SCHEMAS ============

class CustomerFeatures(BaseModel):
    """Single customer features for prediction"""
    
    tenure: float = Field(..., ge=0, le=100, description="Months as customer")
    warehouse_to_home: float = Field(..., ge=0, le=200, description="Distance in km")
    number_of_device_registered: int = Field(..., ge=1, le=10, description="Devices used")
    preferred_order_cat: str = Field(..., min_length=1, description="Product category")
    satisfaction_score: int = Field(..., ge=1, le=5, description="1-5 scale")
    marital_status: str = Field(..., min_length=1, description="Marital status")
    number_of_address: int = Field(..., ge=1, le=50, description="Saved addresses")
    complain: int = Field(..., ge=0, le=1, description="Has complained (0/1)")
    days_since_last_order: float = Field(..., ge=0, le=100, description="Days inactive")
    cashback_amount: float = Field(..., ge=0, le=500, description="Cashback earned")
    
    class Config:
        schema_extra = {
            "example": {
                "tenure": 12.5,
                "warehouse_to_home": 15.3,
                "number_of_device_registered": 3,
                "preferred_order_cat": "Electronics",
                "satisfaction_score": 4,
                "marital_status": "Single",
                "number_of_address": 4,
                "complain": 0,
                "days_since_last_order": 5.2,
                "cashback_amount": 175.50
            }
        }


class PredictRequest(BaseModel):
    """Single prediction request"""
    customer: CustomerFeatures


class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerFeatures] = Field(..., min_items=1, max_items=1000)


# ============ OUTPUT SCHEMAS ============

class PredictionResult(BaseModel):
    """Single prediction result"""
    
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    churn_risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    recommendation: str = Field(..., description="Action recommendation")
    
    class Config:
        schema_extra = {
            "example": {
                "churn_probability": 0.75,
                "churn_risk_level": "HIGH",
                "recommendation": "Immediate outreach: offer loyalty discount"
            }
        }


class PredictResponse(BaseModel):
    """Response for single prediction"""
    success: bool
    data: Optional[PredictionResult] = None
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """Response for batch prediction"""
    success: bool
    processed_count: int
    high_risk_count: int
    predictions: List[PredictionResult]
    processing_time_ms: float
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response"""
    
    status: str
    model_loaded: bool
    database_connected: bool
    version: str
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "database_connected": True,
                "version": "1.0.0"
            }
        }


class PredictionRecord(BaseModel):
    """Stored prediction record"""
    
    id: int
    churn_probability: float
    churn_risk_level: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    """Response for prediction history"""
    
    total_predictions: int
    high_risk_count: int
    average_churn_probability: float
    predictions: List[PredictionRecord]


class ErrorResponse(BaseModel):
    """Standard error response"""
    
    error: str
    detail: Optional[str] = None
    status_code: int