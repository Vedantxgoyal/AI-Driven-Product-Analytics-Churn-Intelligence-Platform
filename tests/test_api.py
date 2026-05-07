"""
test_api.py - Unit tests for REST API
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app, get_db
from app.database import Base

# In-memory SQLite for testing
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestHealthCheck:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestRoot:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data


class TestPrediction:
    def test_predict_valid(self):
        payload = {
            "customer": {
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
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "churn_probability" in data["data"]
    
    def test_predict_invalid_satisfaction(self):
        payload = {
            "customer": {
                "tenure": 12.5,
                "warehouse_to_home": 15.3,
                "number_of_device_registered": 3,
                "preferred_order_cat": "Electronics",
                "satisfaction_score": 10,  # Invalid
                "marital_status": "Single",
                "number_of_address": 4,
                "complain": 0,
                "days_since_last_order": 5.2,
                "cashback_amount": 175.50
            }
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestBatchPrediction:
    def test_batch_predict(self):
        payload = {
            "customers": [
                {
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
            ]
        }
        
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["processed_count"] == 1


class TestHistory:
    def test_get_predictions(self):
        response = client.get("/predictions")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data


class TestStats:
    def test_get_stats(self):
        response = client.get("/predictions/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data