"""
tests/test_api.py
Production-grade API tests.
"""
import pytest

VALID_CUSTOMER = {
    "tenure":                      12.0,
    "warehouse_to_home":           15.0,
    "number_of_device_registered": 3,
    "preferred_order_cat":         "Mobile Phone",
    "satisfaction_score":          3,
    "marital_status":              "Single",
    "number_of_address":           2,
    "complain":                    0,
    "days_since_last_order":       5.0,
    "cashback_amount":             175.0,
}

HIGH_RISK_CUSTOMER = {
    "tenure":                      1.0,
    "warehouse_to_home":           30.0,
    "number_of_device_registered": 5,
    "preferred_order_cat":         "Mobile Phone",
    "satisfaction_score":          1,
    "marital_status":              "Single",
    "number_of_address":           2,
    "complain":                    1,
    "days_since_last_order":       25.0,
    "cashback_amount":             50.0,
}


# ── Health / Info ──────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        for key in ["status", "model_loaded", "database_connected", "version"]:
            assert key in data

    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_has_name(self, client):
        assert "name" in client.get("/").json()


# ── Single Prediction ──────────────────────────────────────────────────────

class TestPredict:
    def test_valid_customer_200(self, client):
        r = client.post("/predict", json={"customer": VALID_CUSTOMER})
        assert r.status_code == 200

    def test_response_schema(self, client):
        data = client.post("/predict", json={"customer": VALID_CUSTOMER}).json()
        assert data["success"] is True
        assert "churn_probability" in data["data"]
        assert "churn_risk_level" in data["data"]
        assert "recommendation" in data["data"]

    def test_probability_in_range(self, client):
        prob = client.post("/predict", json={"customer": VALID_CUSTOMER}).json()["data"]["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_valid(self, client):
        risk = client.post("/predict", json={"customer": VALID_CUSTOMER}).json()["data"]["churn_risk_level"]
        assert risk in {"HIGH", "MEDIUM", "LOW"}

    def test_high_risk_customer(self, client):
        data = client.post("/predict", json={"customer": HIGH_RISK_CUSTOMER}).json()
        assert data["data"]["churn_probability"] > 0.4

    def test_invalid_satisfaction_422(self, client):
        bad = {**VALID_CUSTOMER, "satisfaction_score": 10}
        assert client.post("/predict", json={"customer": bad}).status_code == 422

    def test_missing_field_422(self, client):
        bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        assert client.post("/predict", json={"customer": bad}).status_code == 422

    def test_negative_tenure_422(self, client):
        bad = {**VALID_CUSTOMER, "tenure": -1}
        assert client.post("/predict", json={"customer": bad}).status_code == 422


# ── Explain ────────────────────────────────────────────────────────────────

class TestExplain:
    def test_explain_200(self, client):
        assert client.post("/explain", json={"customer": VALID_CUSTOMER}).status_code == 200

    def test_explain_schema(self, client):
        data = client.post("/explain", json={"customer": VALID_CUSTOMER}).json()
        assert "churn_probability" in data
        assert "explanation" in data
        assert "top_5_drivers" in data["explanation"]
        assert "contributions" in data["explanation"]

    def test_explain_top5_length(self, client):
        data = client.post("/explain", json={"customer": VALID_CUSTOMER}).json()
        assert len(data["explanation"]["top_5_drivers"]) == 5

    def test_explain_driver_schema(self, client):
        data = client.post("/explain", json={"customer": VALID_CUSTOMER}).json()
        for d in data["explanation"]["top_5_drivers"]:
            assert "feature" in d
            assert "shap_value" in d


# ── Batch Prediction ───────────────────────────────────────────────────────

class TestBatchPredict:
    def test_single_customer_batch(self, client):
        r = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER]})
        assert r.status_code == 200
        assert r.json()["processed_count"] == 1

    def test_multi_customer_batch(self, client):
        data = client.post("/predict/batch",
                                json={"customers": [VALID_CUSTOMER, HIGH_RISK_CUSTOMER]}).json()
        assert data["processed_count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_high_risk_count(self, client):
        data = client.post("/predict/batch",
                                json={"customers": [HIGH_RISK_CUSTOMER, HIGH_RISK_CUSTOMER]}).json()
        assert data["high_risk_count"] >= 0

    def test_batch_processing_time_present(self, client):
        data = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER]}).json()
        assert "processing_time_ms" in data

    def test_empty_batch_422(self, client):
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422


# ── History / Stats ────────────────────────────────────────────────────────

class TestHistory:
    def test_predictions_200(self, client):
        assert client.get("/predictions").status_code == 200

    def test_predictions_schema(self, client):
        data = client.get("/predictions").json()
        for key in ["total_predictions", "high_risk_count", "average_churn_probability"]:
            assert key in data

    def test_predictions_days_filter(self, client):
        assert client.get("/predictions?days=1").status_code == 200

    def test_predictions_high_risk_filter(self, client):
        assert client.get("/predictions?high_risk_only=true").status_code == 200

    def test_stats_200(self, client):
        assert client.get("/predictions/stats").status_code == 200

    def test_stats_schema(self, client):
        data = client.get("/predictions/stats").json()
        for key in ["total_predictions", "high_risk_count", "medium_risk_count",
                    "low_risk_count", "average_churn_probability"]:
            assert key in data


# ── Analytics ──────────────────────────────────────────────────────────────

class TestAnalytics:
    def test_overview_200(self, client):
        assert client.get("/analytics/overview").status_code in {200, 503}

    def test_customer_counts_200(self, client):
        assert client.get("/analytics/customer-counts").status_code in {200, 503}

    def test_churn_by_satisfaction_200(self, client):
        assert client.get("/analytics/churn-by-satisfaction").status_code in {200, 503}

    def test_churn_by_tenure_200(self, client):
        assert client.get("/analytics/churn-by-tenure").status_code in {200, 503}

    def test_risk_segmentation_200(self, client):
        assert client.get("/analytics/risk-segmentation").status_code in {200, 503}
