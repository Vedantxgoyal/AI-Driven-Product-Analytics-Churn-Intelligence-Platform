"""
tests/test_training.py
Unit tests for the XGBoost training pipeline.
Tests data loading, feature engineering, model training, SHAP, and promotion gate.
"""
import pathlib
import pytest
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "data_ecommerce_customer_churn.csv"


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    from src.ml.train import load_data
    return load_data()


@pytest.fixture(scope="module")
def features(raw_df):
    from src.ml.train import build_features
    X, y = build_features(raw_df)
    return X, y


@pytest.fixture(scope="module")
def trained_model(features):
    from src.ml.train import train
    X, y = features
    model, metrics = train(X, y)
    return model, metrics


# ── Data Loading ───────────────────────────────────────────────────────────

class TestDataLoading:
    def test_data_file_exists(self):
        assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"

    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_row_count(self, raw_df):
        assert len(raw_df) >= 100

    def test_churn_column_exists(self, raw_df):
        assert "Churn" in raw_df.columns

    def test_churn_is_binary(self, raw_df):
        assert set(raw_df["Churn"].unique()).issubset({0, 1})

    def test_churn_rate_reasonable(self, raw_df):
        rate = raw_df["Churn"].mean()
        assert 0.05 <= rate <= 0.60, f"Unexpected churn rate: {rate:.2%}"

    def test_no_nulls_in_target(self, raw_df):
        assert raw_df["Churn"].isnull().sum() == 0


# ── Feature Engineering ────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_returns_dataframe_and_series(self, features):
        X, y = features
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_feature_count(self, features):
        X, _ = features
        assert X.shape[1] == 10

    def test_no_nulls_in_features(self, features):
        X, _ = features
        assert X.isnull().sum().sum() == 0

    def test_target_aligned_with_features(self, features):
        X, y = features
        assert len(X) == len(y)

    def test_categoricals_encoded(self, features):
        X, _ = features
        assert X["PreferedOrderCat"].dtype in [np.int32, np.int64, np.float64]
        assert X["MaritalStatus"].dtype in [np.int32, np.int64, np.float64]

    def test_expected_columns(self, features):
        X, _ = features
        expected = {
            "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
            "PreferedOrderCat", "SatisfactionScore", "MaritalStatus",
            "NumberOfAddress", "Complain", "DaySinceLastOrder", "CashbackAmount"
        }
        assert expected == set(X.columns)


# ── Model Training ─────────────────────────────────────────────────────────

class TestModelTraining:
    def test_train_returns_model_and_metrics(self, trained_model):
        model, metrics = trained_model
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_keys_present(self, trained_model):
        _, metrics = trained_model
        for key in ["f1", "pr_auc", "roc_auc", "precision", "recall"]:
            assert key in metrics

    def test_f1_above_threshold(self, trained_model):
        _, metrics = trained_model
        assert metrics["f1"] >= 0.65, f"F1 too low: {metrics['f1']:.4f}"

    def test_pr_auc_above_threshold(self, trained_model):
        _, metrics = trained_model
        assert metrics["pr_auc"] >= 0.70, f"PR-AUC too low: {metrics['pr_auc']:.4f}"

    def test_roc_auc_above_threshold(self, trained_model):
        _, metrics = trained_model
        assert metrics["roc_auc"] >= 0.85, f"ROC-AUC too low: {metrics['roc_auc']:.4f}"

    def test_recall_above_threshold(self, trained_model):
        _, metrics = trained_model
        assert metrics["recall"] >= 0.70, f"Recall too low: {metrics['recall']:.4f}"

    def test_model_can_predict(self, trained_model, features):
        model, _ = trained_model
        X, _ = features
        probs = model.predict_proba(X[:10])
        assert probs.shape == (10, 2)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_predictions_sum_to_one(self, trained_model, features):
        model, _ = trained_model
        X, _ = features
        probs = model.predict_proba(X[:5])
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ── SHAP Explainer ─────────────────────────────────────────────────────────

class TestSHAPExplainer:
    def test_explainer_builds(self, trained_model, features):
        from src.ml.train import build_shap_explainer
        model, _ = trained_model
        X, _ = features
        explainer = build_shap_explainer(model, X)
        assert explainer is not None

    def test_shap_values_shape(self, trained_model, features):
        from src.ml.train import build_shap_explainer
        import shap
        model, _ = trained_model
        X, _ = features
        explainer = build_shap_explainer(model, X)
        vals = explainer.shap_values(X.iloc[:5])
        assert vals.shape[0] == 5
        assert vals.shape[1] == X.shape[1]

    def test_shap_values_are_finite(self, trained_model, features):
        from src.ml.train import build_shap_explainer
        model, _ = trained_model
        X, _ = features
        explainer = build_shap_explainer(model, X)
        vals = explainer.shap_values(X.iloc[:5])
        assert np.all(np.isfinite(vals))


# ── Promotion Gate ─────────────────────────────────────────────────────────

class TestPromotionGate:
    def test_promotes_good_model(self):
        from src.ml.train import should_promote
        assert should_promote({"f1": 0.75}) is True

    def test_rejects_bad_model(self):
        from src.ml.train import should_promote
        assert should_promote({"f1": 0.50}) is False

    def test_boundary_at_baseline(self):
        from src.ml.train import should_promote, BASELINE_F1
        assert should_promote({"f1": BASELINE_F1}) is True
        assert should_promote({"f1": BASELINE_F1 - 0.001}) is False


# ── Validation Integration ─────────────────────────────────────────────────

class TestValidationIntegration:
    def test_validate_training_data_passes(self):
        from src.data.validate import validate_file
        passed, report = validate_file()
        assert passed, f"Validation failed: {report['failures']}"

    def test_validate_report_structure(self):
        from src.data.validate import validate_file
        _, report = validate_file()
        for key in ["passed", "total_checks", "passed_checks", "row_count", "churn_rate_pct"]:
            assert key in report