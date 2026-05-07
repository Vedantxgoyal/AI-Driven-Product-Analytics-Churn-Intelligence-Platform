"""
model.py - Load and manage the trained churn prediction model
"""

import pickle
import os
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class ChurnModel:
    """Wrapper for trained churn prediction model"""
    
    def __init__(self, model_path: str = "models/churn_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'Tenure',
            'WarehouseToHome',
            'NumberOfDeviceRegistered',
            'PreferedOrderCat',
            'SatisfactionScore',
            'MaritalStatus',
            'NumberOfAddress',
            'Complain',
            'DaySinceLastOrder',
            'CashbackAmount'
        ]
        self.categorical_features = ['PreferedOrderCat', 'MaritalStatus']
        self.categorical_encodings = {}
    
    def load(self) -> bool:
        """Load model from disk"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"✓ Model loaded from {self.model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def _get_categorical_code(self, feature: str, value: str) -> int:
        """Get or create categorical encoding"""
        if feature not in self.categorical_encodings:
            self.categorical_encodings[feature] = {}
        
        if value not in self.categorical_encodings[feature]:
            code = len(self.categorical_encodings[feature])
            self.categorical_encodings[feature][value] = code
        
        return self.categorical_encodings[feature][value]
    
    def predict_single(self, features: dict) -> Tuple[float, str, str]:
        """Predict churn for single customer
        
        Args:
            features: Dict with customer features
        
        Returns:
            (churn_probability, risk_level, recommendation)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            import pandas as pd
            
            # Create feature array matching training order
            X = []
            for feat_name in self.feature_names:
                if feat_name == 'PreferedOrderCat':
                    code = self._get_categorical_code(feat_name, features['preferred_order_cat'])
                    X.append(code)
                elif feat_name == 'MaritalStatus':
                    code = self._get_categorical_code(feat_name, features['marital_status'])
                    X.append(code)
                elif feat_name == 'Tenure':
                    X.append(features['tenure'])
                elif feat_name == 'WarehouseToHome':
                    X.append(features['warehouse_to_home'])
                elif feat_name == 'NumberOfDeviceRegistered':
                    X.append(features['number_of_device_registered'])
                elif feat_name == 'SatisfactionScore':
                    X.append(features['satisfaction_score'])
                elif feat_name == 'NumberOfAddress':
                    X.append(features['number_of_address'])
                elif feat_name == 'Complain':
                    X.append(features['complain'])
                elif feat_name == 'DaySinceLastOrder':
                    X.append(features['days_since_last_order'])
                elif feat_name == 'CashbackAmount':
                    X.append(features['cashback_amount'])
            
            # Predict
            X_array = pd.DataFrame([X], columns=self.feature_names)
            churn_prob = float(self.model.predict_proba(X_array)[0, 1])
            
            # Risk level
            if churn_prob >= 0.7:
                risk_level = "HIGH"
                recommendation = "Immediate action: Call customer, offer loyalty incentive or special support"
            elif churn_prob >= 0.4:
                risk_level = "MEDIUM"
                recommendation = "Proactive engagement: Send targeted offer, request feedback"
            else:
                risk_level = "LOW"
                recommendation = "Maintain engagement: Regular check-ins, loyalty rewards"
            
            return churn_prob, risk_level, recommendation
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, features_list: List[dict]) -> List[Tuple[float, str, str]]:
        """Predict for multiple customers"""
        predictions = []
        for features in features_list:
            pred = self.predict_single(features)
            predictions.append(pred)
        return predictions


# Global model instance
_model_instance = None


def get_model() -> ChurnModel:
    """Get or create global model instance (singleton)"""
    global _model_instance
    
    if _model_instance is None:
        _model_instance = ChurnModel()
        _model_instance.load()
    
    return _model_instance


def initialize_model(model_path: str = "models/churn_model.pkl") -> bool:
    """Initialize model on startup"""
    global _model_instance
    _model_instance = ChurnModel(model_path)
    return _model_instance.load()