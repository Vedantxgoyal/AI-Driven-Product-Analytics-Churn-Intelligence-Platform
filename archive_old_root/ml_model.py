"""
ml_model.py - Train churn prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
import pickle
import os
from typing import Tuple, Dict

class ChurnModelTrainer:
    """Train and evaluate churn prediction model"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        
        print("\n🔧 PREPARING FEATURES FOR ML")
        print("="*60)
        
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        # Handle Churn encoding
        if y.dtype == 'object':
            y = (y == 'Yes').astype(int)
        
        # Encode categorical columns
        categorical_cols = X.select_dtypes(include='object').columns
        
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Feature names: {self.feature_names}")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")
        print(f"✓ Churn rate: {(y==1).sum() / len(y):.2%}")
        print("="*60)
        
        return X, y
    
    def train(self, test_size: float = 0.2, random_state: int = 42):
        """Train Random Forest model"""
        
        print("\n🤖 TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        X, y = self.prepare_features()
        
        # Train/test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"\nTrain set: {len(self.X_train):,} samples")
        print(f"  Churned: {(self.y_train==1).sum():,} ({(self.y_train==1).sum()/len(self.y_train):.1%})")
        print(f"\nTest set: {len(self.X_test):,} samples")
        print(f"  Churned: {(self.y_test==1).sum():,} ({(self.y_test==1).sum()/len(self.y_test):.1%})")
        
        # Train model
        print(f"\nTraining Random Forest (200 trees)...")
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model trained successfully")
        print("="*60)
        
        # Evaluate
        self._evaluate()
    
    def _evaluate(self):
        """Evaluate model performance"""
        
        print("\n📈 MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        train_acc = self.model.score(self.X_train, self.y_train)
        test_acc = self.model.score(self.X_test, self.y_test)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Store metrics
        self.metrics = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'roc_auc': float(roc_auc),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        print(f"\n✓ ACCURACY")
        print(f"  Train: {train_acc:.4f}")
        print(f"  Test: {test_acc:.4f}")
        
        print(f"\n✓ CLASSIFICATION METRICS")
        print(f"  ROC-AUC: {roc_auc:.4f} ⭐ (Best for imbalanced data)")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        print(f"\n✓ CONFUSION MATRIX")
        print(f"  True Negatives: {tn} (correctly identified active)")
        print(f"  False Positives: {fp} (wrongly flagged)")
        print(f"  False Negatives: {fn} (missed churners) ⚠️")
        print(f"  True Positives: {tp} (correctly identified churners)")
        
        print("\n" + "="*60)
        
        # Feature importance
        self._feature_importance()
    
    def _feature_importance(self):
        """Extract and display feature importance"""
        
        print("\n🎯 FEATURE IMPORTANCE (What drives churn?)")
        print("="*60)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        for idx, row in importance_df.head(10).iterrows():
            pct = row['importance'] * 100
            bar = "█" * int(pct / 2)
            print(f"  {row['feature']:<30} {bar} {pct:.1f}%")
        
        print("\n" + "="*60 + "\n")
        
        return importance_df
    
    def save_model(self, filepath: str = 'models/churn_model.pkl'):
        """Save trained model to disk"""
        
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✓ Model saved to: {filepath}")
        
        # Also save feature names for inference
        feature_path = filepath.replace('.pkl', '_features.txt')
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print(f"✓ Features saved to: {feature_path}")
    
    def get_metrics(self) -> Dict:
        """Get evaluation metrics"""
        return self.metrics