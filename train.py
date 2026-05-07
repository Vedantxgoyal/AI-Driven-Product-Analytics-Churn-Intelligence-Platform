"""
train.py - Main pipeline: Load data → Analytics → Train model
Run this first to train the model and see analytics
"""

import os
import sys
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from analytics import AnalyticsEngine
from ml_model import ChurnModelTrainer

def main():
    print("\n" + "="*70)
    print("🚀 CHURN ANALYTICS PIPELINE")
    print("="*70)
    
    # ============ STAGE 1: LOAD DATA ============
    print("\n[1/4] LOADING DATA...")
    try:
        loader = DataLoader(r"D:\Projects\AI CHURN\data\raw\data_ecommerce_customer_churn.csv")
        df = loader.load()
        loader.print_summary()
    except FileNotFoundError:
        print("❌ Error: data/raw/ecommerce_data.csv not found")
        print("   Download dataset from Kaggle and place in data/raw/")
        sys.exit(1)
    
    # ============ STAGE 2: PREPROCESS DATA ============
    print("\n[2/4] PREPROCESSING DATA...")
    preprocessor = DataPreprocessor(df)
    df = preprocessor.handle_missing_values(strategy='median')
    df, encodings = preprocessor.encode_categorical()
    preprocessor.print_statistics()
    
    # ============ STAGE 3: ANALYTICS ============
    print("\n[3/4] COMPUTING ANALYTICS...")
    analytics = AnalyticsEngine(df)
    analytics.print_report()
    
    # ============ STAGE 4: TRAIN MODEL ============
    print("\n[4/4] TRAINING ML MODEL...")
    
    os.makedirs('models', exist_ok=True)
    
    trainer = ChurnModelTrainer(df)
    trainer.train()
    trainer.save_model('models/churn_model.pkl')
    
    # ============ SUMMARY ============
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"""
📊 OUTPUTS GENERATED:
   ✓ models/churn_model.pkl       (Trained Random Forest)
   ✓ models/churn_model_features.txt (Feature names)

🎯 NEXT STEPS:

   1. Launch Analytics Dashboard:
      $ streamlit run dashboard.py
   
   2. View at:
      http://localhost:8501
   
   3. Then build the REST API:
      $ cd ../churn-api
      $ cp ../ecommerce-churn-analytics/models/churn_model.pkl model/

📈 KEY METRICS:
   • Total Customers: {analytics.customer_count()['total']:,}
   • Churn Rate: {analytics.overall_churn_rate():.1%}
   • Avg Tenure: {analytics.avg_tenure()['overall']:.1f} months
   • High-Risk: {analytics.risk_segmentation()['high_risk (75+)']:,} customers

💡 MODEL PERFORMANCE:
   • Accuracy: {trainer.metrics['test_accuracy']:.1%}
   • ROC-AUC: {trainer.metrics['roc_auc']:.3f}
   • F1-Score: {trainer.metrics['f1_score']:.3f}
   
🚀 READY FOR API DEPLOYMENT!
""")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()