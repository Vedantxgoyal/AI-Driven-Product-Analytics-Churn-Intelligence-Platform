"""
analytics.py - Compute business metrics and KPIs
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import json

class AnalyticsEngine:
    """Compute all business metrics for churn analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Handle both 0/1 and Yes/No formats
        if self.df['Churn'].dtype == 'object':
            self.df['Churn'] = (self.df['Churn'] == 'Yes').astype(int)
        
        self.churn_df = self.df[self.df['Churn'] == 1]
        self.active_df = self.df[self.df['Churn'] == 0]
    
    # ============ FUNDAMENTAL METRICS ============
    
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
    
    # ============ BEHAVIORAL METRICS ============
    
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
    
    def avg_cashback(self) -> Dict[str, float]:
        """Average cashback earned"""
        return {
            'overall': float(self.df['CashbackAmount'].mean()),
            'churned': float(self.churn_df['CashbackAmount'].mean()),
            'active': float(self.active_df['CashbackAmount'].mean())
        }
    
    # ============ SEGMENTATION ANALYSIS ============
    
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
    
    def churn_by_marital_status(self) -> Dict[str, float]:
        """Churn rate by marital status"""
        result = {}
        for status in self.df['MaritalStatus'].unique():
            status_df = self.df[self.df['MaritalStatus'] == status]
            churn_rate = (status_df['Churn'] == 1).sum() / len(status_df)
            result[str(status)] = float(churn_rate)
        return result
    
    def churn_by_product_category(self) -> Dict[str, float]:
        """Churn rate by product preference"""
        result = {}
        for cat in self.df['PreferedOrderCat'].unique():
            cat_df = self.df[self.df['PreferedOrderCat'] == cat]
            churn_rate = (cat_df['Churn'] == 1).sum() / len(cat_df)
            result[str(cat)] = float(churn_rate)
        
        # Sort by churn rate (highest first)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    
    # ============ ADVANCED METRICS ============
    
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
    
    def high_risk_customers(self) -> pd.DataFrame:
        """Identify top risk customers"""
        
        risk_df = self.df.copy()
        
        # Composite risk score
        risk_df['RecencyRisk'] = (risk_df['DaySinceLastOrder'] / risk_df['DaySinceLastOrder'].max()) * 30
        risk_df['SatisfactionRisk'] = ((5 - risk_df['SatisfactionScore']) / 4) * 25
        risk_df['TenureRisk'] = ((risk_df['Tenure'].max() - risk_df['Tenure']) / risk_df['Tenure'].max()) * 20
        risk_df['ComplaintRisk'] = risk_df['Complain'] * 15
        
        risk_df['CompositeRiskScore'] = (
            risk_df['RecencyRisk'] +
            risk_df['SatisfactionRisk'] +
            risk_df['TenureRisk'] +
            risk_df['ComplaintRisk']
        )
        
        high_risk = risk_df[risk_df['CompositeRiskScore'] > 50].sort_values(
            'CompositeRiskScore', ascending=False
        )
        
        return high_risk[['Tenure', 'SatisfactionScore', 'DaySinceLastOrder', 
                         'Complain', 'CompositeRiskScore']].head(50)
    
    # ============ GENERATE REPORT ============
    
    def get_full_report(self) -> Dict:
        """Generate complete analytics report"""
        
        return {
            'churn_rate': self.overall_churn_rate(),
            'customer_counts': self.customer_count(),
            'avg_tenure': self.avg_tenure(),
            'avg_satisfaction': self.avg_satisfaction(),
            'avg_cashback': self.avg_cashback(),
            'churn_by_satisfaction': self.churn_by_satisfaction(),
            'churn_by_tenure_segment': self.churn_by_tenure_segment(),
            'churn_by_complaint': self.churn_by_complaint(),
            'churn_by_marital_status': self.churn_by_marital_status(),
            'churn_by_product_category': self.churn_by_product_category(),
            'risk_segmentation': self.risk_segmentation()
        }
    
    def print_report(self):
        """Print formatted analytics report"""
        
        print("\n" + "="*70)
        print("📊 CHURN ANALYTICS REPORT")
        print("="*70)
        
        # Customer metrics
        counts = self.customer_count()
        print(f"\n🎯 CUSTOMER BASE")
        print(f"  Total Customers: {counts['total']:,}")
        print(f"  Active: {counts['active']:,}")
        print(f"  Churned: {counts['churned']:,}")
        print(f"  Churn Rate: {self.overall_churn_rate():.2%}")
        
        # Tenure
        tenure = self.avg_tenure()
        print(f"\n⏱️  TENURE (months)")
        print(f"  Overall Avg: {tenure['overall']:.1f}m")
        print(f"  Churned Avg: {tenure['churned']:.1f}m")
        print(f"  Active Avg: {tenure['active']:.1f}m")
        
        # Satisfaction
        satisfaction = self.avg_satisfaction()
        print(f"\n😊 SATISFACTION (1-5 scale)")
        print(f"  Overall Avg: {satisfaction['overall']:.2f}/5")
        print(f"  Churned Avg: {satisfaction['churned']:.2f}/5")
        print(f"  Active Avg: {satisfaction['active']:.2f}/5")
        
        # Cashback
        cashback = self.avg_cashback()
        print(f"\n💰 CASHBACK EARNED")
        print(f"  Overall Avg: ${cashback['overall']:.2f}")
        print(f"  Churned Avg: ${cashback['churned']:.2f}")
        print(f"  Active Avg: ${cashback['active']:.2f}")
        
        # Risk segmentation
        risk = self.risk_segmentation()
        print(f"\n⚠️  RISK SEGMENTATION")
        print(f"  High Risk (75+): {risk['high_risk (75+)']:,}")
        print(f"  Medium Risk (40-74): {risk['medium_risk (40-74)']:,}")
        print(f"  Low Risk (<40): {risk['low_risk (<40)']:,}")
        
        # Lifecycle
        lifecycle = self.churn_by_tenure_segment()
        print(f"\n📈 CHURN BY LIFECYCLE STAGE")
        for stage, rate in lifecycle.items():
            print(f"  {stage}: {rate:.2%}")
        
        # Top churn categories
        top_categories = list(self.churn_by_product_category().items())[:3]
        print(f"\n🛍️  TOP CHURN CATEGORIES")
        for cat, rate in top_categories:
            print(f"  {cat}: {rate:.2%}")
        
        print("\n" + "="*70 + "\n")