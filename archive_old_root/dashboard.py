"""
dashboard.py - Streamlit interactive analytics dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from analytics import AnalyticsEngine
from ml_model import ChurnModelTrainer
import pickle
import os

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CACHE ============
@st.cache_data
def load_data():
    """Load and preprocess data"""
    loader = DataLoader(r"D:\Projects\AI CHURN\data\raw\data_ecommerce_customer_churn.csv")
    df = loader.load()
    
    preprocessor = DataPreprocessor(df)
    df = preprocessor.handle_missing_values()
    
    return df

@st.cache_resource
def load_model():
    """Load pre-trained model"""
    try:
        if os.path.exists('models/churn_model.pkl'):
            with open('models/churn_model.pkl', 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None

# ============ LOAD DATA ============
try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("❌ Dataset not found. Place CSV in `data/raw/ecommerce_data.csv`")
    st.stop()

# ============ SIDEBAR ============
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Analytics", "🤖 Predictions", "💡 Insights"]
)

st.sidebar.divider()
st.sidebar.write("### 📈 Quick Stats")

analytics = AnalyticsEngine(df)
counts = analytics.customer_count()
churn_rate = analytics.overall_churn_rate()

st.sidebar.metric("Total Customers", f"{counts['total']:,}")
st.sidebar.metric("Churn Rate", f"{churn_rate:.1%}")
st.sidebar.metric("Avg Tenure", f"{analytics.avg_tenure()['overall']:.1f}m")

# ============ PAGE 1: ANALYTICS DASHBOARD ============
if page == "📊 Analytics":
    st.title("📊 Customer Churn Analytics Dashboard")
    st.markdown("Real-time analytics and customer insights")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{counts['total']:,}")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"{counts['churned']:,} churned")
    with col3:
        tenure = analytics.avg_tenure()
        st.metric("Avg Tenure", f"{tenure['overall']:.1f}m", 
                 delta=f"-{tenure['overall']-tenure['active']:.1f}m vs active")
    with col4:
        satisfaction = analytics.avg_satisfaction()
        st.metric("Avg Satisfaction", f"{satisfaction['overall']:.2f}/5",
                 delta=f"-{satisfaction['overall']-satisfaction['churned']:.2f} vs churned")
    
    st.divider()
    
    # Churn by Product Category
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛍️ Churn by Product Category")
        churn_by_cat = analytics.churn_by_product_category()
        
        cat_df = pd.DataFrame({
            'Category': list(churn_by_cat.keys()),
            'Churn Rate': list(churn_by_cat.values())
        })
        
        fig_cat = px.bar(
            cat_df,
            x='Churn Rate',
            y='Category',
            orientation='h',
            color='Churn Rate',
            color_continuous_scale='Reds'
        )
        fig_cat.update_xaxes(tickformat='.0%')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("😊 Churn by Satisfaction Score")
        satisfaction_churn = analytics.churn_by_satisfaction()
        
        sat_df = pd.DataFrame({
            'Score': list(satisfaction_churn.keys()),
            'Churn Rate': list(satisfaction_churn.values())
        }).sort_values('Score')
        
        fig_sat = px.line(
            sat_df,
            x='Score',
            y='Churn Rate',
            markers=True,
            title="Lower Satisfaction = Higher Churn"
        )
        fig_sat.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_sat, use_container_width=True)
    
    # Lifecycle Stage Analysis
    st.subheader("⏱️ Churn by Lifecycle Stage")
    
    lifecycle = analytics.churn_by_tenure_segment()
    lifecycle_df = pd.DataFrame({
        'Stage': list(lifecycle.keys()),
        'Churn Rate': list(lifecycle.values())
    })
    
    fig_lifecycle = px.bar(
        lifecycle_df,
        x='Stage',
        y='Churn Rate',
        color='Churn Rate',
        color_continuous_scale='RdYlGn_r',
        text='Churn Rate'
    )
    fig_lifecycle.update_traces(texttemplate='<b>%{y:.1%}</b>', textposition='outside')
    fig_lifecycle.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig_lifecycle, use_container_width=True)
    
    st.info("🚨 **New customers (0-3m) have the highest churn rate.** Focus onboarding improvements.")
    
    # Complaint Impact
    st.divider()
    st.subheader("🆘 Complaint Impact on Retention")
    
    col1, col2 = st.columns(2)
    
    complaints = analytics.churn_by_complaint()
    no_complaint = complaints['0']
    with_complaint = complaints['1']
    
    with col1:
        st.metric(
            "No Complaints",
            f"{no_complaint['churn_rate']:.2%}",
            delta=f"{no_complaint['count']:,} customers"
        )
    
    with col2:
        churn_diff = (with_complaint['churn_rate'] - no_complaint['churn_rate']) * 100
        st.metric(
            "With Complaints",
            f"{with_complaint['churn_rate']:.2%}",
            delta=f"+{churn_diff:.1f}pp higher!",
            delta_color="inverse"
        )
    
    st.warning(
        f"**Customers with complaints are {with_complaint['churn_rate']/no_complaint['churn_rate']:.1f}x more likely to churn.** "
        f"Improve complaint resolution time."
    )

# ============ PAGE 2: PREDICTIONS ============
elif page == "🤖 Predictions":
    st.title("🤖 Churn Risk Analysis")
    
    if model is None:
        st.error("❌ Model not trained. Run `python train.py` first.")
    else:
        # Get predictions
        trainer = ChurnModelTrainer(df)
        X, y = trainer.prepare_features()
        churn_probs = model.predict_proba(X)[:, 1]
        df['ChurnProbability'] = churn_probs
        
        # Risk distribution
        st.subheader("📊 Risk Distribution")
        
        high_risk = len(df[df['ChurnProbability'] > 0.7])
        medium_risk = len(df[(df['ChurnProbability'] >= 0.4) & (df['ChurnProbability'] <= 0.7)])
        low_risk = len(df[df['ChurnProbability'] < 0.4])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 High Risk (>70%)", high_risk, 
                     delta=f"{high_risk/len(df):.1%} of base")
        with col2:
            st.metric("🟡 Medium Risk (40-70%)", medium_risk,
                     delta=f"{medium_risk/len(df):.1%} of base")
        with col3:
            st.metric("🟢 Low Risk (<40%)", low_risk,
                     delta=f"{low_risk/len(df):.1%} of base")
        
        # Pie chart
        risk_dist = pd.DataFrame({
            'Risk': ['🔴 High', '🟡 Medium', '🟢 Low'],
            'Count': [high_risk, medium_risk, low_risk]
        })
        
        fig_risk = px.pie(
            risk_dist,
            values='Count',
            names='Risk',
            color_discrete_map={'🔴 High': '#ff4b4b', '🟡 Medium': '#ffa421', '🟢 Low': '#26a339'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # High-risk customers table
        st.divider()
        st.subheader("🚨 Top 30 High-Risk Customers")
        
        high_risk_df = df[df['ChurnProbability'] > 0.6].sort_values(
            'ChurnProbability', ascending=False
        )[['Tenure', 'SatisfactionScore', 'DaySinceLastOrder', 'Complain', 'ChurnProbability']].head(30)
        
        high_risk_df['ChurnProbability'] = high_risk_df['ChurnProbability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)

# ============ PAGE 3: INSIGHTS ============
elif page == "💡 Insights":
    st.title("💡 Key Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ New Customer Crisis")
        lifecycle = analytics.churn_by_tenure_segment()
        new_churn = lifecycle['New (0-3m)']
        mature_churn = lifecycle['Mature (13+m)']
        
        st.metric("New vs Mature Churn", f"{new_churn/mature_churn:.1f}x", 
                 delta=f"New: {new_churn:.1%}, Mature: {mature_churn:.1%}")
        st.write("**Action:** Improve onboarding, reduce friction in first 30 days")
    
    with col2:
        st.subheader("2️⃣ Satisfaction Impact")
        satisfaction_churn = analytics.churn_by_satisfaction()
        low_sat = satisfaction_churn.get(1, 0)
        high_sat = satisfaction_churn.get(5, 0)
        
        if low_sat > 0 and high_sat > 0:
            st.metric("Low vs High Satisfaction", f"{low_sat/high_sat:.1f}x",
                     delta=f"Low: {low_sat:.1%}, High: {high_sat:.1%}")
        st.write("**Action:** NPS program, product UX improvements, customer support")
    
    st.divider()
    
    complaints = analytics.churn_by_complaint()
    complaint_impact = complaints['1']['churn_rate'] / complaints['0']['churn_rate']
    
    st.subheader("3️⃣ Complaint Handling Critical")
    st.metric("Complaint Impact", f"{complaint_impact:.1f}x higher churn")
    st.write("**Action:** <48h resolution SLA, manager escalation, goodwill gesture")
    
    st.divider()
    
    st.subheader("📋 Recommended Retention Strategies")
    
    strategies = {
        "🎁 **Early Lifecycle Intervention**": {
            "Target": "New customers (0-3 months)",
            "Actions": [
                "Onboarding email sequence (Day 1, 7, 14, 30)",
                "Free shipping on first 3 orders",
                "Dedicated support hotline",
                "Reduce checkout friction"
            ]
        },
        "😊 **Satisfaction Recovery**": {
            "Target": "Customers with score 1-2",
            "Actions": [
                "Automated satisfaction survey",
                "Personal account manager outreach",
                "Root cause analysis",
                "Tailored incentive offer"
            ]
        },
        "📞 **Complaint Escalation Protocol**": {
            "Target": "All customer complaints",
            "Actions": [
                "Auto escalation to manager",
                "<48 hour resolution commitment",
                "Proactive follow-up",
                "15% credit on next order"
            ]
        }
    }
    
    for strategy_name, details in strategies.items():
        with st.expander(strategy_name):
            st.write(f"**Target:** {details['Target']}")
            st.write("**Actions:**")
            for action in details['Actions']:
                st.write(f"- {action}")

# ============ FOOTER ============
st.divider()
st.markdown("""
---
**Churn Intelligence Dashboard** | Data-driven customer retention insights
""")