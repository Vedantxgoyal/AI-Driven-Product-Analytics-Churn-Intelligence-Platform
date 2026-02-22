# AI-Driven-Product-Analytics-Churn-Intelligence-Platform
This project is a cloud-native end-to-end analytics system designed to simulate a real-world product analytics infrastructure. It integrates data engineering, machine learning, and generative AI to predict customer churn and automatically generate business retention strategies.
The system ingests raw customer activity data into an AWS S3 data lake, processes and warehouses it using Apache Hive, performs distributed transformations with Apache Spark, and extracts key product metrics using SQL. A machine learning model predicts customer churn probability, and a Generative AI layer converts analytical outputs into actionable business recommendations.

The entire application is containerized using Docker and automated with CI/CD pipelines to reflect production-level deployment practices.

🚀 Key Features

Cloud-based data lake architecture using Amazon Web Services (S3)

Data warehousing with Apache Hive

Distributed data processing using Apache Spark

Advanced SQL-based KPI computation (Churn Rate, ARPU, Contract-wise Risk)

Machine Learning churn prediction using Random Forest

AI-generated retention strategies via OpenAI GPT

Streamlit-based interactive analytics dashboard

Containerized deployment using Docker

Automated CI/CD workflow using GitHub Actions

🏗 Architecture Overview

Raw Customer Data → AWS S3 (Data Lake) → Hive Warehouse → Spark Processing → SQL Analytics → ML Churn Model → GenAI Strategy Engine → Streamlit Dashboard → Dockerized Deployment

📊 Business Impact

Predicts high-risk customers before churn

Identifies revenue-impacting segments

Automatically generates data-driven retention strategies

Simulates enterprise-grade product analytics infrastructure

🛠 Tech Stack

Cloud: AWS S3, EC2
Data Warehouse: Apache Hive
Processing Engine: Apache Spark
Analytics: SQL
Machine Learning: Python (scikit-learn)
Generative AI: OpenAI GPT API
Visualization: Streamlit
DevOps: Docker, GitHub Actions

🎯 Learning Outcomes

End-to-end data pipeline design

Cloud-based analytics architecture

Applied machine learning for business problems

Generative AI integration for automated insight generation

Containerized deployment & CI/CD automation
