"""
database.py - SQLAlchemy ORM models for prediction storage
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class PredictionRecord(Base):
    """Store all predictions for audit and analytics"""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Prediction results
    churn_probability = Column(Float, nullable=False)
    churn_risk_level = Column(String(10), nullable=False)
    recommendation = Column(String(500), nullable=False)
    
    # Input features (for audit trail)
    tenure = Column(Float, nullable=False)
    warehouse_to_home = Column(Float, nullable=False)
    number_of_device_registered = Column(Integer, nullable=False)
    preferred_order_cat = Column(String(100), nullable=False)
    satisfaction_score = Column(Integer, nullable=False)
    marital_status = Column(String(50), nullable=False)
    number_of_address = Column(Integer, nullable=False)
    complain = Column(Integer, nullable=False)
    days_since_last_order = Column(Float, nullable=False)
    cashback_amount = Column(Float, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    is_high_risk = Column(Boolean, default=False, index=True)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, risk={self.churn_risk_level}, prob={self.churn_probability:.2%})>"


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()