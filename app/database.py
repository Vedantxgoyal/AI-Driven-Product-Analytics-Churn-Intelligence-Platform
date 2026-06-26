"""
app/database.py - SQLAlchemy ORM models for prediction storage
"""
from datetime import datetime, timezone
import os

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id                         = Column(Integer, primary_key=True, index=True, autoincrement=True)
    churn_probability          = Column(Float,   nullable=False)
    churn_risk_level           = Column(String(10),  nullable=False)
    recommendation             = Column(String(500), nullable=False)
    tenure                     = Column(Float,   nullable=False)
    warehouse_to_home          = Column(Float,   nullable=False)
    number_of_device_registered = Column(Integer, nullable=False)
    preferred_order_cat        = Column(String(100), nullable=False)
    satisfaction_score         = Column(Integer, nullable=False)
    marital_status             = Column(String(50),  nullable=False)
    number_of_address          = Column(Integer, nullable=False)
    complain                   = Column(Integer, nullable=False)
    days_since_last_order      = Column(Float,   nullable=False)
    cashback_amount            = Column(Float,   nullable=False)
    created_at                 = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                                        nullable=False, index=True)
    is_high_risk               = Column(Boolean, default=False, index=True)

    def __repr__(self):
        return f"<Prediction(id={self.id}, risk={self.churn_risk_level}, prob={self.churn_probability:.2%})>"


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# NOTE: create_all is intentionally NOT called here.
# Tables are created in app/main.py startup and in tests/conftest.py.

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()