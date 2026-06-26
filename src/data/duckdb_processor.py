"""
duckdb_processor.py - DuckDB-based high-performance analytics processor
Replaces Spark with production-grade OLAP engine
Works perfectly on Windows, Linux, Mac
"""

import duckdb
import pandas as pd
import logging
from typing import Dict, List, Tuple
import os

logger = logging.getLogger(__name__)


class DuckDBProcessor:
    """High-performance OLAP analytics using DuckDB"""
    
    def __init__(self, app_name: str = "Churn Intelligence"):
        """Initialize DuckDB processor"""
        self.app_name = app_name
        self.conn = duckdb.connect(':memory:')  # In-memory database
        self.table_name = "customers"
        logger.info(f"✓ DuckDB processor initialized (in-memory)")
    
    def load_csv(self, filepath: str) -> bool:
        """Load CSV into DuckDB"""
        try:
            # Load CSV directly into DuckDB table
            self.conn.execute(f"""
                CREATE TABLE {self.table_name} AS 
                SELECT * FROM read_csv_auto('{filepath}')
            """)
            
            # Get row count
            row_count = self.conn.execute(
                f"SELECT COUNT(*) as cnt FROM {self.table_name}"
            ).fetchall()[0][0]
            
            logger.info(f"✓ DuckDB loaded {row_count:,} rows from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            return False
    
    def get_schema(self) -> Dict:
        """Get table schema"""
        try:
            schema_info = self.conn.execute(
                f"DESCRIBE {self.table_name}"
            ).fetchall()
            
            schema = {}
            for col_name, col_type, *_ in schema_info:
                schema[col_name] = col_type
            
            return schema
        
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}")
            return {}
    
    def compute_churn_metrics(self) -> Dict:
        """Compute comprehensive churn metrics using DuckDB SQL"""
        
        try:
            # Main metrics query
            query = f"""
            SELECT
                COUNT(*) as total_customers,
                SUM(CASE WHEN "Churn" = 1 THEN 1 ELSE 0 END) as churned_customers,
                COUNT(*) - SUM(CASE WHEN "Churn" = 1 THEN 1 ELSE 0 END) as active_customers,
                ROUND(CAST(SUM(CASE WHEN "Churn" = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 4) as churn_rate,
                ROUND(AVG("Tenure"), 2) as avg_tenure,
                ROUND(AVG(CASE WHEN "Churn" = 1 THEN "Tenure" ELSE NULL END), 2) as avg_tenure_churned,
                ROUND(AVG(CASE WHEN "Churn" = 0 THEN "Tenure" ELSE NULL END), 2) as avg_tenure_active,
                ROUND(AVG("SatisfactionScore"), 2) as avg_satisfaction,
                ROUND(AVG(CASE WHEN "Churn" = 1 THEN "SatisfactionScore" ELSE NULL END), 2) as avg_satisfaction_churned,
                ROUND(AVG(CASE WHEN "Churn" = 0 THEN "SatisfactionScore" ELSE NULL END), 2) as avg_satisfaction_active,
                ROUND(AVG("CashbackAmount"), 2) as avg_cashback,
                ROUND(AVG(CASE WHEN "Churn" = 1 THEN "CashbackAmount" ELSE NULL END), 2) as avg_cashback_churned,
                ROUND(AVG(CASE WHEN "Churn" = 0 THEN "CashbackAmount" ELSE NULL END), 2) as avg_cashback_active
            FROM {self.table_name}
            """
            
            result = self.conn.execute(query).fetchall()[0]
            
            # Complaint impact query
            complaint_query = f"""
            SELECT
                "Complain",
                COUNT(*) as count,
                ROUND(CAST(SUM(CASE WHEN "Churn" = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 4) as churn_rate
            FROM {self.table_name}
            GROUP BY "Complain"
            """
            
            complaint_results = self.conn.execute(complaint_query).fetchall()
            complaint_dict = {}
            
            for complain_status, count, churn_rate in complaint_results:
                complaint_dict[int(complain_status)] = {
                    "count": int(count),
                    "churn_rate": float(churn_rate)
                }
            
            return {
                "total_customers": int(result[0]),
                "churned_customers": int(result[1]),
                "active_customers": int(result[2]),
                "churn_rate": float(result[3]),
                "tenure": {
                    "average": float(result[4]) if result[4] else 0,
                    "churned_average": float(result[5]) if result[5] else 0,
                    "active_average": float(result[6]) if result[6] else 0
                },
                "satisfaction": {
                    "average": float(result[7]) if result[7] else 0,
                    "churned_average": float(result[8]) if result[8] else 0,
                    "active_average": float(result[9]) if result[9] else 0
                },
                "cashback": {
                    "average": float(result[10]) if result[10] else 0,
                    "churned_average": float(result[11]) if result[11] else 0,
                    "active_average": float(result[12]) if result[12] else 0
                },
                "complaints": complaint_dict
            }
        
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {}
    
    def compute_churn_by_satisfaction(self) -> Dict:
        """Churn rate by satisfaction score"""
        
        try:
            query = f"""
            SELECT
                "SatisfactionScore",
                COUNT(*) as count,
                ROUND(CAST(SUM(CASE WHEN "Churn" = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 4) as churn_rate
            FROM {self.table_name}
            GROUP BY "SatisfactionScore"
            ORDER BY "SatisfactionScore"
            """
            
            results = self.conn.execute(query).fetchall()
            
            result_dict = {}
            for score, count, churn_rate in results:
                result_dict[int(score)] = float(churn_rate)
            
            return result_dict
        
        except Exception as e:
            logger.error(f"Error computing satisfaction churn: {str(e)}")
            return {}
    
    def compute_churn_by_tenure_segment(self) -> Dict:
        """Churn rate by customer lifecycle segment"""
        
        try:
            query = f"""
            SELECT
                CASE
                    WHEN "Tenure" <= 3 THEN 'New (0-3m)'
                    WHEN "Tenure" <= 12 THEN 'Growth (4-12m)'
                    ELSE 'Mature (13+m)'
                END as tenure_segment,
                COUNT(*) as count,
                ROUND(CAST(SUM(CASE WHEN "Churn" = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 4) as churn_rate
            FROM {self.table_name}
            GROUP BY tenure_segment
            ORDER BY tenure_segment
            """
            
            results = self.conn.execute(query).fetchall()
            
            result_dict = {}
            for segment, count, churn_rate in results:
                result_dict[segment] = float(churn_rate)
            
            return result_dict
        
        except Exception as e:
            logger.error(f"Error computing tenure segment churn: {str(e)}")
            return {}
    
    def identify_high_risk_customers(self, limit: int = 50) -> pd.DataFrame:
        """Identify high-risk customers using composite scoring"""
        
        try:
            query = f"""
            SELECT
                "Tenure",
                "SatisfactionScore",
                "DaySinceLastOrder",
                "Complain",
                (
                    CASE WHEN "SatisfactionScore" <= 2 THEN 25 ELSE 0 END +
                    CASE WHEN "DaySinceLastOrder" > 10 THEN 25 ELSE 0 END +
                    CASE WHEN "Complain" = 1 THEN 20 ELSE 0 END +
                    CASE WHEN "Tenure" < 3 THEN 30 ELSE 0 END
                ) as risk_score
            FROM {self.table_name}
            WHERE (
                CASE WHEN "SatisfactionScore" <= 2 THEN 25 ELSE 0 END +
                CASE WHEN "DaySinceLastOrder" > 10 THEN 25 ELSE 0 END +
                CASE WHEN "Complain" = 1 THEN 20 ELSE 0 END +
                CASE WHEN "Tenure" < 3 THEN 30 ELSE 0 END
            ) >= 50
            ORDER BY risk_score DESC
            LIMIT {limit}
            """
            
            results = self.conn.execute(query).fetchall()
            
            df = pd.DataFrame(
                results,
                columns=['Tenure', 'SatisfactionScore', 'DaySinceLastOrder', 'Complain', 'RiskScore']
            )
            
            return df
        
        except Exception as e:
            logger.error(f"Error identifying high-risk customers: {str(e)}")
            return pd.DataFrame()
    
    def compute_percentile_metrics(self) -> Dict:
        """Compute percentile-based metrics"""
        
        try:
            query = f"""
            SELECT
                QUANTILE_CONT("Tenure", 0.25) as p25_tenure,
                QUANTILE_CONT("Tenure", 0.75) as p75_tenure,
                QUANTILE_CONT("DaySinceLastOrder", 0.25) as p25_days,
                QUANTILE_CONT("DaySinceLastOrder", 0.75) as p75_days,
                QUANTILE_CONT("CashbackAmount", 0.25) as p25_cashback,
                QUANTILE_CONT("CashbackAmount", 0.75) as p75_cashback
            FROM {self.table_name}
            """
            
            result = self.conn.execute(query).fetchall()[0]
            
            return {
                "tenure_percentiles": {
                    "p25": float(result[0]) if result[0] else 0,
                    "p75": float(result[1]) if result[1] else 0
                },
                "days_percentiles": {
                    "p25": float(result[2]) if result[2] else 0,
                    "p75": float(result[3]) if result[3] else 0
                },
                "cashback_percentiles": {
                    "p25": float(result[4]) if result[4] else 0,
                    "p75": float(result[5]) if result[5] else 0
                }
            }
        
        except Exception as e:
            logger.error(f"Error computing percentiles: {str(e)}")
            return {}
    
    def compute_churn_by_segment(self, segment_column: str) -> Dict:
        """Compute churn rate by any segment"""
        
        try:
            query = f"""
            SELECT
                "{segment_column}",
                COUNT(*) as count,
                ROUND(CAST(SUM(CASE WHEN "Churn" = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 4) as churn_rate
            FROM {self.table_name}
            GROUP BY "{segment_column}"
            ORDER BY churn_rate DESC
            """
            
            results = self.conn.execute(query).fetchall()
            
            result_dict = {}
            for segment_value, count, churn_rate in results:
                result_dict[str(segment_value)] = {
                    "count": int(count),
                    "churn_rate": float(churn_rate)
                }
            
            return result_dict
        
        except Exception as e:
            logger.error(f"Error computing segment churn: {str(e)}")
            return {}
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get entire table as Pandas DataFrame"""
        
        try:
            return self.conn.execute(f"SELECT * FROM {self.table_name}").df()
        except Exception as e:
            logger.error(f"Error getting dataframe: {str(e)}")
            return pd.DataFrame()
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute custom SQL query"""
        
        try:
            return self.conn.execute(sql).df()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close DuckDB connection"""
        try:
            self.conn.close()
            logger.info("✓ DuckDB connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")


# Global DuckDB processor instance
_duckdb_processor = None


def get_duckdb_processor() -> DuckDBProcessor:
    """Get or create global DuckDB processor (singleton)"""
    global _duckdb_processor
    
    if _duckdb_processor is None:
        _duckdb_processor = DuckDBProcessor()
    
    return _duckdb_processor


def initialize_duckdb(data_path: str = "data/raw/ecommerce_data.csv") -> bool:
    """Initialize DuckDB and load data on startup"""
    global _duckdb_processor
    
    processor = get_duckdb_processor()
    success = processor.load_csv(data_path)
    
    if success:
        logger.info("✓ DuckDB processor initialized and data loaded")
    else:
        logger.warning("⚠ DuckDB initialized but data loading failed")
    
    return success