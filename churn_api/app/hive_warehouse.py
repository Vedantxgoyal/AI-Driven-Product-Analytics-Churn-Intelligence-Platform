"""
hive_warehouse.py - Simulated Hive data warehouse
Stores processed data in a structured, queryable format
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class HiveWarehouse:
    """Simulated Hive warehouse for storing processed analytics"""
    
    def __init__(self, warehouse_path: str = "warehouse"):
        """Initialize Hive warehouse"""
        self.warehouse_path = warehouse_path
        self.tables = {}
        
        # Create warehouse directory
        os.makedirs(warehouse_path, exist_ok=True)
        
        # Initialize default tables
        self._create_tables()
        
        logger.info(f"✓ Hive warehouse initialized at {warehouse_path}")
    
    def _create_tables(self):
        """Create default warehouse tables"""
        
        self.tables = {
            "customers": {
                "location": os.path.join(self.warehouse_path, "customers"),
                "schema": {
                    "customer_id": "int",
                    "tenure": "float",
                    "satisfaction_score": "int",
                    "churn": "int"
                }
            },
            "analytics": {
                "location": os.path.join(self.warehouse_path, "analytics"),
                "schema": {
                    "metric_name": "string",
                    "metric_value": "float",
                    "timestamp": "string"
                }
            },
            "predictions": {
                "location": os.path.join(self.warehouse_path, "predictions"),
                "schema": {
                    "customer_id": "int",
                    "churn_probability": "float",
                    "risk_level": "string",
                    "timestamp": "string"
                }
            },
            "transformations": {
                "location": os.path.join(self.warehouse_path, "transformations"),
                "schema": {
                    "transformation_id": "string",
                    "input_rows": "int",
                    "output_rows": "int",
                    "duration_ms": "float",
                    "timestamp": "string"
                }
            }
        }
        
        # Create table directories
        for table_name, table_info in self.tables.items():
            os.makedirs(table_info["location"], exist_ok=True)
            
            # Create schema file
            schema_file = os.path.join(table_info["location"], "_schema.json")
            if not os.path.exists(schema_file):
                with open(schema_file, 'w') as f:
                    json.dump(table_info["schema"], f, indent=2)
        
        logger.info(f"✓ Created {len(self.tables)} warehouse tables")
    
    def insert_analytics(self, metrics: Dict[str, Any]) -> bool:
        """Insert analytics metrics into warehouse"""
        
        try:
            analytics_dir = self.tables["analytics"]["location"]
            timestamp = datetime.utcnow().isoformat()
            
            # Create partition by date
            date_partition = datetime.utcnow().strftime("%Y/%m/%d")
            partition_dir = os.path.join(analytics_dir, date_partition)
            os.makedirs(partition_dir, exist_ok=True)
            
            # Write metrics
            filename = os.path.join(
                partition_dir,
                f"metrics_{timestamp.replace(':', '-')}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "metrics": metrics
                }, f, indent=2)
            
            logger.info(f"✓ Inserted analytics metrics into warehouse")
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert analytics: {str(e)}")
            return False
    
    def insert_predictions(self, predictions: List[Dict]) -> bool:
        """Insert predictions into warehouse"""
        
        try:
            predictions_dir = self.tables["predictions"]["location"]
            timestamp = datetime.utcnow().isoformat()
            
            # Create partition by date
            date_partition = datetime.utcnow().strftime("%Y/%m/%d")
            partition_dir = os.path.join(predictions_dir, date_partition)
            os.makedirs(partition_dir, exist_ok=True)
            
            # Write predictions
            filename = os.path.join(
                partition_dir,
                f"predictions_{timestamp.replace(':', '-')}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "predictions": predictions,
                    "count": len(predictions)
                }, f, indent=2)
            
            logger.info(f"✓ Inserted {len(predictions)} predictions into warehouse")
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert predictions: {str(e)}")
            return False
    
    def insert_transformation(self, transformation_id: str, 
                             input_rows: int, output_rows: int, 
                             duration_ms: float) -> bool:
        """Log transformation to warehouse"""
        
        try:
            transform_dir = self.tables["transformations"]["location"]
            timestamp = datetime.utcnow().isoformat()
            
            # Create partition by date
            date_partition = datetime.utcnow().strftime("%Y/%m/%d")
            partition_dir = os.path.join(transform_dir, date_partition)
            os.makedirs(partition_dir, exist_ok=True)
            
            # Write transformation record
            filename = os.path.join(
                partition_dir,
                f"transform_{transformation_id}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump({
                    "transformation_id": transformation_id,
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                    "duration_ms": duration_ms,
                    "timestamp": timestamp
                }, f, indent=2)
            
            logger.info(f"✓ Logged transformation {transformation_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to log transformation: {str(e)}")
            return False
    
    def query_analytics(self, days: int = 7) -> List[Dict]:
        """Query analytics from warehouse (simple implementation)"""
        
        try:
            analytics_dir = self.tables["analytics"]["location"]
            results = []
            
            # Walk through date partitions
            for root, dirs, files in os.walk(analytics_dir):
                for file in files:
                    if file.endswith(".json") and file != "_schema.json":
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            results.append(data)
            
            logger.info(f"✓ Queried {len(results)} analytics records")
            return results
        
        except Exception as e:
            logger.error(f"Failed to query analytics: {str(e)}")
            return []
    
    def query_predictions(self, limit: int = 100) -> List[Dict]:
        """Query predictions from warehouse"""
        
        try:
            predictions_dir = self.tables["predictions"]["location"]
            all_predictions = []
            
            # Walk through date partitions
            for root, dirs, files in os.walk(predictions_dir):
                for file in files:
                    if file.endswith(".json") and file != "_schema.json":
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            all_predictions.extend(data.get("predictions", []))
            
            # Return latest
            return all_predictions[-limit:]
        
        except Exception as e:
            logger.error(f"Failed to query predictions: {str(e)}")
            return []
    
    def get_table_info(self) -> Dict:
        """Get information about warehouse tables"""
        
        table_info = {}
        for table_name, table_data in self.tables.items():
            location = table_data["location"]
            
            # Count files
            file_count = 0
            size_bytes = 0
            for root, dirs, files in os.walk(location):
                for file in files:
                    if file.endswith(".json"):
                        file_count += 1
                        filepath = os.path.join(root, file)
                        size_bytes += os.path.getsize(filepath)
            
            table_info[table_name] = {
                "location": location,
                "file_count": file_count,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "schema": table_data["schema"]
            }
        
        return table_info
    
    def cleanup(self):
        """Clean up warehouse data (for testing)"""
        
        import shutil
        try:
            shutil.rmtree(self.warehouse_path)
            os.makedirs(self.warehouse_path, exist_ok=True)
            self._create_tables()
            logger.info("✓ Warehouse cleaned")
        except Exception as e:
            logger.error(f"Failed to cleanup warehouse: {str(e)}")


# Global warehouse instance
_warehouse = None


def get_warehouse() -> HiveWarehouse:
    """Get or create global warehouse (singleton)"""
    global _warehouse
    
    if _warehouse is None:
        _warehouse = HiveWarehouse()
    
    return _warehouse


def initialize_warehouse(path: str = "warehouse") -> HiveWarehouse:
    """Initialize warehouse on startup"""
    global _warehouse
    _warehouse = HiveWarehouse(path)
    return _warehouse