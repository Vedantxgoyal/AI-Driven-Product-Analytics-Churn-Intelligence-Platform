"""
s3_simulator.py - Simulated AWS S3 for local development
Mimics S3 bucket structure for data lake
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class S3Simulator:
    """Simulates AWS S3 for local development"""
    
    def __init__(self, bucket_name: str = "churn-data-lake"):
        """Initialize S3 simulator"""
        self.bucket_name = bucket_name
        self.bucket_path = os.path.join("s3_data", bucket_name)
        
        # Create bucket directory
        os.makedirs(self.bucket_path, exist_ok=True)
        
        # Create standard S3 prefixes
        self.prefixes = {
            "raw": os.path.join(self.bucket_path, "raw"),
            "processed": os.path.join(self.bucket_path, "processed"),
            "analytics": os.path.join(self.bucket_path, "analytics"),
            "models": os.path.join(self.bucket_path, "models"),
            "logs": os.path.join(self.bucket_path, "logs")
        }
        
        # Create prefix directories
        for prefix_path in self.prefixes.values():
            os.makedirs(prefix_path, exist_ok=True)
        
        logger.info(f"✓ S3 simulator initialized: s3://{bucket_name}/")
    
    def upload_file(self, local_path: str, s3_key: str, prefix: str = "raw") -> bool:
        """Upload file to simulated S3"""
        
        try:
            # Get prefix path
            prefix_path = self.prefixes.get(prefix)
            if not prefix_path:
                logger.error(f"Unknown prefix: {prefix}")
                return False
            
            # Create key path
            s3_path = os.path.join(prefix_path, s3_key)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(s3_path), exist_ok=True)
            
            # Copy file
            shutil.copy(local_path, s3_path)
            
            logger.info(f"✓ Uploaded {s3_key} to s3://{self.bucket_name}/{prefix}/")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            return False
    
    def upload_dataframe_csv(self, df, s3_key: str, prefix: str = "processed") -> bool:
        """Upload DataFrame as CSV to simulated S3"""
        
        try:
            prefix_path = self.prefixes.get(prefix)
            if not prefix_path:
                return False
            
            s3_path = os.path.join(prefix_path, s3_key)
            os.makedirs(os.path.dirname(s3_path), exist_ok=True)
            
            # Convert Spark DF to Pandas and save
            if hasattr(df, 'toPandas'):
                pandas_df = df.toPandas()
            else:
                pandas_df = df
            
            pandas_df.to_csv(s3_path, index=False)
            
            logger.info(f"✓ Uploaded DataFrame to s3://{self.bucket_name}/{prefix}/{s3_key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {str(e)}")
            return False
    
    def upload_json(self, data: Dict, s3_key: str, prefix: str = "analytics") -> bool:
        """Upload JSON data to simulated S3"""
        
        try:
            prefix_path = self.prefixes.get(prefix)
            if not prefix_path:
                return False
            
            s3_path = os.path.join(prefix_path, s3_key)
            os.makedirs(os.path.dirname(s3_path), exist_ok=True)
            
            with open(s3_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✓ Uploaded JSON to s3://{self.bucket_name}/{prefix}/{s3_key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upload JSON: {str(e)}")
            return False
    
    def list_objects(self, prefix: str = "raw") -> List[Dict]:
        """List objects in S3 prefix"""
        
        try:
            prefix_path = self.prefixes.get(prefix)
            if not prefix_path:
                return []
            
            objects = []
            for root, dirs, files in os.walk(prefix_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # Get file key (relative path)
                    key = os.path.relpath(filepath, prefix_path)
                    
                    # Get file stats
                    stat = os.stat(filepath)
                    
                    objects.append({
                        "key": key,
                        "size": stat.st_size,
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": "file"
                    })
            
            return objects
        
        except Exception as e:
            logger.error(f"Failed to list objects: {str(e)}")
            return []
    
    def get_object(self, s3_key: str, prefix: str = "processed") -> Optional[str]:
        """Get object from S3"""
        
        try:
            prefix_path = self.prefixes.get(prefix)
            if not prefix_path:
                return None
            
            s3_path = os.path.join(prefix_path, s3_key)
            
            if os.path.exists(s3_path):
                with open(s3_path, 'r') as f:
                    return f.read()
            
            logger.warning(f"Object not found: {s3_key}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to get object: {str(e)}")
            return None
    
    def get_bucket_stats(self) -> Dict:
        """Get bucket statistics"""
        
        try:
            stats = {
                "bucket_name": self.bucket_name,
                "total_size_mb": 0,
                "total_files": 0,
                "prefixes": {}
            }
            
            for prefix_name, prefix_path in self.prefixes.items():
                prefix_size = 0
                prefix_files = 0
                
                for root, dirs, files in os.walk(prefix_path):
                    for file in files:
                        filepath = os.path.join(root, file)
                        prefix_size += os.path.getsize(filepath)
                        prefix_files += 1
                
                stats["prefixes"][prefix_name] = {
                    "size_mb": round(prefix_size / (1024 * 1024), 2),
                    "file_count": prefix_files
                }
                
                stats["total_size_mb"] += round(prefix_size / (1024 * 1024), 2)
                stats["total_files"] += prefix_files
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get bucket stats: {str(e)}")
            return {}
    
    def delete_prefix(self, prefix: str) -> bool:
        """Delete all objects in prefix"""
        
        try:
            prefix_path = self.prefixes.get(prefix)
            if not prefix_path:
                return False
            
            # Remove all files but keep directory
            for root, dirs, files in os.walk(prefix_path):
                for file in files:
                    os.remove(os.path.join(root, file))
            
            # Remove empty directories
            for root, dirs, files in os.walk(prefix_path, topdown=False):
                for dir_name in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir_name))
                    except:
                        pass
            
            logger.info(f"✓ Deleted all objects in {prefix} prefix")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete prefix: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up S3 simulator"""
        
        try:
            shutil.rmtree(self.bucket_path)
            os.makedirs(self.bucket_path, exist_ok=True)
            
            # Recreate prefixes
            for prefix_path in self.prefixes.values():
                os.makedirs(prefix_path, exist_ok=True)
            
            logger.info("✓ S3 simulator cleaned")
        except Exception as e:
            logger.error(f"Failed to cleanup: {str(e)}")


# Global S3 instance
_s3 = None


def get_s3() -> S3Simulator:
    """Get or create global S3 simulator (singleton)"""
    global _s3
    
    if _s3 is None:
        _s3 = S3Simulator()
    
    return _s3


def initialize_s3(bucket_name: str = "churn-data-lake") -> S3Simulator:
    """Initialize S3 on startup"""
    global _s3
    _s3 = S3Simulator(bucket_name)
    return _s3