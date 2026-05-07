from churn_api.app.duckdb_processor import initialize_spark, get_spark_processor
from hive_warehouse import initialize_warehouse
from s3_simulator import initialize_s3

print("="*70)
print("🚀 TESTING ENTERPRISE ARCHITECTURE")
print("="*70)

# Test 1: Spark
print("\n[1/3] Testing Spark Processor...")
try:
    initialize_spark("data/raw/ecommerce_data.csv")
    processor = get_spark_processor()
    metrics = processor.compute_churn_metrics()
    print(f"✓ Spark loaded {metrics['total_customers']:,} customers")
    print(f"✓ Churn Rate: {metrics['churn_rate']:.2%}")
except Exception as e:
    print(f"❌ Spark Error: {str(e)}")

# Test 2: Hive Warehouse
print("\n[2/3] Testing Hive Warehouse...")
try:
    warehouse = initialize_warehouse()
    warehouse.insert_analytics(metrics)
    print("✓ Warehouse created and data stored")
    info = warehouse.get_table_info()
    print(f"✓ Warehouse has {len(info)} tables")
except Exception as e:
    print(f"❌ Hive Error: {str(e)}")

# Test 3: S3 Simulator
print("\n[3/3] Testing S3 Simulator...")
try:
    s3 = initialize_s3()
    s3.upload_json(metrics, "test_metrics.json", prefix="analytics")
    print("✓ S3 bucket created and data uploaded")
    stats = s3.get_bucket_stats()
    print(f"✓ S3 bucket has {stats['total_files']} files")
except Exception as e:
    print(f"❌ S3 Error: {str(e)}")

print("\n" + "="*70)
print("✅ ENTERPRISE ARCHITECTURE TEST COMPLETE")
print("="*70)