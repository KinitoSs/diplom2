from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ETL_Clean") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

# Пример: читаем parquet из staging
df = spark.read.parquet("s3a://mlflow-artifacts/staging/")

# Простая очистка null
df_clean = df.dropna()

# Сохраняем обратно
df_clean.write.mode("overwrite").parquet("s3a://mlflow-artifacts/staging_clean/")

spark.stop()
