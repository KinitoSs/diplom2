from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MinIO-ToLocal").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- MinIO config ---
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "minioadmin")
hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# --- schema ---
from pyspark.sql.types import StructType, StructField, BinaryType

schema = StructType([StructField("image", BinaryType(), True)])

# --- streaming read ---
df = (
    spark.readStream.schema(schema)
    .format("parquet")
    .load("s3a://staged/images_parquet/")
)


# --- обработка батча ---
def process_batch(batch_df, batch_id):
    print(f"Processing batch {batch_id}")

    rows = batch_df.collect()

    for i, row in enumerate(rows):
        image_bytes = row["image"]

        file_path = f"/data/filtered_photos/image_{batch_id}_{i}.jpg"

        with open(file_path, "wb") as f:
            f.write(image_bytes)


# --- запуск ---
query = (
    df.writeStream.foreachBatch(process_batch)
    .option("checkpointLocation", "s3a://staged/to_local_checkpoints/")
    .start()
)

query.awaitTermination()
