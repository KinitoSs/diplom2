from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, BinaryType

spark = SparkSession.builder.appName("MinIO-ETL").getOrCreate()

hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "minioadmin")
hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

schema = StructType([StructField("image", BinaryType(), True)])

df = spark.readStream.format("parquet").schema(schema).load("s3a://raw/images_parquet/")

filtered = df  # пока без фильтра

query = (
    filtered.writeStream.format("parquet")
    .option("path", "s3a://staged/images/")
    .option("checkpointLocation", "s3a://staged/checkpoints/")
    .start()
)

query.awaitTermination()
