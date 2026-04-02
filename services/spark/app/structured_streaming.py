from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("KafkaToMinIO").getOrCreate()

hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "minioadmin")
hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "images")
    .option("startingOffsets", "earliest")
    .load()
)

images = df.selectExpr("CAST(value AS BINARY) as image")

query = (
    images.writeStream.format("parquet")
    .option("path", "s3a://raw/images_parquet/")
    .option("checkpointLocation", "s3a://raw/checkpoints/")
    .start()
)

query.awaitTermination()
