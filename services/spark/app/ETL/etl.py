from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length

spark = SparkSession.builder.appName("MinIO-ETL").getOrCreate()

hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "minioadmin")
hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# читаем raw слой как стрим
df = spark.readStream.format("parquet").load("s3a://raw/images_parquet/")

# 🔥 ПРИМЕР ФИЛЬТРАЦИИ
# например: отбрасываем слишком маленькие "картинки"
filtered = df.filter(length(col("image")) > 1000)

# можно добавить любые ML / OpenCV позже

query = (
    filtered.writeStream.format("parquet")
    .option("path", "s3a://staged/images/")
    .option("checkpointLocation", "s3a://staged/checkpoints/")
    .start()
)

query.awaitTermination()
