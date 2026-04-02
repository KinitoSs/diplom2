#!/bin/bash

echo "Starting RAW streaming..."
/opt/spark/bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4 \
  /opt/etl/structured_streaming.py &

sleep 10

echo "Starting ETL streaming..."
/opt/spark/bin/spark-submit \
  --packages org.apache.hadoop:hadoop-aws:3.3.4 \
  /opt/etl/etl.py &

wait
