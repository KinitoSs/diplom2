from kafka import KafkaProducer
import os
import time
import boto3
from botocore.client import Config

FOLDER = "/data/photos"
CHECK_INTERVAL = 2  # секунды


def ensure_bucket(bucket_name):
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]

    if bucket_name not in buckets:
        print(f"Bucket '{bucket_name}' not found. Creating...")
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created!")
    else:
        print(f"Bucket '{bucket_name}' already exists.")


producer = KafkaProducer(bootstrap_servers="kafka:9092", value_serializer=lambda v: v)

processed = set()

ensure_bucket("raw")
ensure_bucket("staged")


def is_image(file):
    return file.lower().endswith((".jpg", ".jpeg", ".png"))


while True:
    try:
        files = os.listdir(FOLDER)

        for file in files:
            path = os.path.join(FOLDER, file)

            # пропускаем не картинки
            if not is_image(file):
                continue

            # пропускаем уже обработанные
            if path in processed:
                continue

            # проверка: файл полностью записан (важно!)
            if not os.path.isfile(path):
                continue

            try:
                with open(path, "rb") as f:
                    img_bytes = f.read()

                producer.send("images", img_bytes)
                producer.flush()

                print(f"Sent: {file}")

                # удаляем после отправки
                os.remove(path)

                processed.add(path)

            except Exception as e:
                print(f"Error processing {file}: {e}")

        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print(f"Watcher error: {e}")
        time.sleep(5)
