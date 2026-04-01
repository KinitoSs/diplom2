from kafka import KafkaProducer
import os
import sys

if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = "/data/photos"

producer = KafkaProducer(bootstrap_servers="kafka:9092", value_serializer=lambda v: v)

print(f"Using folder: {folder}")
print("Start sending images...")

sended_files = 0

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    if not file.lower().endswith((".jpg", ".png")):
        continue

    with open(path, "rb") as f:
        img_bytes = f.read()

    producer.send("images", img_bytes)
    print(f"Sent: {file}")
    sended_files += 1

producer.flush()

if sended_files > 0:
    print("Done!")
else:
    print("No files sent!")
