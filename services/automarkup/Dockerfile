FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache-dir -r requirements.txt

COPY DeepLabV3_MobileNetV3_best.pth .
COPY . .