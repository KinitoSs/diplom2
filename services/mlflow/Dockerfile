FROM ghcr.io/mlflow/mlflow:v2.0.1

# Устанавливаем необходимые пакеты один раз
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache-dir psycopg2-binary boto3
