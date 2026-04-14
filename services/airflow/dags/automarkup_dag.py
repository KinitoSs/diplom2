from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime

with DAG(
    dag_id="automarkup",
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
) as dag:
    automarkup = DockerOperator(
        task_id="run_automarkup",
        image="diplom2-automarkup-init:latest",
        command="python /app/inference.py",  # если скрипт лежит в корне
        network_mode="diplom2_default",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        container_name="automarkup",
        environment={
            "MINIO_ENDPOINT": "minio:9000",
            "MINIO_ACCESS_KEY": "minioadmin",
            "MINIO_SECRET_KEY": "minioadmin",
            "SOURCE_BUCKET": "staged",
            "TARGET_BUCKET": "automarkup",
            "MODEL_TYPE": "mobilenet",
            "BATCH_SIZE": "16"
        },
        mount_tmp_dir=False,
    )