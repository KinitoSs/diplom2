from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime


with DAG(
    dag_id="tolocal",
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
) as dag:
    tolocal = DockerOperator(
        task_id="to_local",
        image="diplom2-spark-master:latest",
        command="/opt/spark/bin/spark-submit --master spark://spark-master:7077 /app/ETL/to_local.py",
        network_mode="diplom2_default",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        container_name="to_local",
        mounts=[
            {
                "source": "E:/kin_hdd/Study/12/diplom2/filtered_photos",  # Абсолютный путь на хосте
                "target": "/data/filtered_photos",
                "type": "bind"
            }
        ],
    )
