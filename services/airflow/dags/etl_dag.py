from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime

with DAG(
    dag_id="etl",
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
) as dag:
    etl = DockerOperator(
        task_id="run_etl",
        image="diplom2-spark-master:latest",
        command="/opt/spark/bin/spark-submit "
                "--master spark://spark-master:7077 "
                "--driver-memory 2g "
                "--executor-memory 2g "
                "--executor-cores 2 "
                "/app/ETL/etl.py",
        network_mode="diplom2_default",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        container_name="etl",
    )