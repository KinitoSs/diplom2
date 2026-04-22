from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime


with DAG(
    dag_id="automarkup_to_local",
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
) as dag:
    automarkup_to_local = DockerOperator(
        task_id="automarkup_to_local",
        image="diplom2-spark-master:latest",
        command="/opt/spark/bin/spark-submit "
                "--master spark://spark-master:7077 "
                "--driver-memory 2g "
                "--executor-memory 2g "
                "--executor-cores 2 "
                "/app/automarkup_to_local.py",
        network_mode="diplom2_default",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        container_name="automarkup_to_local",
        mounts=[
            {
                "source": "E:/kin_hdd/Study/12/diplom2/automarkup_photos",
                "target": "/data/automarkup_photos",
                "type": "bind"
            }
        ],
    )