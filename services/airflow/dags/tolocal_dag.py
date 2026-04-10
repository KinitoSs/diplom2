from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="tolocal",
    start_date=datetime(2024, 1, 1),
    schedule_interval="*/5 * * * *",  # каждые 5 минут
    catchup=False,
) as dag:
    to_local = BashOperator(
        task_id="to_local",
        bash_command="""
        docker exec spark-tolocal /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        /app/ETL/to_local.py
        """,
    )
