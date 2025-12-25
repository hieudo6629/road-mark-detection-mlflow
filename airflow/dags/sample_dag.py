"""
Sample DAG to demonstrate Airflow setup.
This DAG runs a simple Python task every day.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def hello_world():
    """Simple hello world function."""
    print("Hello from Airflow!")
    print("This is a sample DAG to verify the Airflow setup.")
    return "Success"


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'sample_dag',
    default_args=default_args,
    description='A simple sample DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['example', 'sample'],
) as dag:

    # Define the task
    hello_task = PythonOperator(
        task_id='hello_world_task',
        python_callable=hello_world,
    )

    hello_task
