"""
ML Training Pipeline DAG
Orchestrates YOLO training, model logging, and model registration.
Each step runs in a Docker container and only proceeds if the previous step succeeds.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Get the project root directory (3 levels up from this DAG file)
# DAG file is at: <project>/airflow/dags/ml_training_pipeline.py
# Project root is: <project>/
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.absolute())

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    "ml_training_pipeline",
    default_args=default_args,
    description="YOLO training, logging, and registration pipeline",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=["ml", "yolo", "training", "mlflow"],
) as dag:

    # Task 1: Train YOLO model
    train_yolo = DockerOperator(
        task_id="train_yolo",
        image="python:3.11-slim",
        api_version="auto",
        auto_remove=True,
        command=[
            "bash",
            "-c",
            """
            # set -e
            # Install dependencies
            pip install --no-cache-dir ultralytics mlflow boto3 psycopg2-binary
            
            # Run training script
            cd /workspace
            python scripts/road_mark_detection.py
            
            echo "✅ Training completed successfully"
            """,
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="road-mark-detection-mlflow_default",  # Use the same network as docker-compose
        mounts=[
            Mount(
                source=PROJECT_ROOT,
                target="/workspace",
                type="bind",
            )
        ],
        mount_tmp_dir=False,
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        },
    )

    # Task 2: Log model to MLflow
    log_model = DockerOperator(
        task_id="log_model",
        image="python:3.11-slim",
        api_version="auto",
        auto_remove=True,
        command=[
            "bash",
            "-c",
            """
            set -e
            # Install dependencies
            pip install --no-cache-dir ultralytics mlflow boto3 psycopg2-binary
            
            # Run model logging script
            cd /workspace
            python scripts/log_model.py
            
            echo "✅ Model logging completed successfully"
            """,
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="road-mark-detection-mlflow_default",
        mounts=[
            Mount(
                source=PROJECT_ROOT,
                target="/workspace",
                type="bind",
            )
        ],
        mount_tmp_dir=False,
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        },
    )

    # Task 3: Register model and promote to Production
    register_model = DockerOperator(
        task_id="register_model",
        image="python:3.11-slim",
        api_version="auto",
        auto_remove=True,
        command=[
            "bash",
            "-c",
            """
            # set -e
            # Install dependencies
            pip install --no-cache-dir mlflow boto3 psycopg2-binary
            
            # Run model registration script
            cd /workspace
            python scripts/register_model.py
            
            echo "✅ Model registration completed successfully"
            """,
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="road-mark-detection-mlflow_default",
        mounts=[
            Mount(
                source=PROJECT_ROOT,
                target="/workspace",
                type="bind",
            )
        ],
        mount_tmp_dir=False,
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "AWS_ACCESS_KEY_ID": "minio",
            "AWS_SECRET_ACCESS_KEY": "minio123",
        },
    )

    # Define task dependencies (sequential execution)
    train_yolo >> log_model >> register_model
