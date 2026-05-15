"""
job_market_dag.py
-----------------
Apache Airflow DAG that orchestrates the full Job Market Analyzer pipeline.

Schedule: Daily at 6:00 AM UTC
Pipeline: Ingest → Transform (Bronze→Silver→Gold) → Summarize → Alert

DAG Structure:
    ingest_jobs → transform_data → generate_summary → pipeline_success
                       ↓ (on failure)
                   notify_failure
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email  import EmailOperator
from airflow.utils.trigger_rule import TriggerRule

# Import pipeline functions
import sys
sys.path.append("/opt/airflow/project")

from src.ingestion.api_ingestor       import run_ingestion
from src.transformation.transformer   import run_transformation
from src.summarization.llm_summarizer import run_summarization

# ── Default args ───────────────────────────────────────────────────
default_args = {
    "owner":            "arpitha_raghu",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "email":            ["arpithagowda2205@gmail.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          3,                          # Retry 3 times on failure
    "retry_delay":      timedelta(minutes=5),       # Wait 5 min between retries
}

# ── DAG Definition ─────────────────────────────────────────────────
with DAG(
    dag_id="job_market_pipeline",
    default_args=default_args,
    description="Daily job market data pipeline: ingest → transform → summarize",
    schedule_interval="0 6 * * *",          # Every day at 6:00 AM UTC
    catchup=False,                           # Don't backfill missed runs
    max_active_runs=1,                       # Only 1 run at a time
    tags=["data-engineering", "etl", "aws", "llm"],
) as dag:

    # ── Task 1: Ingest raw job data from API → S3 Bronze ──────────
    ingest_task = PythonOperator(
        task_id="ingest_jobs",
        python_callable=run_ingestion,
        op_kwargs={"run_date": "{{ ds }}"},  # Airflow passes the execution date
        doc_md="""
        **Ingest Jobs**
        Fetches job postings from JSearch API for configured job titles.
        Uploads raw JSON to S3 Bronze layer partitioned by date.
        Expected: ~200 records per run across 4 job categories.
        """
    )

    # ── Task 2: Transform Bronze → Silver → Gold ───────────────────
    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=run_transformation,
        op_kwargs={"run_date": "{{ ds }}"},
        doc_md="""
        **Transform Data**
        Bronze → Silver: dedup, clean, standardize, cast types.
        Silver → Gold: skill frequency, salary stats, remote trends.
        Uploads Parquet (Silver) and JSON insights (Gold) to S3.
        """
    )

    # ── Task 3: LLM Summarization ──────────────────────────────────
    summarize_task = PythonOperator(
        task_id="generate_summary",
        python_callable=run_summarization,
        op_kwargs={"run_date": "{{ ds }}"},
        doc_md="""
        **Generate LLM Summary**
        Reads Gold insights and sends to OpenAI GPT for summarization.
        Produces a human-readable market report saved to S3.
        """
    )

    # ── Task 4: Success notification ──────────────────────────────
    success_task = EmailOperator(
        task_id="pipeline_success",
        to="arpithagowda2205@gmail.com",
        subject="✅ Job Market Pipeline Success — {{ ds }}",
        html_content="""
        <h3>Job Market Pipeline completed successfully!</h3>
        <p><b>Run Date:</b> {{ ds }}</p>
        <p><b>Status:</b> All stages completed — Ingest, Transform, Summarize</p>
        <p>Check S3 for today's market summary report.</p>
        """,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    # ── Task 5: Failure notification ──────────────────────────────
    failure_task = EmailOperator(
        task_id="notify_failure",
        to="arpithagowda2205@gmail.com",
        subject="❌ Job Market Pipeline FAILED — {{ ds }}",
        html_content="""
        <h3>Job Market Pipeline FAILED!</h3>
        <p><b>Run Date:</b> {{ ds }}</p>
        <p>Please check the Airflow logs for details.</p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED   # Fires if ANY task fails
    )

    # ── Task Dependencies ──────────────────────────────────────────
    # Linear pipeline with parallel failure notification
    ingest_task >> transform_task >> summarize_task >> success_task
    [ingest_task, transform_task, summarize_task] >> failure_task
