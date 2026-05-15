"""
transformer.py
--------------
Reads raw JSON from the Bronze S3 layer, cleans and standardizes
the data (Silver layer), then aggregates insights (Gold layer).

Bronze → Silver : dedup, nulls, type casting, column standardization
Silver → Gold   : skill frequency, salary stats, remote trends
"""

import json
import logging
import re
import boto3
import pandas as pd
from io import BytesIO
from datetime import datetime
from config.config import (
    S3_BUCKET, S3_BRONZE_PREFIX, S3_SILVER_PREFIX, S3_GOLD_PREFIX,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Skills to track across job descriptions
TRACKED_SKILLS = [
    "python", "sql", "spark", "airflow", "kafka", "aws", "azure", "gcp",
    "snowflake", "redshift", "databricks", "dbt", "hadoop", "scala",
    "pandas", "numpy", "docker", "kubernetes", "terraform", "git",
    "tableau", "power bi", "looker", "glue", "lambda", "s3", "pyspark"
]


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


# ── Bronze → Silver ────────────────────────────────────────────────

def load_bronze_data(s3_client, run_date: str) -> pd.DataFrame:
    """
    Load all Bronze JSON files for a given run date into a DataFrame.

    Args:
        s3_client: Boto3 S3 client
        run_date:  Pipeline run date (YYYY-MM-DD)

    Returns:
        Combined DataFrame of all raw job records
    """
    prefix   = f"{S3_BRONZE_PREFIX}/{run_date}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    files    = response.get("Contents", [])

    if not files:
        raise FileNotFoundError(f"No Bronze files found for date={run_date}")

    all_records = []
    for obj in files:
        raw = s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
        data = json.loads(raw["Body"].read())
        all_records.extend(data.get("data", []))

    logger.info(f"Loaded {len(all_records)} raw records from Bronze layer")
    return pd.DataFrame(all_records)


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw job data and standardize columns for Silver layer.

    Steps:
        1. Select and rename relevant columns
        2. Drop duplicates and null job titles
        3. Standardize employment type and remote flag
        4. Parse min/max salary from nested fields

    Args:
        df: Raw Bronze DataFrame

    Returns:
        Cleaned Silver DataFrame
    """
    logger.info("Starting Silver transformation...")

    # Select relevant fields (handle missing columns gracefully)
    cols_map = {
        "job_id":                  "job_id",
        "job_title":               "job_title",
        "employer_name":           "company",
        "job_city":                "city",
        "job_state":               "state",
        "job_country":             "country",
        "job_employment_type":     "employment_type",
        "job_is_remote":           "is_remote",
        "job_description":         "description",
        "job_min_salary":          "salary_min",
        "job_max_salary":          "salary_max",
        "job_salary_currency":     "salary_currency",
        "job_posted_at_timestamp": "posted_at",
        "job_required_experience": "required_experience",
    }

    existing_cols = {k: v for k, v in cols_map.items() if k in df.columns}
    df = df[list(existing_cols.keys())].rename(columns=existing_cols)

    # Drop duplicates and records with no job title
    before = len(df)
    df = df.drop_duplicates(subset=["job_id"]).dropna(subset=["job_title"])
    logger.info(f"Dropped {before - len(df)} duplicate/null records")

    # Standardize types
    df["is_remote"]    = df["is_remote"].fillna(False).astype(bool)
    df["salary_min"]   = pd.to_numeric(df.get("salary_min"), errors="coerce")
    df["salary_max"]   = pd.to_numeric(df.get("salary_max"), errors="coerce")
    df["salary_avg"]   = (df["salary_min"] + df["salary_max"]) / 2
    df["job_title"]    = df["job_title"].str.strip().str.title()
    df["company"]      = df["company"].str.strip()
    df["description"]  = df["description"].fillna("").str.lower()
    df["posted_at"]    = pd.to_datetime(df.get("posted_at"), unit="s", errors="coerce")
    df["ingested_at"]  = datetime.utcnow()

    logger.info(f"Silver transformation complete | {len(df)} clean records")
    return df


def upload_silver(s3_client, df: pd.DataFrame, run_date: str) -> str:
    """Upload cleaned DataFrame to Silver S3 layer as Parquet."""
    s3_key = f"{S3_SILVER_PREFIX}/{run_date}/jobs.parquet"
    buffer = BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)

    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue())
    logger.info(f"Silver data uploaded | s3://{S3_BUCKET}/{s3_key}")
    return s3_key


# ── Silver → Gold ──────────────────────────────────────────────────

def extract_skills(description: str) -> list:
    """Extract tracked skills mentioned in a job description."""
    return [skill for skill in TRACKED_SKILLS if re.search(rf"\b{re.escape(skill)}\b", description)]


def build_gold_insights(df: pd.DataFrame) -> dict:
    """
    Aggregate Silver data into Gold insights.

    Returns a dict with:
        - skill_counts    : skill → count
        - salary_by_title : job_title → avg salary
        - remote_stats    : % remote vs on-site
        - top_companies   : companies with most openings
    """
    logger.info("Building Gold insights...")

    # Skill frequency
    df["skills_found"] = df["description"].apply(extract_skills)
    skill_series       = df["skills_found"].explode()
    skill_counts       = skill_series.value_counts().head(20).to_dict()

    # Average salary by job title (only where salary data exists)
    salary_df     = df.dropna(subset=["salary_avg"])
    salary_by_title = (
        salary_df.groupby("job_title")["salary_avg"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(2)
        .to_dict()
    )

    # Remote vs on-site
    total         = len(df)
    remote_count  = df["is_remote"].sum()
    remote_stats  = {
        "remote_count":  int(remote_count),
        "onsite_count":  int(total - remote_count),
        "remote_pct":    round(remote_count / total * 100, 1) if total else 0
    }

    # Top hiring companies
    top_companies = df["company"].value_counts().head(10).to_dict()

    insights = {
        "run_date":       df["ingested_at"].max().strftime("%Y-%m-%d") if "ingested_at" in df else "N/A",
        "total_jobs":     total,
        "skill_counts":   skill_counts,
        "salary_by_title": salary_by_title,
        "remote_stats":   remote_stats,
        "top_companies":  top_companies
    }

    logger.info(f"Gold insights built | total_jobs={total}")
    return insights


def upload_gold(s3_client, insights: dict, run_date: str) -> str:
    """Upload Gold insights JSON to S3."""
    s3_key = f"{S3_GOLD_PREFIX}/{run_date}/insights.json"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(insights, indent=2),
        ContentType="application/json"
    )
    logger.info(f"Gold insights uploaded | s3://{S3_BUCKET}/{s3_key}")
    return s3_key


# ── Main entrypoint ────────────────────────────────────────────────

def run_transformation(run_date: str = None) -> dict:
    """
    Run full Bronze → Silver → Gold transformation.

    Args:
        run_date: Pipeline run date (YYYY-MM-DD). Defaults to today.

    Returns:
        Gold insights dictionary
    """
    run_date  = run_date or datetime.today().strftime("%Y-%m-%d")
    s3_client = get_s3_client()

    # Bronze → Silver
    raw_df    = load_bronze_data(s3_client, run_date)
    silver_df = clean_and_standardize(raw_df)
    upload_silver(s3_client, silver_df, run_date)

    # Silver → Gold
    insights = build_gold_insights(silver_df)
    upload_gold(s3_client, insights, run_date)

    return insights


if __name__ == "__main__":
    run_transformation()
