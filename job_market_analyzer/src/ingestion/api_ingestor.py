"""
api_ingestor.py
---------------
Fetches job postings from the JSearch API (via RapidAPI) and uploads
raw JSON responses to the Bronze layer in AWS S3.

Bronze Layer = raw data, no transformations, preserved as-is.
"""

import json
import logging
import boto3
import requests
from datetime import datetime
from config.config import (
    RAPIDAPI_KEY, RAPIDAPI_HOST,
    S3_BUCKET, S3_BRONZE_PREFIX,
    JOB_QUERIES, JOBS_PER_PAGE, MAX_PAGES,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
)

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def get_s3_client():
    """Initialize and return an S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def fetch_jobs_from_api(query: str, page: int) -> dict:
    """
    Fetch one page of job postings from JSearch API.

    Args:
        query: Job title to search (e.g. 'Data Engineer')
        page:  Page number for pagination

    Returns:
        Raw API response as a dictionary
    """
    url = f"https://{RAPIDAPI_HOST}/search"
    headers = {
        "X-RapidAPI-Key":  RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    params = {
        "query":           query,
        "page":            page,
        "num_pages":       1,
        "date_posted":     "today",
        "employment_types": "FULLTIME"
    }

    logger.info(f"Fetching jobs | query='{query}' | page={page}")
    response = requests.get(url, headers=headers, params=params, timeout=30)

    # Raise an error for bad HTTP status codes (4xx, 5xx)
    response.raise_for_status()
    return response.json()


def upload_to_s3(s3_client, data: dict, query: str, page: int, run_date: str) -> str:
    """
    Upload raw API response JSON to S3 Bronze layer.

    S3 path: bronze/jobs/YYYY-MM-DD/data_engineer_page_1.json

    Args:
        s3_client: Boto3 S3 client
        data:      Raw API response dict
        query:     Job query string
        page:      Page number
        run_date:  Pipeline run date (YYYY-MM-DD)

    Returns:
        S3 key of the uploaded file
    """
    # Build a clean filename from the query string
    query_slug = query.lower().replace(" ", "_")
    s3_key = f"{S3_BRONZE_PREFIX}/{run_date}/{query_slug}_page_{page}.json"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json"
    )

    logger.info(f"Uploaded to S3 | s3://{S3_BUCKET}/{s3_key}")
    return s3_key


def run_ingestion(run_date: str = None) -> list:
    """
    Main ingestion function — fetches all job queries and uploads to S3.

    Args:
        run_date: Date string (YYYY-MM-DD). Defaults to today.

    Returns:
        List of S3 keys for all uploaded files
    """
    run_date = run_date or datetime.today().strftime("%Y-%m-%d")
    s3_client = get_s3_client()
    uploaded_keys = []
    total_records = 0

    logger.info(f"Starting ingestion | run_date={run_date}")

    for query in JOB_QUERIES:
        for page in range(1, MAX_PAGES + 1):
            try:
                raw_data = fetch_jobs_from_api(query, page)
                records  = len(raw_data.get("data", []))

                if records == 0:
                    logger.info(f"No more results for query='{query}' at page={page}. Stopping.")
                    break

                s3_key = upload_to_s3(s3_client, raw_data, query, page, run_date)
                uploaded_keys.append(s3_key)
                total_records += records

            except requests.exceptions.HTTPError as e:
                logger.error(f"API error | query='{query}' page={page} | {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error | query='{query}' page={page} | {e}")
                raise

    logger.info(f"Ingestion complete | total_records={total_records} | files_uploaded={len(uploaded_keys)}")
    return uploaded_keys


if __name__ == "__main__":
    run_ingestion()
