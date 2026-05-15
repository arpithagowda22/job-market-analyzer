import os

# ── AWS Configuration ──────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET             = os.getenv("AWS_S3_BUCKET", "job-market-analyzer")

# S3 paths following medallion architecture
S3_BRONZE_PREFIX = "bronze/jobs"
S3_SILVER_PREFIX = "silver/jobs"
S3_GOLD_PREFIX   = "gold/insights"
S3_REPORT_PREFIX = "gold/reports"

# ── API Configuration ──────────────────────────────────────────────
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"

# Job search parameters
JOB_QUERIES   = ["Data Engineer", "Data Analyst", "Analytics Engineer", "ETL Developer"]
JOBS_PER_PAGE = 10
MAX_PAGES     = 5   # 50 records per query, ~200 total per run

# ── LLM Configuration ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-3.5-turbo"
MAX_TOKENS     = 1000

# ── Pipeline Configuration ─────────────────────────────────────────
PIPELINE_RETRIES    = 3
RETRY_DELAY_SECONDS = 30
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
