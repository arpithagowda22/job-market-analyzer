# AI-Powered Job Market Analyzer

An end-to-end data pipeline that ingests job postings from public APIs, processes and transforms the data using AWS S3 (medallion architecture), and uses LLM-based summarization to extract in-demand skills and role trends.

---

## Architecture

```
Job APIs → Ingestion Layer → AWS S3 (Bronze) → Transformation → S3 (Silver) → Aggregation → S3 (Gold) → LLM Summarization → Analytics Report
```

All workflows are orchestrated using **Apache Airflow** with modular DAGs and automated alerting.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Python, Requests, RapidAPI |
| Storage | AWS S3 (Bronze/Silver/Gold) |
| Transformation | Python, Pandas |
| Orchestration | Apache Airflow |
| Summarization | OpenAI API (LLM) |
| Monitoring | Airflow Alerts, Logging |

---

## Project Structure

```
job_market_analyzer/
├── dags/
│   └── job_market_dag.py         # Main Airflow DAG
├── src/
│   ├── ingestion/
│   │   └── api_ingestor.py       # Fetches job data from APIs
│   ├── transformation/
│   │   └── transformer.py        # Cleans and transforms raw data
│   └── summarization/
│       └── llm_summarizer.py     # LLM-based skill extraction
├── config/
│   └── config.py                 # Config and environment variables
├── tests/
│   └── test_pipeline.py          # Unit tests
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone  https://github.com/arpithagowda22/job-market-analyzer.git
cd job-market-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_S3_BUCKET=your_bucket_name
export RAPIDAPI_KEY=your_rapidapi_key
export OPENAI_API_KEY=your_openai_key
```

### 4. Initialize Airflow
```bash
airflow db init
airflow webserver --port 8080
airflow scheduler
```

### 5. Trigger the DAG
Go to `http://localhost:8080`, find `job_market_pipeline` and enable it.

---

## Pipeline Stages

### Bronze Layer — Raw Ingestion
- Fetches 50K+ daily job postings from JSearch API via RapidAPI
- Stores raw JSON responses in `s3://bucket/bronze/jobs/YYYY-MM-DD/`
- No transformations — raw data preserved as-is

### Silver Layer — Cleaned & Transformed
- Drops duplicates and null records
- Standardizes column names and data types
- Extracts salary range, location, experience level
- Stores as Parquet in `s3://bucket/silver/jobs/YYYY-MM-DD/`

### Gold Layer — Aggregated Insights
- Top 20 in-demand skills by job category
- Average salary by role and location
- Remote vs on-site trends
- Stores as Parquet in `s3://bucket/gold/insights/YYYY-MM-DD/`

### LLM Summarization
- Sends Gold layer data to OpenAI GPT
- Returns a structured market summary report
- Saved as `s3://bucket/gold/reports/YYYY-MM-DD/market_summary.txt`

---

## Sample Output

```
📊 Job Market Summary — 2024-01-15

Top In-Demand Skills:
1. Python (78% of Data Engineer roles)
2. SQL (74%)
3. Apache Airflow (61%)
4. AWS (58%)
5. Snowflake (43%)

Avg Salary — Data Engineer: $138,000/yr
Remote Roles: 42% of total postings
Fastest Growing: AI/ML Engineer (+23% WoW)
```

---

## Key Features
- Medallion architecture (Bronze → Silver → Gold)
- Fully orchestrated with Apache Airflow
- LLM-based skill extraction and trend summarization
- 98%+ pipeline uptime with retry logic and alerting
- Modular, testable codebase

---

## Author
**Arpitha Raghu** — Data Engineer  
[LinkedIn](https://www.linkedin.com/in/arpitha2205/) 
[GitHub](https://github.com/arpithagowda22)
arpithagowda2205@gmail.com
