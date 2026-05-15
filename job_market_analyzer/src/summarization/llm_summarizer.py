"""
llm_summarizer.py
-----------------
Reads Gold layer insights from S3 and sends them to an LLM (OpenAI GPT)
to generate a human-readable job market summary report.

The report is saved back to S3 as a text file.
"""

import json
import logging
import boto3
from datetime import datetime
from openai import OpenAI
from config.config import (
    S3_BUCKET, S3_GOLD_PREFIX, S3_REPORT_PREFIX,
    OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def load_gold_insights(s3_client, run_date: str) -> dict:
    """
    Load Gold insights JSON from S3.

    Args:
        s3_client: Boto3 S3 client
        run_date:  Pipeline run date (YYYY-MM-DD)

    Returns:
        Insights dictionary
    """
    s3_key   = f"{S3_GOLD_PREFIX}/{run_date}/insights.json"
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
    insights = json.loads(response["Body"].read())
    logger.info(f"Loaded Gold insights | run_date={run_date}")
    return insights


def build_prompt(insights: dict) -> str:
    """
    Build a structured prompt from Gold insights for the LLM.

    Args:
        insights: Gold layer insights dict

    Returns:
        Formatted prompt string
    """
    skill_list   = "\n".join([f"  - {k}: {v} mentions" for k, v in list(insights["skill_counts"].items())[:10]])
    salary_list  = "\n".join([f"  - {k}: ${v:,.0f}/yr" for k, v in list(insights["salary_by_title"].items())[:5]])
    remote_stats = insights["remote_stats"]

    prompt = f"""
You are a data analyst specializing in labor market trends.
Based on the following job market data, write a concise, professional market summary report.

--- DATA ---
Date: {insights['run_date']}
Total Jobs Analyzed: {insights['total_jobs']}

Top In-Demand Skills:
{skill_list}

Average Salary by Role:
{salary_list}

Remote Work Stats:
  - Remote Jobs: {remote_stats['remote_count']} ({remote_stats['remote_pct']}%)
  - On-site Jobs: {remote_stats['onsite_count']}

--- INSTRUCTIONS ---
Write a 3-4 paragraph market summary with:
1. Key skill trends and what they signal for the market
2. Salary insights and what roles pay the most
3. Remote work trends
4. 3 actionable recommendations for job seekers in data engineering

Keep it professional, data-driven, and concise.
"""
    return prompt.strip()


def generate_summary(prompt: str) -> str:
    """
    Send prompt to OpenAI and return the generated summary.

    Args:
        prompt: Formatted market insights prompt

    Returns:
        LLM-generated summary text
    """
    logger.info("Sending insights to LLM for summarization...")

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional data analyst writing market reports."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.4   # Low temp = factual, consistent output
    )

    summary = response.choices[0].message.content.strip()
    logger.info("LLM summary generated successfully")
    return summary


def upload_report(s3_client, summary: str, run_date: str) -> str:
    """
    Upload the LLM-generated summary report to S3.

    Args:
        s3_client: Boto3 S3 client
        summary:   Generated report text
        run_date:  Pipeline run date

    Returns:
        S3 key of the uploaded report
    """
    s3_key = f"{S3_REPORT_PREFIX}/{run_date}/market_summary.txt"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=summary.encode("utf-8"),
        ContentType="text/plain"
    )
    logger.info(f"Report uploaded | s3://{S3_BUCKET}/{s3_key}")
    return s3_key


def run_summarization(run_date: str = None) -> str:
    """
    Main function — load Gold insights, generate LLM summary, upload report.

    Args:
        run_date: Pipeline run date (YYYY-MM-DD). Defaults to today.

    Returns:
        Generated summary text
    """
    run_date  = run_date or datetime.today().strftime("%Y-%m-%d")
    s3_client = get_s3_client()

    insights = load_gold_insights(s3_client, run_date)
    prompt   = build_prompt(insights)
    summary  = generate_summary(prompt)
    upload_report(s3_client, summary, run_date)

    # Print report to console
    print("\n" + "="*60)
    print(f"JOB MARKET SUMMARY — {run_date}")
    print("="*60)
    print(summary)
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    run_summarization()
