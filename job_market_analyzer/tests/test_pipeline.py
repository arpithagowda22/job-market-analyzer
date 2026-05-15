"""
test_pipeline.py
----------------
Unit tests for the Job Market Analyzer pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.transformation.transformer import clean_and_standardize, extract_skills, build_gold_insights
from src.summarization.llm_summarizer import build_prompt


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Sample raw Bronze data mimicking API response structure."""
    return pd.DataFrame([
        {
            "job_id":                  "job_001",
            "job_title":               "data engineer",
            "employer_name":           "TechCorp",
            "job_city":                "New York",
            "job_state":               "NY",
            "job_country":             "US",
            "job_employment_type":     "FULLTIME",
            "job_is_remote":           True,
            "job_description":         "Looking for python, sql, airflow, aws expert.",
            "job_min_salary":          120000,
            "job_max_salary":          160000,
            "job_salary_currency":     "USD",
            "job_posted_at_timestamp": 1700000000,
            "job_required_experience": None
        },
        {
            "job_id":                  "job_002",
            "job_title":               "data analyst",
            "employer_name":           "FinanceCo",
            "job_city":                "Chicago",
            "job_state":               "IL",
            "job_country":             "US",
            "job_employment_type":     "FULLTIME",
            "job_is_remote":           False,
            "job_description":         "Must know sql, tableau, power bi.",
            "job_min_salary":          80000,
            "job_max_salary":          110000,
            "job_salary_currency":     "USD",
            "job_posted_at_timestamp": 1700000000,
            "job_required_experience": None
        },
        {
            "job_id":                  "job_001",   # Duplicate — should be dropped
            "job_title":               "data engineer",
            "employer_name":           "TechCorp",
            "job_city":                "New York",
            "job_state":               "NY",
            "job_country":             "US",
            "job_employment_type":     "FULLTIME",
            "job_is_remote":           True,
            "job_description":         "python, sql, airflow",
            "job_min_salary":          120000,
            "job_max_salary":          160000,
            "job_salary_currency":     "USD",
            "job_posted_at_timestamp": 1700000000,
            "job_required_experience": None
        }
    ])


@pytest.fixture
def sample_insights():
    """Sample Gold insights dict for summarization tests."""
    return {
        "run_date":    "2024-01-15",
        "total_jobs":  100,
        "skill_counts": {
            "python":   80,
            "sql":      75,
            "airflow":  60,
            "aws":      55,
            "snowflake": 40
        },
        "salary_by_title": {
            "Data Engineer":  138000,
            "Data Analyst":   95000,
        },
        "remote_stats": {
            "remote_count": 42,
            "onsite_count": 58,
            "remote_pct":   42.0
        },
        "top_companies": {"TechCorp": 5, "FinanceCo": 3}
    }


# ── Transformation Tests ───────────────────────────────────────────

class TestCleanAndStandardize:

    def test_removes_duplicates(self, sample_raw_df):
        """Duplicate job_ids should be removed."""
        result = clean_and_standardize(sample_raw_df)
        assert len(result) == 2, "Should have 2 unique records after dedup"

    def test_title_case_normalization(self, sample_raw_df):
        """Job titles should be title-cased."""
        result = clean_and_standardize(sample_raw_df)
        assert result["job_title"].iloc[0] == "Data Engineer"

    def test_salary_avg_calculated(self, sample_raw_df):
        """salary_avg should be the mean of min and max."""
        result = clean_and_standardize(sample_raw_df)
        assert result["salary_avg"].iloc[0] == 140000.0

    def test_is_remote_is_boolean(self, sample_raw_df):
        """is_remote column should be boolean type."""
        result = clean_and_standardize(sample_raw_df)
        assert result["is_remote"].dtype == bool

    def test_description_lowercased(self, sample_raw_df):
        """Description should be lowercased for skill matching."""
        result = clean_and_standardize(sample_raw_df)
        assert result["description"].iloc[0] == result["description"].iloc[0].lower()


# ── Skill Extraction Tests ─────────────────────────────────────────

class TestExtractSkills:

    def test_extracts_known_skills(self):
        """Should extract skills present in the description."""
        desc   = "we need python, sql, and airflow experience"
        skills = extract_skills(desc)
        assert "python"  in skills
        assert "sql"     in skills
        assert "airflow" in skills

    def test_ignores_unknown_skills(self):
        """Should not extract skills not in the tracked list."""
        desc   = "we need cobol and fortran"
        skills = extract_skills(desc)
        assert skills == []

    def test_empty_description(self):
        """Empty description should return empty list."""
        assert extract_skills("") == []

    def test_no_partial_matches(self):
        """Should not match partial words (e.g. 'awsome' should not match 'aws')."""
        desc   = "awsome candidate needed"
        skills = extract_skills(desc)
        assert "aws" not in skills


# ── Gold Insights Tests ────────────────────────────────────────────

class TestBuildGoldInsights:

    def test_total_jobs_count(self, sample_raw_df):
        """total_jobs should match number of clean records."""
        df       = clean_and_standardize(sample_raw_df)
        insights = build_gold_insights(df)
        assert insights["total_jobs"] == 2

    def test_remote_stats_calculated(self, sample_raw_df):
        """Remote stats should reflect the data."""
        df       = clean_and_standardize(sample_raw_df)
        insights = build_gold_insights(df)
        assert insights["remote_stats"]["remote_count"] == 1
        assert insights["remote_stats"]["remote_pct"] == 50.0

    def test_skill_counts_not_empty(self, sample_raw_df):
        """Skill counts should have entries for jobs with descriptions."""
        df       = clean_and_standardize(sample_raw_df)
        insights = build_gold_insights(df)
        assert len(insights["skill_counts"]) > 0


# ── Summarization Tests ────────────────────────────────────────────

class TestBuildPrompt:

    def test_prompt_contains_run_date(self, sample_insights):
        """Prompt should include the run date."""
        prompt = build_prompt(sample_insights)
        assert "2024-01-15" in prompt

    def test_prompt_contains_total_jobs(self, sample_insights):
        """Prompt should mention total jobs count."""
        prompt = build_prompt(sample_insights)
        assert "100" in prompt

    def test_prompt_contains_skills(self, sample_insights):
        """Prompt should list skills."""
        prompt = build_prompt(sample_insights)
        assert "python" in prompt.lower()

    def test_prompt_is_non_empty(self, sample_insights):
        """Prompt should never be empty."""
        prompt = build_prompt(sample_insights)
        assert len(prompt) > 100
