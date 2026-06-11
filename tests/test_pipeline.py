"""
Unit Tests for the Analytics Pipeline
Run: pytest tests/ -v --cov
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("."))

from data.generate_data import generate_ecommerce_data
from scripts.data_cleaning import clean_data


# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw_df():
    return generate_ecommerce_data(n_records=500, seed=99)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


# ── Data Generation Tests ─────────────────────────────────────
class TestDataGeneration:
    def test_shape(self, raw_df):
        assert raw_df.shape[0] >= 500, "Should have at least 500 rows"

    def test_required_columns(self, raw_df):
        required = [
            "order_id", "customer_id", "order_date", "category",
            "total_amount", "customer_rating", "order_status",
        ]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_has_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() > 0, "Should have some missing values"

    def test_has_duplicates(self, raw_df):
        assert raw_df.duplicated(subset="order_id").sum() > 0, "Should have duplicates"

    def test_reviews_generated(self, raw_df):
        assert "customer_review" in raw_df.columns
        assert raw_df["customer_review"].notnull().all()


# ── Data Cleaning Tests ───────────────────────────────────────
class TestDataCleaning:
    def test_no_missing_values(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_no_duplicates(self, clean_df):
        assert clean_df.duplicated(subset="order_id").sum() == 0

    def test_no_negative_prices(self, clean_df):
        assert (clean_df["unit_price"] < 0).sum() == 0

    def test_new_features_exist(self, clean_df):
        expected = [
            "order_year", "order_month", "order_quarter",
            "is_weekend", "age_group", "is_high_value",
        ]
        for col in expected:
            assert col in clean_df.columns, f"Missing feature: {col}"

    def test_age_range(self, clean_df):
        assert clean_df["customer_age"].min() >= 18
        assert clean_df["customer_age"].max() <= 80

    def test_total_amount_positive(self, clean_df):
        assert (clean_df["total_amount"] >= 0).all()

    def test_order_dates_valid(self, clean_df):
        assert pd.to_datetime(clean_df["order_date"]).notnull().all()


# ── ML Tests ─────────────────────────────────────────────────
class TestMLModels:
    def test_churn_features(self, clean_df):
        from ml_models.churn_predictor import ChurnPredictor
        predictor = ChurnPredictor()
        feats = predictor.engineer_features(clean_df)
        assert "is_churned" in feats.columns
        assert feats["is_churned"].isin([0, 1]).all()

    def test_segmentation_clusters(self, clean_df):
        from ml_models.smart_segmentation import MLCustomerSegmentation
        seg = MLCustomerSegmentation(n_clusters=3)
        result = seg.fit(clean_df)
        assert "cluster" in result.columns
        assert result["cluster"].nunique() == 3

    def test_anomaly_detector(self, clean_df):
        from ml_models.anomaly_detector import AnomalyDetector
        det = AnomalyDetector(contamination=0.05)
        flagged = det.detect(clean_df)
        assert "is_anomaly" in flagged.columns
        assert flagged["is_anomaly"].isin([0, 1]).all()


# ── NLP Tests ─────────────────────────────────────────────────
class TestNLP:
    def test_sentiment_analysis(self, clean_df):
        from nlp.sentiment_analyzer import SentimentAnalyzer
        nlp = SentimentAnalyzer(use_bert=False)
        result = nlp.analyze_all(clean_df.head(50))
        assert "sentiment_label" in result.columns
        assert result["sentiment_label"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"]).all()