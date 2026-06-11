"""
Master Pipeline Runner
Runs the entire project end-to-end in one command:
python scripts/run_pipeline.py
"""

import sys
import os
import time
from loguru import logger

sys.path.append(os.path.abspath("."))

from data.generate_data import generate_ecommerce_data
from scripts.data_cleaning import load_raw_data, clean_data, save_clean_data
from ml_models.churn_predictor import ChurnPredictor
from ml_models.sales_forecaster import SalesForecaster
from ml_models.smart_segmentation import MLCustomerSegmentation
from ml_models.anomaly_detector import AnomalyDetector
from nlp.sentiment_analyzer import SentimentAnalyzer
from ai_insights.llm_summarizer import AIInsightGenerator
import pandas as pd


def run_full_pipeline():
    start = time.time()
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("\n" + "🚀 " + "=" * 60)
    print("   E-COMMERCE AI ANALYTICS — FULL PIPELINE")
    print("=" * 63)

    # ── 1. Generate Data ────────────────────────────────
    print("\n[1/7] Generating synthetic data...")
    df_raw = generate_ecommerce_data(n_records=10000)
    df_raw.to_csv("data/ecommerce_raw.csv", index=False)

    # ── 2. Clean Data ───────────────────────────────────
    print("\n[2/7] Cleaning and preprocessing...")
    df = clean_data(df_raw)
    save_clean_data(df)

    # ── 3. Churn Prediction ─────────────────────────────
    print("\n[3/7] Training churn prediction model...")
    predictor = ChurnPredictor()
    results = predictor.train(df)
    predictor.plot_results(results)
    predictor.save()
    risk_df = predictor.predict_risk(df)
    risk_df.to_csv("data/churn_predictions.csv", index=False)

    # ── 4. Sales Forecasting ────────────────────────────
    print("\n[4/7] Generating sales forecast...")
    forecaster = SalesForecaster(periods=30)
    forecast = forecaster.train(df)
    forecast_summary = forecaster.get_summary()
    forecaster.plot()

    # ── 5. Customer Segmentation ────────────────────────
    print("\n[5/7] Running ML customer segmentation...")
    segmenter = MLCustomerSegmentation(n_clusters=5)
    segments = segmenter.fit(df)
    segmenter.visualize()
    segments.to_csv("data/customer_segments.csv", index=False)

    # ── 6. NLP Sentiment Analysis ───────────────────────
    print("\n[6/7] Running NLP sentiment analysis...")
    analyzer = SentimentAnalyzer(use_bert=False)
    df_analyzed = analyzer.analyze_all(df)
    analyzer.visualize(df_analyzed)
    df_analyzed.to_csv("data/sentiment_results.csv", index=False)

    # ── 7. Anomaly Detection ────────────────────────────
    print("\n[7/7] Running anomaly detection...")
    detector = AnomalyDetector()
    df_flagged = detector.detect(df)
    detector.visualize(df_flagged)
    df_flagged[df_flagged["is_anomaly"] == 1].to_csv(
        "data/flagged_transactions.csv", index=False
    )

    # ── Generate AI Report ──────────────────────────────
    print("\n✨ Generating AI executive report...")
    completed = df[df["order_status"].isin(["Delivered", "Shipped", "Processing"])]
    high_risk = risk_df[risk_df["risk_level"] == "High Risk"]

    kpis = {
        "revenue": f"${completed['total_amount'].sum():,.0f}",
        "orders": f"{len(completed):,}",
        "customers": f"{completed['customer_id'].nunique():,}",
        "aov": f"${completed['total_amount'].mean():.2f}",
        "cancellation_rate": f"{(df['order_status']=='Cancelled').mean()*100:.1f}%",
        "return_rate": f"{(df['order_status']=='Returned').mean()*100:.1f}%",
        "avg_rating": f"{completed['customer_rating'].mean():.2f}/5",
    }
    churn_stats = {
        "high_risk": len(high_risk),
        "revenue_at_risk": high_risk["monetary"].sum(),
        "avg_recency": int(high_risk["recency"].mean()),
        "auc": round(results["auc_roc"], 3),
    }
    sentiment_stats = {
        "positive_pct": (df_analyzed["sentiment_label"] == "POSITIVE").mean() * 100,
        "top_complaints": "quality, shipping, packaging",
    }

    ai = AIInsightGenerator()
    report = ai.generate_full_report(kpis, churn_stats, forecast_summary, sentiment_stats)
    with open("reports/ai_executive_report.md", "w") as f:
        f.write(report)

    elapsed = time.time() - start
    print(f"\n{'=' * 63}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  📁 Outputs:")
    print(f"     data/ecommerce_clean.csv")
    print(f"     data/churn_predictions.csv")
    print(f"     data/customer_segments.csv")
    print(f"     data/sentiment_results.csv")
    print(f"     data/flagged_transactions.csv")
    print(f"     reports/ai_executive_report.md")
    print(f"     visuals/ (all charts)")
    print(f"{'=' * 63}\n")


if __name__ == "__main__":
    run_full_pipeline()