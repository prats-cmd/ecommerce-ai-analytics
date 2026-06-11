"""
NLP Sentiment Analysis Pipeline
Uses VADER (fast) or BERT (accurate) to analyze customer reviews
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from loguru import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER not installed. Run: pip install vaderSentiment")

try:
    from transformers import pipeline as hf_pipeline
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


class SentimentAnalyzer:
    """Customer review NLP pipeline."""

    def __init__(self, use_bert=False):
        self.use_bert = use_bert and BERT_AVAILABLE
        if self.use_bert:
            logger.info("Loading BERT model (this may take a moment)...")
            self.bert = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        elif VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            logger.warning("No NLP library available. Using rule-based fallback.")

    def _analyze_one(self, text):
        """Analyze a single review."""
        text = str(text)[:512]

        if self.use_bert and BERT_AVAILABLE:
            result = self.bert(text)[0]
            score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
            label = result["label"]
        elif VADER_AVAILABLE:
            score = self.vader.polarity_scores(text)["compound"]
            if score >= 0.05:
                label = "POSITIVE"
            elif score <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
        else:
            # Simple keyword fallback
            text_lower = text.lower()
            positive_words = ["great", "love", "amazing", "perfect", "excellent",
                              "fantastic", "best", "super", "wonderful", "happy"]
            negative_words = ["terrible", "awful", "worst", "broke", "damaged",
                              "disappoint", "waste", "return", "broken", "bad"]
            pos = sum(1 for w in positive_words if w in text_lower)
            neg = sum(1 for w in negative_words if w in text_lower)
            score = (pos - neg) / max(pos + neg, 1)
            label = "POSITIVE" if score > 0 else ("NEGATIVE" if score < 0 else "NEUTRAL")

        return score, label

    def analyze_all(self, df, text_col="customer_review"):
        """Batch analyze all reviews."""
        logger.info(f"Analyzing {len(df):,} reviews...")
        scores, labels = [], []

        for text in df[text_col]:
            s, l = self._analyze_one(text)
            scores.append(s)
            labels.append(l)

        df = df.copy()
        df["sentiment_score"] = scores
        df["sentiment_label"] = labels

        print("\n" + "=" * 55)
        print("  SENTIMENT ANALYSIS RESULTS")
        print("=" * 55)
        vc = pd.Series(labels).value_counts()
        for label, count in vc.items():
            print(f"  {label:<12}: {count:,} ({count/len(labels):.1%})")
        print(f"\n  Avg Score : {np.mean(scores):.4f}")

        return df

    def product_sentiment_report(self, df):
        """Aggregate sentiment by product."""
        return (
            df.groupby("product_name")
            .agg(
                avg_sentiment=("sentiment_score", "mean"),
                review_count=("sentiment_score", "count"),
                positive_pct=("sentiment_label", lambda x: (x == "POSITIVE").mean() * 100),
                negative_pct=("sentiment_label", lambda x: (x == "NEGATIVE").mean() * 100),
                avg_rating=("customer_rating", "mean"),
            )
            .round(2)
            .sort_values("avg_sentiment", ascending=False)
            .reset_index()
        )

    def extract_keywords(self, df, sentiment="NEGATIVE", top_n=15):
        """Extract most common keywords from positive/negative reviews."""
        texts = df[df["sentiment_label"] == sentiment]["customer_review"]
        words = re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(texts).lower())
        stopwords = {
            "this", "that", "with", "from", "have", "just", "very",
            "your", "they", "them", "their", "been", "will", "also",
            "product", "bought", "would", "after", "when", "good",
        }
        words = [w for w in words if w not in stopwords]
        return Counter(words).most_common(top_n)

    def visualize(self, df, save_dir="visuals"):
        """Comprehensive sentiment visualization."""
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {"POSITIVE": "#44BBA4", "NEUTRAL": "#F18F01", "NEGATIVE": "#C73E1D"}

        # Sentiment distribution
        vc = df["sentiment_label"].value_counts()
        axes[0, 0].pie(vc, labels=vc.index, autopct="%1.1f%%",
                       colors=[colors.get(l, "gray") for l in vc.index])
        axes[0, 0].set_title("Sentiment Distribution")

        # Sentiment by category
        cat_sent = df.groupby("category")["sentiment_score"].mean().sort_values()
        bar_colors = ["#C73E1D" if v < 0 else "#44BBA4" for v in cat_sent.values]
        axes[0, 1].barh(cat_sent.index, cat_sent.values, color=bar_colors)
        axes[0, 1].axvline(0, color="black", lw=0.8)
        axes[0, 1].set_title("Avg Sentiment by Category")
        axes[0, 1].set_xlabel("Sentiment Score (-1 to +1)")

        # Score histogram
        for lbl, grp in df.groupby("sentiment_label"):
            axes[1, 0].hist(grp["sentiment_score"], bins=30, alpha=0.6,
                            label=lbl, color=colors.get(lbl, "gray"))
        axes[1, 0].set_title("Sentiment Score Distribution")
        axes[1, 0].set_xlabel("Score")
        axes[1, 0].legend()

        # Rating vs Sentiment
        axes[1, 1].boxplot(
            [df[df["customer_rating"] == r]["sentiment_score"].values for r in range(1, 6)],
            labels=["1★", "2★", "3★", "4★", "5★"]
        )
        axes[1, 1].set_title("Sentiment Score vs Customer Rating")
        axes[1, 1].set_xlabel("Rating")
        axes[1, 1].set_ylabel("Sentiment Score")

        plt.suptitle("NLP Sentiment Analysis Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sentiment_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()
        logger.info(f"Saved → {save_dir}/sentiment_analysis.png")


if __name__ == "__main__":
    df = pd.read_csv("data/ecommerce_clean.csv")
    analyzer = SentimentAnalyzer(use_bert=False)
    df_analyzed = analyzer.analyze_all(df)
    product_report = analyzer.product_sentiment_report(df_analyzed)

    print("\nTop 5 Best Reviewed Products:")
    print(product_report.head().to_string())
    print("\nTop 5 Worst Reviewed:")
    print(product_report.tail().to_string())

    print("\nTop Negative Keywords:")
    for word, count in analyzer.extract_keywords(df_analyzed, "NEGATIVE"):
        print(f"  {word:<20} {count}")

    analyzer.visualize(df_analyzed)
    df_analyzed.to_csv("data/sentiment_results.csv", index=False)