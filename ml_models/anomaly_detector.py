"""
Anomaly / Fraud Detection using Isolation Forest
Detects suspicious transactions based on behavioral patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
from loguru import logger


class AnomalyDetector:
    """Isolation Forest for transaction anomaly detection."""

    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def prepare_features(self, df):
        """Select numerical features for anomaly detection."""
        feature_cols = [
            "unit_price", "quantity", "discount_pct",
            "total_amount", "shipping_cost",
        ]
        X = df[feature_cols].copy()
        return X, feature_cols

    def detect(self, df):
        """Run anomaly detection."""
        X, cols = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)

        logger.info("Running Isolation Forest anomaly detection...")
        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        df = df.copy()
        df["anomaly_score"] = scores
        df["is_anomaly"] = (predictions == -1).astype(int)

        n_anomalies = df["is_anomaly"].sum()
        anomaly_revenue = df[df["is_anomaly"] == 1]["total_amount"].sum()

        print("\n" + "=" * 55)
        print("  ANOMALY DETECTION RESULTS")
        print("=" * 55)
        print(f"  Anomalies Found   : {n_anomalies:,} ({n_anomalies/len(df):.1%})")
        print(f"  Revenue Flagged   : ${anomaly_revenue:,.2f}")
        print(f"  Avg Anomaly Score : {df[df['is_anomaly']==1]['anomaly_score'].mean():.4f}")
        print("\n  Top Flagged Characteristics:")
        print(df[df["is_anomaly"] == 1][
            ["unit_price", "quantity", "discount_pct", "total_amount"]
        ].describe().round(2).to_string())

        return df

    def visualize(self, df, save_dir="visuals"):
        """Plot anomaly distribution."""
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Price vs Total Amount
        normal = df[df["is_anomaly"] == 0]
        anomaly = df[df["is_anomaly"] == 1]

        axes[0].scatter(normal["unit_price"], normal["total_amount"],
                        alpha=0.3, s=10, color="#2E86AB", label="Normal")
        axes[0].scatter(anomaly["unit_price"], anomaly["total_amount"],
                        alpha=0.8, s=30, color="#C73E1D", label="Anomaly", marker="x")
        axes[0].set_xlabel("Unit Price")
        axes[0].set_ylabel("Total Amount")
        axes[0].set_title("Anomaly: Price vs Total")
        axes[0].legend()

        # Score Distribution
        axes[1].hist(df[df["is_anomaly"] == 0]["anomaly_score"],
                     bins=50, color="#2E86AB", alpha=0.7, label="Normal")
        axes[1].hist(df[df["is_anomaly"] == 1]["anomaly_score"],
                     bins=20, color="#C73E1D", alpha=0.8, label="Anomaly")
        axes[1].set_xlabel("Anomaly Score")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Anomaly Score Distribution")
        axes[1].legend()

        # Anomalies by Category
        anom_cat = df.groupby("category")["is_anomaly"].sum().sort_values()
        axes[2].barh(anom_cat.index, anom_cat.values, color="#F18F01")
        axes[2].set_xlabel("Anomaly Count")
        axes[2].set_title("Anomalies by Category")

        plt.suptitle("Transaction Anomaly Detection (Isolation Forest)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/anomaly_detection.png", dpi=150, bbox_inches="tight")
        plt.show()
        logger.info(f"Saved → {save_dir}/anomaly_detection.png")


if __name__ == "__main__":
    df = pd.read_csv("data/ecommerce_clean.csv")
    detector = AnomalyDetector(contamination=0.05)
    df_flagged = detector.detect(df)
    detector.visualize(df_flagged)
    df_flagged[df_flagged["is_anomaly"] == 1].to_csv(
        "data/flagged_transactions.csv", index=False
    )
    logger.info("Saved → data/flagged_transactions.csv")