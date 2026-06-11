"""
Customer Churn Prediction Model
Algorithm : Random Forest Classifier
Target    : Will a customer churn? (no purchase in 90+ days)
Metrics   : AUC-ROC, Precision, Recall, F1
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from loguru import logger


class ChurnPredictor:
    """End-to-end churn prediction pipeline."""

    CHURN_DAYS = 90

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    # ── Feature Engineering ─────────────────────────────────
    def engineer_features(self, df):
        ref_date = pd.to_datetime(df["order_date"]).max() + timedelta(days=1)

        features = (
            df.groupby("customer_id")
            .agg(
                recency=("order_date", lambda x: (ref_date - pd.to_datetime(x).max()).days),
                frequency=("order_id", "nunique"),
                monetary=("total_amount", "sum"),
                avg_order_value=("total_amount", "mean"),
                order_std=("total_amount", "std"),
                total_items=("quantity", "sum"),
                unique_categories=("category", "nunique"),
                avg_rating=("customer_rating", "mean"),
                min_rating=("customer_rating", "min"),
                return_rate=("order_status", lambda x: (x == "Returned").mean()),
                cancel_rate=("order_status", lambda x: (x == "Cancelled").mean()),
                discount_sensitivity=("discount_pct", "mean"),
                weekend_ratio=("is_weekend", "mean"),
                holiday_ratio=("is_holiday_season", "mean"),
                express_pct=("shipping_type", lambda x: (x == "Express").mean()),
                max_order=("total_amount", "max"),
                days_active=("order_date", lambda x: (
                    pd.to_datetime(x).max() - pd.to_datetime(x).min()
                ).days + 1),
            )
            .reset_index()
        )

        features["order_std"] = features["order_std"].fillna(0)
        features["orders_per_active_day"] = (
            features["frequency"] / features["days_active"].clip(1)
        )
        features["is_churned"] = (features["recency"] > self.CHURN_DAYS).astype(int)

        return features

    # ── Training ────────────────────────────────────────────
    def train(self, df):
        logger.info("Engineering features for churn model...")
        feats = self.engineer_features(df)

        drop_cols = ["customer_id", "is_churned"]
        X = feats.drop(columns=drop_cols)
        y = feats["is_churned"]

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        logger.info("Training Random Forest Classifier...")
        self.model.fit(X_train_s, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_s)
        y_prob = self.model.predict_proba(X_test_s)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        cv_scores = cross_val_score(
            self.model, self.scaler.transform(X), y, cv=5, scoring="roc_auc"
        )

        print("\n" + "=" * 60)
        print("  CHURN PREDICTION MODEL — RESULTS")
        print("=" * 60)
        print(f"  AUC-ROC         : {auc:.4f}")
        print(f"  CV AUC (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n  Churn Rate      : {y.mean():.1%}")
        print(f"  Train samples   : {len(X_train):,}")
        print(f"  Test samples    : {len(X_test):,}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return {
            "auc_roc": auc,
            "cv_scores": cv_scores,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "importance_df": importance_df,
            "features": feats,
        }

    # ── Visualization ────────────────────────────────────────
    def plot_results(self, results, save_dir="visuals"):
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(results["y_test"], results["y_prob"])
        axes[0].plot(fpr, tpr, color="#2E86AB", lw=2,
                     label=f"AUC = {results['auc_roc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve — Churn Prediction")
        axes[0].legend()

        # Confusion Matrix
        cm = confusion_matrix(results["y_test"], results["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=["Active", "Churned"],
                    yticklabels=["Active", "Churned"])
        axes[1].set_title("Confusion Matrix")
        axes[1].set_ylabel("Actual")
        axes[1].set_xlabel("Predicted")

        # Feature Importance
        top10 = results["importance_df"].head(10)
        axes[2].barh(top10["feature"], top10["importance"], color="#A23B72")
        axes[2].set_title("Top 10 Feature Importances")
        axes[2].set_xlabel("Importance")
        axes[2].invert_yaxis()

        plt.suptitle("Churn Prediction Model Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/churn_model_results.png", dpi=150, bbox_inches="tight")
        plt.show()
        logger.info(f"Saved → {save_dir}/churn_model_results.png")

    # ── Predict ──────────────────────────────────────────────
    def predict_risk(self, df):
        assert self.is_trained, "Train the model first!"
        feats = self.engineer_features(df)
        X = feats[self.feature_names]
        X_s = self.scaler.transform(X)
        feats["churn_probability"] = self.model.predict_proba(X_s)[:, 1]
        feats["risk_level"] = pd.cut(
            feats["churn_probability"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
        return feats[["customer_id", "churn_probability", "risk_level",
                       "recency", "frequency", "monetary"]].sort_values(
            "churn_probability", ascending=False
        )

    def save(self, path="ml_models/churn_model.pkl"):
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "features": self.feature_names}, path)
        logger.info(f"Model saved → {path}")


if __name__ == "__main__":
    df = pd.read_csv("data/ecommerce_clean.csv")
    predictor = ChurnPredictor()
    results = predictor.train(df)
    predictor.plot_results(results)
    predictor.save()
    risk_df = predictor.predict_risk(df)
    print(f"\n  High Risk Customers : {(risk_df['risk_level'] == 'High Risk').sum():,}")
    risk_df.to_csv("data/churn_predictions.csv", index=False)