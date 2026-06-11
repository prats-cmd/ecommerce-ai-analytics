"""
ML Customer Segmentation using K-Means + PCA
Goes beyond RFM with 10+ behavioral features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
from loguru import logger


class MLCustomerSegmentation:
    """K-Means clustering with PCA visualization."""

    SEGMENT_NAMES = {
        0: "VIP Loyalists",
        1: "Deal Seekers",
        2: "Dormant High-Value",
        3: "One-Time Buyers",
        4: "Occasional Shoppers",
    }

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.features_ = None

    def create_features(self, df):
        """Rich behavioral feature set."""
        ref = pd.to_datetime(df["order_date"]).max()
        feats = (
            df.groupby("customer_id")
            .agg(
                recency=("order_date", lambda x: (ref - pd.to_datetime(x).max()).days),
                frequency=("order_id", "nunique"),
                monetary=("total_amount", "sum"),
                avg_order_value=("total_amount", "mean"),
                order_variability=("total_amount", "std"),
                category_diversity=("category", "nunique"),
                discount_sensitivity=("discount_pct", "mean"),
                weekend_shopper=("is_weekend", "mean"),
                return_rate=("order_status", lambda x: (x == "Returned").mean()),
                express_ratio=("shipping_type", lambda x: (x == "Express").mean()),
                avg_rating=("customer_rating", "mean"),
            )
            .fillna(0)
            .reset_index()
        )
        return feats

    def find_optimal_k(self, X_scaled, k_range=range(2, 10)):
        """Elbow method + Silhouette to find optimal clusters."""
        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, km.labels_))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(list(k_range), inertias, "o-", color="#2E86AB")
        axes[0].set_title("Elbow Method")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia")

        axes[1].plot(list(k_range), silhouettes, "o-", color="#A23B72")
        axes[1].set_title("Silhouette Scores")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")

        plt.tight_layout()
        plt.savefig("visuals/optimal_k.png", dpi=150, bbox_inches="tight")
        plt.show()
        return inertias, silhouettes

    def fit(self, df):
        """Fit segmentation model."""
        feats = self.create_features(df)
        feat_cols = [c for c in feats.columns if c != "customer_id"]

        X = feats[feat_cols].values
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal k
        self.find_optimal_k(X_scaled)

        # Fit K-Means
        feats["cluster"] = self.kmeans.fit_predict(X_scaled)

        # Silhouette score
        sil = silhouette_score(X_scaled, feats["cluster"])
        logger.info(f"Silhouette Score: {sil:.4f}")

        # PCA for 2D visualization
        X_pca = self.pca.fit_transform(X_scaled)
        feats["pca_x"] = X_pca[:, 0]
        feats["pca_y"] = X_pca[:, 1]

        # Auto-name clusters
        profiles = feats.groupby("cluster")[feat_cols].mean()
        names = self._auto_name_clusters(profiles)
        feats["segment_name"] = feats["cluster"].map(names)

        self.features_ = feats
        self.feat_cols = feat_cols

        print(f"\n  Silhouette Score : {sil:.4f}")
        print(f"  PCA Variance Explained : "
              f"{self.pca.explained_variance_ratio_.sum():.1%}\n")
        print(feats["segment_name"].value_counts().to_string())
        return feats

    def _auto_name_clusters(self, profiles):
        """Assign names based on cluster characteristics."""
        names = {}
        for idx in profiles.index:
            row = profiles.loc[idx]
            high_monetary = row["monetary"] > profiles["monetary"].quantile(0.75)
            high_freq = row["frequency"] > profiles["frequency"].quantile(0.75)
            high_recency = row["recency"] > profiles["recency"].quantile(0.75)
            high_discount = row["discount_sensitivity"] > profiles["discount_sensitivity"].quantile(0.75)

            if high_monetary and high_freq:
                names[idx] = "VIP Loyalists"
            elif high_monetary and high_recency:
                names[idx] = "Dormant High-Value"
            elif high_discount:
                names[idx] = "Deal Seekers"
            elif row["frequency"] <= 1:
                names[idx] = "One-Time Buyers"
            else:
                names[idx] = "Occasional Shoppers"
        return names

    def visualize(self, save_dir="visuals"):
        """PCA scatter + segment profiles."""
        os.makedirs(save_dir, exist_ok=True)
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#44BBA4"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PCA Scatter
        for i, (name, grp) in enumerate(self.features_.groupby("segment_name")):
            axes[0].scatter(
                grp["pca_x"], grp["pca_y"],
                label=name, alpha=0.6, s=60,
                color=colors[i % len(colors)]
            )
        axes[0].set_title(
            f"Customer Segments — PCA\n"
            f"(Variance Explained: {self.pca.explained_variance_ratio_.sum():.1%})"
        )
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")
        axes[0].legend(fontsize=8)

        # Segment Revenue
        seg_rev = self.features_.groupby("segment_name")["monetary"].sum().sort_values()
        axes[1].barh(seg_rev.index, seg_rev.values, color=colors[:len(seg_rev)])
        axes[1].set_title("Total Revenue by Segment")
        axes[1].set_xlabel("Revenue ($)")

        plt.suptitle("ML Customer Segmentation (K-Means + PCA)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ml_segmentation.png", dpi=150, bbox_inches="tight")
        plt.show()
        logger.info(f"Saved → {save_dir}/ml_segmentation.png")

    def get_profiles(self):
        """Return cluster profile summary."""
        return self.features_.groupby("segment_name")[self.feat_cols].mean().round(2)


if __name__ == "__main__":
    df = pd.read_csv("data/ecommerce_clean.csv")
    seg = MLCustomerSegmentation(n_clusters=5)
    segments = seg.fit(df)
    seg.visualize()
    print("\nCluster Profiles:")
    print(seg.get_profiles().T.to_string())
    segments.to_csv("data/customer_segments.csv", index=False)