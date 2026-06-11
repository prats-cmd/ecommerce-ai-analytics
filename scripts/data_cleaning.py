"""
Data Cleaning & Preprocessing Pipeline
Handles: missing values, duplicates, outliers,
type conversion, standardization, feature engineering
"""

import pandas as pd
import numpy as np
import os
import yaml
from loguru import logger


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(filepath="data/ecommerce_raw.csv"):
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def data_quality_report(df):
    """Generate a full data quality report."""
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notnull().sum(),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "sample": [
            df[c].dropna().iloc[0] if df[c].notnull().any() else None
            for c in df.columns
        ],
    })

    logger.info("Data Quality Report Generated")
    print("\n" + "=" * 80)
    print("  DATA QUALITY REPORT")
    print("=" * 80)
    print(report.to_string())
    print(f"\n  Duplicate order_ids : {df.duplicated(subset='order_id').sum()}")
    print(f"  Total rows          : {len(df):,}")
    print("=" * 80)
    return report


def clean_data(df):
    """Full cleaning pipeline with detailed logging."""
    df = df.copy()
    steps = []

    # ── Step 1 ─ Remove Duplicates ───────────────────────
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    removed = before - len(df)
    steps.append(f"[1] Removed {removed} duplicate rows")

    # ── Step 2 ─ Fix Data Types ──────────────────────────
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])

    num_cols = [
        "unit_price", "quantity", "discount_pct", "subtotal",
        "discount_amount", "total_amount", "shipping_cost",
        "customer_age", "customer_rating",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    steps.append("[2] Fixed data types (dates, numerics)")

    # ── Step 3 ─ Standardize Categoricals ───────────────
    str_cols = [
        "region", "category", "product_name", "customer_gender",
        "payment_method", "shipping_type", "order_status",
    ]
    for col in str_cols:
        df[col] = df[col].str.strip().str.title()
    steps.append("[3] Standardized categorical columns (strip + title case)")

    # ── Step 4 ─ Handle Missing Values ──────────────────
    fills = {
        "customer_age": df["customer_age"].median(),
        "customer_gender": df["customer_gender"].mode()[0],
        "customer_rating": df["customer_rating"].median(),
        "region": df["region"].mode()[0],
    }
    missing_counts = {col: df[col].isnull().sum() for col in fills}
    df.fillna(fills, inplace=True)
    steps.append(
        f"[4] Filled missing: age({missing_counts['customer_age']}), "
        f"gender({missing_counts['customer_gender']}), "
        f"rating({missing_counts['customer_rating']}), "
        f"region({missing_counts['region']})"
    )

    # ── Step 5 ─ Fix Outliers ────────────────────────────
    neg = (df["unit_price"] < 0).sum()
    df["unit_price"] = df["unit_price"].abs()
    df["subtotal"] = (df["unit_price"] * df["quantity"]).round(2)
    df["discount_amount"] = (df["subtotal"] * df["discount_pct"] / 100).round(2)
    df["total_amount"] = (df["subtotal"] - df["discount_amount"]).round(2)
    df["customer_age"] = df["customer_age"].clip(18, 80)
    steps.append(f"[5] Fixed {neg} negative prices, recalculated totals, clipped age")

    # ── Step 6 ─ Feature Engineering ────────────────────
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_month_name"] = df["order_date"].dt.month_name()
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_quarter_label"] = "Q" + df["order_quarter"].astype(str)
    df["order_day_of_week"] = df["order_date"].dt.day_name()
    df["order_hour"] = df["order_datetime"].dt.hour
    df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_holiday_season"] = df["order_month"].isin([11, 12]).astype(int)
    df["revenue_per_item"] = (df["total_amount"] / df["quantity"]).round(2)
    df["has_discount"] = (df["discount_pct"] > 0).astype(int)

    bins = [17, 25, 35, 45, 55, 80]
    labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    df["age_group"] = pd.cut(df["customer_age"], bins=bins, labels=labels)

    q75 = df["total_amount"].quantile(0.75)
    df["is_high_value"] = (df["total_amount"] >= q75).astype(int)

    steps.append("[6] Engineered 13 new features")

    # ── Step 7 ─ Validate ────────────────────────────────
    assert df.isnull().sum().sum() == 0, "ERROR: Missing values remain!"
    assert (df["unit_price"] < 0).sum() == 0, "ERROR: Negative prices remain!"
    assert df.duplicated(subset="order_id").sum() == 0, "ERROR: Duplicates remain!"
    steps.append("[7] ✅ All validations passed")

    # ── Print Log ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CLEANING PIPELINE LOG")
    print("=" * 65)
    for s in steps:
        print(f"  {s}")
    print(f"\n  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 65)

    logger.success("Data cleaning completed successfully")
    return df


def save_clean_data(df, filepath="data/ecommerce_clean.csv"):
    df.to_csv(filepath, index=False)
    logger.info(f"Saved clean data → {filepath}")


if __name__ == "__main__":
    df_raw = load_raw_data()
    data_quality_report(df_raw)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean)