"""
Enhanced E-Commerce Data Generator
Generates realistic transaction data with:
- Synthetic customer reviews for NLP
- Precise timestamps for time series
- Realistic data quality issues
- Behavioral patterns and seasonality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import yaml
from loguru import logger


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_review_text(rating, product, category):
    """Generate synthetic customer review based on rating."""
    positive = [
        f"Absolutely love this {product}! Great value for money.",
        f"Best {category} purchase I've made this year. Highly recommend!",
        f"Exceeded my expectations. Fast delivery too!",
        f"Amazing quality for the price. Will definitely buy again.",
        f"Perfect product. Exactly what I was looking for.",
        f"Super happy with this {product}. Works flawlessly!",
        f"Five stars! The {product} is fantastic quality.",
    ]
    neutral = [
        f"It's okay. Does what it says but nothing special.",
        f"Average {product}. Met basic expectations.",
        f"Decent quality, but shipping was slower than expected.",
        f"Good enough for the price I guess. Nothing extraordinary.",
        f"Fine product. Probably wouldn't repurchase though.",
    ]
    negative = [
        f"Very disappointed with this {product}. Broke after a week.",
        f"Terrible quality. Complete waste of money!",
        f"Nothing like the description. Returning immediately.",
        f"Worst {category} purchase ever. Avoid at all costs!",
        f"Overpriced and underdelivered. Do not recommend.",
        f"Product arrived damaged. Customer service was unhelpful.",
    ]

    if rating >= 4:
        return random.choice(positive)
    elif rating == 3:
        return random.choice(neutral)
    else:
        return random.choice(negative)


def generate_ecommerce_data(n_records=10000, seed=42):
    """Generate complete e-commerce dataset."""
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Generating {n_records} records with seed={seed}")

    # ── Category Configuration ──────────────────────────────
    categories = {
        "Electronics": {
            "products": [
                "Laptop", "Smartphone", "Tablet", "Headphones",
                "Smartwatch", "Camera", "Speaker", "Monitor",
            ],
            "price_range": (50, 1500),
            "weight": 0.25,
        },
        "Clothing": {
            "products": [
                "T-Shirt", "Jeans", "Jacket", "Dress",
                "Sneakers", "Hoodie", "Shorts", "Sweater",
            ],
            "price_range": (15, 200),
            "weight": 0.25,
        },
        "Home & Garden": {
            "products": [
                "Lamp", "Cushion", "Rug", "Vase",
                "Plant Pot", "Blanket", "Candle Set", "Wall Art",
            ],
            "price_range": (10, 300),
            "weight": 0.20,
        },
        "Books": {
            "products": [
                "Fiction Novel", "Textbook", "Cookbook", "Biography",
                "Self-Help", "Science Book", "History Book", "Art Book",
            ],
            "price_range": (8, 60),
            "weight": 0.15,
        },
        "Sports": {
            "products": [
                "Yoga Mat", "Dumbbell Set", "Running Shoes",
                "Bicycle", "Tennis Racket", "Football",
                "Fitness Tracker", "Water Bottle",
            ],
            "price_range": (10, 500),
            "weight": 0.15,
        },
    }

    regions = ["West", "East", "North", "South", "Central"]
    region_weights = [0.28, 0.25, 0.18, 0.17, 0.12]

    payment_methods = [
        "Credit Card", "Debit Card", "PayPal",
        "Bank Transfer", "Cash on Delivery",
    ]
    payment_weights = [0.35, 0.25, 0.20, 0.10, 0.10]

    shipping_types = ["Standard", "Express", "Same-Day"]
    shipping_weights = [0.55, 0.30, 0.15]
    shipping_costs = {"Standard": 5.99, "Express": 12.99, "Same-Day": 19.99}

    # ── Hour Distribution (Pre-computed and Normalized) ─────
    hour_weights = np.array([
        1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 6,
        6, 6, 5, 5, 6, 7, 8, 7, 6, 5, 3, 2
    ], dtype=float)
    hour_weights = hour_weights / hour_weights.sum()

    # ── Customer Pool ──────────────────────────────────────
    n_customers = int(n_records * 0.35)
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    cust_weights = np.random.pareto(1.5, n_customers) + 1
    cust_weights /= cust_weights.sum()

    # ── Date Setup ─────────────────────────────────────────
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []

    for i in range(n_records):
        order_id = f"ORD-{str(i + 1).zfill(6)}"
        customer_id = np.random.choice(customer_ids, p=cust_weights)

                # Date with seasonal boost
        day_offset = np.random.randint(0, date_range_days)
        order_date = start_date + timedelta(days=day_offset)
        month = order_date.month
        
        # Seasonal shift (FIXED to prevent 'day out of range' errors)
        if month in [11, 12] and random.random() < 0.6:
            new_month = random.choice([11, 12])
            # Use day=random.randint(1, 28) to ensure the date always exists
            order_date = order_date.replace(month=new_month, day=random.randint(1, 28))
        elif month == 7 and random.random() < 0.3:
            order_date = order_date.replace(month=7, day=random.randint(1, 28))

        # Precise timestamp (FIXED - using normalized weights)
        order_hour = int(np.random.choice(range(24), p=hour_weights))
        order_datetime = order_date.replace(
            hour=order_hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

        # Product
        cat_names = list(categories.keys())
        cat_weights_list = [categories[c]["weight"] for c in cat_names]
        category = np.random.choice(cat_names, p=cat_weights_list)
        product = random.choice(categories[category]["products"])

        # Pricing
        low, high = categories[category]["price_range"]
        unit_price = round(np.random.uniform(low, high), 2)
        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.50, 0.25, 0.13, 0.07, 0.05])
        discount_pct = np.random.choice(
            [0, 5, 10, 15, 20, 25, 30],
            p=[0.40, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05],
        )
        subtotal = round(unit_price * quantity, 2)
        discount_amount = round(subtotal * discount_pct / 100, 2)
        total_amount = round(subtotal - discount_amount, 2)

        shipping_type = np.random.choice(shipping_types, p=shipping_weights)
        shipping_cost = shipping_costs[shipping_type]

        region = np.random.choice(regions, p=region_weights)
        payment_method = np.random.choice(payment_methods, p=payment_weights)

        age = int(np.clip(np.random.normal(35, 12), 18, 70))
        gender = np.random.choice(["Male", "Female", "Other"], p=[0.48, 0.48, 0.04])

        rating = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.17, 0.35, 0.35]))

        status = np.random.choice(
            ["Delivered", "Shipped", "Processing", "Cancelled", "Returned"],
            p=[0.72, 0.08, 0.05, 0.08, 0.07],
        )

        review = generate_review_text(rating, product, category)

        records.append({
            "order_id": order_id,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "order_datetime": order_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": customer_id,
            "customer_age": age,
            "customer_gender": gender,
            "region": region,
            "category": category,
            "product_name": product,
            "unit_price": unit_price,
            "quantity": quantity,
            "discount_pct": discount_pct,
            "subtotal": subtotal,
            "discount_amount": discount_amount,
            "total_amount": total_amount,
            "shipping_type": shipping_type,
            "shipping_cost": shipping_cost,
            "payment_method": payment_method,
            "customer_rating": rating,
            "customer_review": review,
            "order_status": status,
        })

    df = pd.DataFrame(records)

    # ── Inject Data Quality Issues ──────────────────────────
    missing_idx = np.random.choice(df.index, size=int(n_records * 0.03), replace=False)
    df.loc[missing_idx[:50], "customer_age"] = np.nan
    df.loc[missing_idx[50:80], "customer_gender"] = np.nan
    df.loc[missing_idx[80:110], "customer_rating"] = np.nan
    df.loc[missing_idx[110:], "region"] = np.nan

    dup_idx = np.random.choice(df.index, size=int(n_records * 0.01), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    incon_idx = np.random.choice(df.index, size=50, replace=False)
    df.loc[incon_idx, "region"] = df.loc[incon_idx, "region"].apply(
        lambda x: x.upper() if pd.notna(x) and random.random() > 0.5 else (
            x.lower() if pd.notna(x) else x
        )
    )

    out_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[out_idx, "unit_price"] = -abs(df.loc[out_idx, "unit_price"])

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.success(f"Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    try:
        cfg = load_config()
        n = cfg["data"]["n_records"]
        seed = cfg["data"]["random_seed"]
    except Exception:
        n, seed = 10000, 42

    df = generate_ecommerce_data(n_records=n, seed=seed)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/ecommerce_raw.csv", index=False)
    logger.info("Saved → data/ecommerce_raw.csv")

    print("\n📊 Dataset Summary")
    print(f"   Shape       : {df.shape}")
    print(f"   Date Range  : {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"   Customers   : {df['customer_id'].nunique():,}")
    print(f"   Categories  : {df['category'].nunique()}")
    print(f"   Missing vals: {df.isnull().sum().sum()}")
    print(f"   Duplicates  : {df.duplicated(subset='order_id').sum()}")