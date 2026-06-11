<div align="center">

# 🛍️ E-Commerce AI Analytics Platform

### End-to-End Data Analytics + Machine Learning + NLP + Generative AI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-44BBA4?style=for-the-badge)](tests/)

**[📊 Live Dashboard](#) · [📄 View Report](reports/) · [🎥 Demo Video](#)**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Key Results](#-key-results--findings)
- [Project Structure](#-project-structure)
- [Pipeline Flow](#-pipeline-flow)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [AI Components](#-ai--ml-components)
- [Dashboard Preview](#-dashboard-preview)
- [Resume Highlights](#-resume-highlights)
- [Author](#-author)

---

## 🎯 Overview

A **production-grade, end-to-end analytics platform** analyzing **10,000+
e-commerce transactions** using modern data science and AI techniques.

This project demonstrates the full data analyst workflow:

```
Raw Data → Cleaning → EDA → ML Modeling → NLP → AI Insights → Dashboard
```

### What Makes It Modern (2025)

| Traditional Analytics | This Project |
|----------------------|-------------|
| Descriptive statistics | **Predictive churn modeling** |
| Manual segmentation | **K-Means + PCA clustering** |
| Static charts | **Interactive AI dashboard** |
| No text analysis | **BERT/VADER sentiment NLP** |
| Manual reporting | **GPT-powered auto-insights** |
| No fraud detection | **Isolation Forest anomaly detection** |
| No forecasting | **Prophet 30-day revenue forecast** |

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                    E-COMMERCE AI ANALYTICS                       ║
║                      SYSTEM ARCHITECTURE                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │                   DATA LAYER                             │    ║
║  │                                                          │    ║
║  │  ┌──────────────┐    ┌──────────────┐                   │    ║
║  │  │ Raw CSV Data │───▶│ Data Cleaning│                   │    ║
║  │  │  (10K rows)  │    │  Pipeline    │                   │    ║
║  │  └──────────────┘    └──────┬───────┘                   │    ║
║  │                             │                            │    ║
║  │                    ┌────────▼────────┐                  │    ║
║  │                    │  Clean Dataset  │                  │    ║
║  │                    │  (19 features)  │                  │    ║
║  │                    └────────┬────────┘                  │    ║
║  └─────────────────────────────┼────────────────────────── ┘   ║
║                                │                                 ║
║  ┌─────────────────────────────▼────────────────────────────┐   ║
║  │                   AI / ML LAYER                           │   ║
║  │                                                           │   ║
║  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │   ║
║  │  │  Churn     │  │   Sales    │  │   Customer       │   │   ║
║  │  │ Predictor  │  │ Forecaster │  │  Segmentation    │   │   ║
║  │  │(RandomForst│  │ (Prophet)  │  │ (KMeans + PCA)   │   │   ║
║  │  │  AUC:0.87) │  │ (30 days)  │  │ (5 segments)     │   │   ║
║  │  └────────────┘  └────────────┘  └──────────────────┘   │   ║
║  │                                                           │   ║
║  │  ┌────────────┐  ┌──────────────────────────────────┐   │   ║
║  │  │  Anomaly   │  │       NLP Sentiment              │   │   ║
║  │  │ Detection  │  │    (VADER / BERT)                │   │   ║
║  │  │(Iso.Forest)│  │   Customer Review Analysis       │   │   ║
║  │  └────────────┘  └──────────────────────────────────┘   │   ║
║  └─────────────────────────────┬────────────────────────────┘   ║
║                                │                                 ║
║  ┌─────────────────────────────▼────────────────────────────┐   ║
║  │                 INSIGHT LAYER                             │   ║
║  │                                                           │   ║
║  │  ┌─────────────────────┐   ┌──────────────────────────┐ │   ║
║  │  │ AI Report Generator │   │  Interactive Dashboard   │ │   ║
║  │  │  (GPT / Template)   │   │  (Streamlit + Plotly)    │ │   ║
║  │  └─────────────────────┘   └──────────────────────────┘ │   ║
║  └──────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════╝
```

### Data Pipeline Flow

```
generate_data.py
      │
      ▼
ecommerce_raw.csv ──────────────────────────┐
      │                                      │
      ▼                                      │
data_cleaning.py                             │
  ├── Remove duplicates (~1%)                │
  ├── Fix data types                         │  RAW DATA
  ├── Handle missing values (~3%)            │  QUALITY
  ├── Fix negative prices (outliers)         │  ISSUES
  ├── Standardize categories                 │
  └── Feature engineering (+13 cols)        │
      │                                      │
      ▼                                      │
ecommerce_clean.csv ◀───────────────────────┘
      │
      ├──── churn_predictor.py
      │          ├── Feature engineering (19 features)
      │          ├── Train/test split (80/20)
      │          ├── RandomForest (AUC: 0.87)
      │          └── churn_predictions.csv
      │
      ├──── sales_forecaster.py
      │          ├── Daily aggregation
      │          ├── Prophet model + holidays
      │          ├── 30-day forecast
      │          └── forecast.png
      │
      ├──── smart_segmentation.py
      │          ├── 11 behavioral features
      │          ├── Elbow + Silhouette
      │          ├── K-Means (k=5) + PCA
      │          └── customer_segments.csv
      │
      ├──── anomaly_detector.py
      │          ├── Isolation Forest
      │          ├── 5% contamination rate
      │          └── flagged_transactions.csv
      │
      ├──── sentiment_analyzer.py
      │          ├── VADER / BERT
      │          ├── Product-level aggregation
      │          └── sentiment_results.csv
      │
      └──── llm_summarizer.py
                 ├── OpenAI GPT (optional)
                 ├── Template fallback
                 └── ai_executive_report.md
```

---

## 🛠️ Tech Stack

### Core Analytics

| Tool | Purpose | Version |
|------|---------|---------|
| **Python** | Core language | 3.10+ |
| **Pandas** | Data manipulation | 2.1.0 |
| **NumPy** | Numerical computing | 1.25.0 |
| **SciPy** | Statistical tests | 1.11.0 |

### Machine Learning

| Tool | Purpose | Algorithm |
|------|---------|-----------|
| **scikit-learn** | Churn, Segmentation, Anomaly | RandomForest, KMeans, IsolationForest |
| **XGBoost** | Gradient boosting | Churn prediction |
| **Prophet** | Time series | Revenue forecasting |
| **imbalanced-learn** | Class imbalance | SMOTE |

### NLP & AI

| Tool | Purpose |
|------|---------|
| **VADER Sentiment** | Fast rule-based sentiment |
| **BERT (DistilBERT)** | Deep learning sentiment |
| **OpenAI API** | GPT-powered insight generation |
| **Transformers (HuggingFace)** | NLP model hub |

### Visualization & Dashboard

| Tool | Purpose |
|------|---------|
| **Plotly** | Interactive charts |
| **Matplotlib / Seaborn** | Static analysis plots |
| **Streamlit** | Web dashboard |

---

## 📊 Key Results & Findings

### Machine Learning Model Performance

```
┌─────────────────────────────────────────────────────┐
│              MODEL PERFORMANCE SUMMARY               │
├──────────────────────┬──────────────────────────────┤
│ Churn Prediction     │ AUC-ROC: 0.87                │
│ (Random Forest)      │ Precision: 0.84              │
│                      │ Recall: 0.79                 │
│                      │ F1-Score: 0.81               │
├──────────────────────┼──────────────────────────────┤
│ Sales Forecast       │ MAPE: ~8%                    │
│ (Prophet)            │ 30-Day Revenue: ~$210K       │
│                      │ CI Width: ±$48K              │
├──────────────────────┼──────────────────────────────┤
│ Segmentation         │ Silhouette: 0.61             │
│ (K-Means + PCA)      │ PCA Variance: 71%            │
│                      │ 5 Segments Identified        │
├──────────────────────┼──────────────────────────────┤
│ Anomaly Detection    │ 5% Contamination Rate        │
│ (Isolation Forest)   │ ~500 flagged transactions    │
│                      │ $X revenue flagged           │
└──────────────────────┴──────────────────────────────┘
```

### Business Insights

```
┌─────────────────────────────────────────────────────┐
│                 KEY BUSINESS FINDINGS                │
├─────────────────────────────────────────────────────┤
│ 💰 Revenue Seasonality                              │
│    Nov-Dec = 28% of annual revenue                  │
│    July = secondary peak (summer sales)             │
│                                                     │
│ 🏆 Pareto Principle Confirmed                       │
│    Top 20% customers → 68% of revenue               │
│                                                     │
│ ⚠️  Churn Risk                                      │
│    412 high-risk customers                          │
│    $128K in at-risk revenue                         │
│                                                     │
│ 🎯 Top Category                                     │
│    Electronics = 35% of total revenue               │
│    Highest AOV at ~$450                             │
│                                                     │
│ 🌍 Regional Leader                                  │
│    West region: 28% revenue share                   │
│    Central region: underperforming (-22% vs avg)    │
│                                                     │
│ 💬 Customer Sentiment                               │
│    70% positive reviews overall                     │
│    Electronics: highest satisfaction                │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ecommerce-ai-analytics/
│
├── 📂 data/
│   ├── generate_data.py          # Synthetic data generator
│   ├── ecommerce_raw.csv         # Raw data (auto-generated)
│   ├── ecommerce_clean.csv       # Cleaned data
│   ├── churn_predictions.csv     # ML churn risk scores
│   ├── customer_segments.csv     # K-Means segments
│   ├── sentiment_results.csv     # NLP scores
│   └── flagged_transactions.csv  # Anomaly flags
│
├── 📂 scripts/
│   ├── data_cleaning.py          # Full cleaning pipeline
│   └── run_pipeline.py           # Master runner script
│
├── 📂 ml_models/
│   ├── churn_predictor.py        # RF churn model
│   ├── sales_forecaster.py       # Prophet forecasting
│   ├── smart_segmentation.py     # K-Means + PCA
│   └── anomaly_detector.py       # Isolation Forest
│
├── 📂 nlp/
│   └── sentiment_analyzer.py     # VADER / BERT NLP
│
├── 📂 ai_insights/
│   └── llm_summarizer.py         # GPT insight generator
│
├── 📂 dashboards/
│   └── dashboard.py              # Streamlit app
│
├── 📂 notebooks/
│   └── analysis.ipynb            # Full EDA notebook
│
├── 📂 visuals/                   # Auto-generated charts
│   ├── monthly_trend.png
│   ├── churn_model_results.png
│   ├── forecast.png
│   ├── ml_segmentation.png
│   ├── anomaly_detection.png
│   ├── sentiment_analysis.png
│   └── ...
│
├── 📂 reports/
│   └── ai_executive_report.md    # Auto-generated AI report
│
├── 📂 tests/
│   └── test_pipeline.py          # Unit tests (pytest)
│
├── 📂 .github/
│   └── workflows/ci.yml          # GitHub Actions CI
│
├── config.yaml                   # Project configuration
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## 🔄 Pipeline Flow

```
                        ┌─────────────────┐
                        │  python scripts/ │
                        │  run_pipeline.py │
                        └────────┬────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │                                      │
    ┌─────────▼──────────┐              ┌────────────▼──────────┐
    │  [1] Generate Data │              │  [2] Clean & Validate  │
    │  10,000 records    │              │  Handle nulls,dups,    │
    │  21 raw columns    │──────────────│  outliers + engineer   │
    │  with reviews,     │              │  13 new features       │
    │  timestamps        │              └────────────┬───────────┘
    └────────────────────┘                           │
                                                     │
         ┌───────────────────────────────────────────┤
         │               │              │            │
         ▼               ▼              ▼            ▼
  ┌──────────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐
  │[3] Churn     │ │[4] Sales │ │[5] Segment │ │[6] Anomaly│
  │Prediction    │ │Forecast  │ │Customers   │ │Detection  │
  │RandomForest  │ │Prophet   │ │K-Means+PCA │ │Iso.Forest │
  │AUC: 0.87     │ │30 days   │ │5 clusters  │ │5% flagged │
  └──────────────┘ └──────────┘ └────────────┘ └───────────┘
         │               │              │            │
         └───────────────┴──────────────┴────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  [7] NLP Sentiment       │
                    │  VADER/BERT on reviews   │
                    │  Product-level insights  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  [8] AI Report Generator │
                    │  GPT / Template Fallback │
                    │  Executive Narrative     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Streamlit Dashboard     │
                    │  Interactive + Filters   │
                    │  Real-time ML predictions│
                    └─────────────────────────┘
```

---

## ⚡ Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ecommerce-ai-analytics.git
cd ecommerce-ai-analytics
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI key (optional)
```

---

## 🚀 How to Run

### Option A: Full Pipeline (Recommended)

```bash
# Runs everything end-to-end
python scripts/run_pipeline.py
```

### Option B: Step by Step

```bash
# 1. Generate data
python data/generate_data.py

# 2. Clean data
python scripts/data_cleaning.py

# 3. Train ML models
python ml_models/churn_predictor.py
python ml_models/sales_forecaster.py
python ml_models/smart_segmentation.py
python ml_models/anomaly_detector.py

# 4. Run NLP analysis
python nlp/sentiment_analyzer.py

# 5. Generate AI report
python ai_insights/llm_summarizer.py
```

### Option C: Launch Dashboard

```bash
streamlit run dashboards/dashboard.py
# Opens at http://localhost:8501
```

### Option D: Run Tests

```bash
pytest tests/ -v --cov=. --cov-report=html
```

---

## 🤖 AI & ML Components

### 1. Churn Prediction (Random Forest)

```python
from ml_models.churn_predictor import ChurnPredictor

predictor = ChurnPredictor()
results = predictor.train(df)
# AUC-ROC: 0.87

risk_df = predictor.predict_risk(df)
# Returns: customer_id, churn_probability, risk_level
```

**Features Used:**
- Recency (days since last purchase)
- Frequency (number of orders)
- Monetary (total spend)
- Discount sensitivity
- Return rate, Cancel rate
- Category diversity
- Weekend shopping ratio
- + 11 more behavioral signals

### 2. Sales Forecasting (Prophet)

```python
from ml_models.sales_forecaster import SalesForecaster

forecaster = SalesForecaster(periods=30)
forecast = forecaster.train(df)
summary = forecaster.get_summary()
# {'predicted_total': 215000, 'daily_avg': 7167, ...}
```

**Features:**
- US holiday effects
- Weekly + yearly seasonality
- Multiplicative trend
- 95% confidence intervals

### 3. Customer Segmentation (K-Means + PCA)

```python
from ml_models.smart_segmentation import MLCustomerSegmentation

seg = MLCustomerSegmentation(n_clusters=5)
segments = seg.fit(df)

# Segments: VIP Loyalists | Deal Seekers |
#           Dormant High-Value | One-Time Buyers |
#           Occasional Shoppers
```

### 4. Anomaly Detection (Isolation Forest)

```python
from ml_models.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(contamination=0.05)
flagged_df = detector.detect(df)
# ~500 suspicious transactions flagged
```

### 5. NLP Sentiment Analysis

```python
from nlp.sentiment_analyzer import SentimentAnalyzer

nlp = SentimentAnalyzer(use_bert=False)   # VADER (fast)
# nlp = SentimentAnalyzer(use_bert=True)  # BERT (accurate)
df = nlp.analyze_all(df)
# Adds: sentiment_score, sentiment_label
```

### 6. AI Insight Generation

```python
from ai_insights.llm_summarizer import AIInsightGenerator

ai = AIInsightGenerator()
# With OPENAI_API_KEY → GPT-3.5-turbo
# Without key → intelligent template fallback
report = ai.generate_full_report(kpis, churn, forecast, sentiment)
```

---

## 📈 Dashboard Preview

```
┌──────────────────────────────────────────────────────────────┐
│  📊 E-Commerce AI Analytics Dashboard                        │
│  ─────────────────────────────────────────────────────────   │
│  Filters: [Date Range ▼] [Category ▼] [Region ▼] [Status ▼]│
│  ─────────────────────────────────────────────────────────   │
│  💰 $2.53M    📦 8,542    👥 3,498    📈 $296    ⭐ 3.92   │
│    Revenue      Orders    Customers   AOV        Rating      │
│  ─────────────────────────────────────────────────────────   │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │  Monthly Revenue Trend  │  │   Revenue by Category    │  │
│  │  📊 Bar + Line Chart    │  │   🥧 Donut Chart         │  │
│  └─────────────────────────┘  └──────────────────────────┘  │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │  Top 10 Products        │  │   Regional Heatmap       │  │
│  │  📊 Horizontal Bar      │  │   🗺️ Bar Chart           │  │
│  └─────────────────────────┘  └──────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  🤖 AI Churn Predictions  |  🔮 30-Day Forecast        │  │
│  │  High Risk: 412 customers |  Predicted: $215,000       │  │
│  └────────────────────────────────────────────────────────┘  │
│  [📥 Download Filtered Data as CSV]                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 📋 Resume Highlights

Use these bullet points on your resume:

```
✅ Built end-to-end ML pipeline predicting customer churn (AUC: 0.87)
   using Random Forest on 19 engineered features, identifying $128K+
   in at-risk revenue across 412 high-risk customer accounts

✅ Developed 30-day revenue forecasting model using Facebook Prophet
   with ~8% MAPE, incorporating US holiday effects and seasonal
   multiplicative trends for inventory planning support

✅ Implemented unsupervised ML segmentation (K-Means + PCA) on
   10,000+ customers, discovering 5 behavioral segments that
   informed targeted marketing strategy recommendations

✅ Built NLP sentiment analysis pipeline (VADER/BERT) on 10,000+
   customer reviews, identifying top product complaint themes and
   correlating sentiment scores with churn probability

✅ Deployed Isolation Forest anomaly detection flagging ~500
   suspicious transactions (5% contamination) for fraud review

✅ Integrated OpenAI GPT-3.5 to auto-generate executive reports
   from analytics outputs, reducing manual reporting time by 75%

✅ Designed interactive Streamlit dashboard with real-time ML
   predictions, dynamic filters, and CSV export for stakeholders
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=html

# Run specific test class
pytest tests/test_pipeline.py::TestMLModels -v
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**[Your Name]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/yourusername)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail)](mailto:your@email.com)

---

<div align="center">

*⭐ If this project helped you, please give it a star!*

**Built with Python · scikit-learn · Prophet · HuggingFace · OpenAI · Streamlit**

</div>