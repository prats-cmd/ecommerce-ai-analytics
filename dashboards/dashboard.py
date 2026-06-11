"""
Interactive Streamlit Dashboard for E-Commerce AI Analytics
Run: streamlit run dashboards/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e1e4e8;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all datasets with caching."""
    data = {}
    
    # Main cleaned data
    if os.path.exists('data/ecommerce_clean.csv'):
        data['main'] = pd.read_csv('data/ecommerce_clean.csv')
        data['main']['order_date'] = pd.to_datetime(data['main']['order_date'])
    else:
        st.error("⚠️ data/ecommerce_clean.csv not found! Run: python scripts/run_pipeline.py")
        st.stop()
    
    # Optional ML outputs
    optional_files = {
        'churn': 'data/churn_predictions.csv',
        'segments': 'data/customer_segments.csv',
        'sentiment': 'data/sentiment_results.csv',
        'anomalies': 'data/flagged_transactions.csv',
    }
    
    for key, path in optional_files.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
    
    return data


# ── Load all data ────────────────────────────────────────────
data = load_data()
df = data['main']

# ══════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

# Date range filter
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()
date_range = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Category filter
categories = ['All'] + sorted(df['category'].unique().tolist())
selected_category = st.sidebar.selectbox("🏷️ Category", categories)

# Region filter
regions = ['All'] + sorted(df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("🌍 Region", regions)

# Order Status filter
statuses = ['All'] + sorted(df['order_status'].unique().tolist())
selected_status = st.sidebar.selectbox("📦 Order Status", statuses)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Use filters to drill down into specific segments")

# ── Apply Filters ────────────────────────────────────────────
filtered = df.copy()

if len(date_range) == 2:
    filtered = filtered[
        (filtered['order_date'].dt.date >= date_range[0]) &
        (filtered['order_date'].dt.date <= date_range[1])
    ]

if selected_category != 'All':
    filtered = filtered[filtered['category'] == selected_category]

if selected_region != 'All':
    filtered = filtered[filtered['region'] == selected_region]

if selected_status != 'All':
    filtered = filtered[filtered['order_status'] == selected_status]

completed = filtered[
    filtered['order_status'].isin(['Delivered', 'Shipped', 'Processing'])
]

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">📊 E-Commerce AI Analytics Dashboard</p>',
            unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align: center; color: gray;'>"
    f"Showing <b>{len(filtered):,}</b> orders | "
    f"<b>{filtered['customer_id'].nunique():,}</b> customers</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# KPI CARDS (ROW 1)
# ══════════════════════════════════════════════════════════════
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="💰 Total Revenue",
        value=f"${completed['total_amount'].sum()/1000:.1f}K",
        delta=f"{len(completed):,} orders"
    )

with col2:
    st.metric(
        label="📦 Total Orders",
        value=f"{len(completed):,}",
        delta=f"{(completed['quantity'].sum()):,} items"
    )

with col3:
    st.metric(
        label="👥 Unique Customers",
        value=f"{completed['customer_id'].nunique():,}",
        delta=f"{completed.shape[0]/max(completed['customer_id'].nunique(),1):.1f} orders/cust"
    )

with col4:
    st.metric(
        label="📈 Avg Order Value",
        value=f"${completed['total_amount'].mean():.2f}",
        delta=f"${completed['total_amount'].median():.2f} median"
    )

with col5:
    st.metric(
        label="⭐ Avg Rating",
        value=f"{completed['customer_rating'].mean():.2f}/5",
        delta=f"{(completed['customer_rating']>=4).mean()*100:.0f}% 4+ stars"
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS FOR DIFFERENT VIEWS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Sales Overview",
    "🏷️ Products & Categories",
    "👥 Customer Insights",
    "🤖 AI Predictions"
])

# ──────────────────────────────────────────────────────────────
# TAB 1: SALES OVERVIEW
# ──────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("📈 Monthly Revenue Trend")
        monthly = (
            completed.groupby(completed['order_date'].dt.to_period('M'))
            .agg(revenue=('total_amount', 'sum'),
                 orders=('order_id', 'count'))
            .reset_index()
        )
        monthly['order_date'] = monthly['order_date'].dt.to_timestamp()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=monthly['order_date'], y=monthly['revenue'],
                   name="Revenue", marker_color='#2E86AB',
                   hovertemplate='%{x|%b %Y}<br>Revenue: $%{y:,.0f}<extra></extra>'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=monthly['order_date'], y=monthly['orders'],
                       name="Orders", line=dict(color='#A23B72', width=3),
                       mode='lines+markers', marker=dict(size=8),
                       hovertemplate='%{x|%b %Y}<br>Orders: %{y:,}<extra></extra>'),
            secondary_y=True
        )
        fig.update_layout(height=400, hovermode='x unified',
                          legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="# Orders", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.subheader("🥧 Revenue by Category")
        cat_rev = completed.groupby('category')['total_amount'].sum().reset_index()
        fig2 = px.pie(cat_rev, values='total_amount', names='category',
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      hole=0.4)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Day of Week Analysis
    st.subheader("📅 Sales by Day of Week")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday', 'Sunday']
    daily = (
        completed.groupby('order_day_of_week')
        .agg(revenue=('total_amount', 'sum'),
             orders=('order_id', 'count'),
             aov=('total_amount', 'mean'))
        .reindex(day_order)
        .reset_index()
    )
    
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        fig_rev = px.bar(daily, x='order_day_of_week', y='revenue',
                         color='revenue', color_continuous_scale='Blues')
        fig_rev.update_layout(showlegend=False, height=300,
                              title="Revenue", xaxis_title="")
        st.plotly_chart(fig_rev, use_container_width=True)
    
    with col_y:
        fig_ord = px.bar(daily, x='order_day_of_week', y='orders',
                         color='orders', color_continuous_scale='Purples')
        fig_ord.update_layout(showlegend=False, height=300,
                              title="Orders", xaxis_title="")
        st.plotly_chart(fig_ord, use_container_width=True)
    
    with col_z:
        fig_aov = px.bar(daily, x='order_day_of_week', y='aov',
                         color='aov', color_continuous_scale='Oranges')
        fig_aov.update_layout(showlegend=False, height=300,
                              title="AOV", xaxis_title="")
        st.plotly_chart(fig_aov, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 2: PRODUCTS & CATEGORIES
# ──────────────────────────────────────────────────────────────
with tab2:
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("🏆 Top 10 Products by Revenue")
        top_prod = (
            completed.groupby('product_name')['total_amount']
            .sum().nlargest(10).reset_index()
        )
        fig3 = px.bar(top_prod, x='total_amount', y='product_name',
                      orientation='h', color='total_amount',
                      color_continuous_scale='Blues',
                      text=top_prod['total_amount'].apply(lambda x: f'${x:,.0f}'))
        fig3.update_layout(height=450,
                           yaxis={'categoryorder': 'total ascending'},
                           showlegend=False, xaxis_title="Revenue ($)",
                           yaxis_title="")
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_p2:
        st.subheader("🌍 Revenue by Region")
        reg_rev = (
            completed.groupby('region')
            .agg(revenue=('total_amount', 'sum'),
                 orders=('order_id', 'count'),
                 aov=('total_amount', 'mean'))
            .reset_index()
            .sort_values('revenue', ascending=True)
        )
        fig4 = px.bar(reg_rev, x='revenue', y='region', orientation='h',
                      color='region',
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      text=reg_rev['revenue'].apply(lambda x: f'${x:,.0f}'))
        fig4.update_layout(height=450, showlegend=False,
                           xaxis_title="Revenue ($)", yaxis_title="")
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)
    
    # Category deep dive
    st.subheader("📊 Category Performance Matrix")
    cat_matrix = (
        completed.groupby('category')
        .agg(revenue=('total_amount', 'sum'),
             orders=('order_id', 'count'),
             aov=('total_amount', 'mean'),
             rating=('customer_rating', 'mean'))
        .reset_index()
    )
    
    fig5 = px.scatter(cat_matrix, x='aov', y='orders',
                      size='revenue', color='category',
                      hover_name='category',
                      labels={'aov': 'Average Order Value ($)',
                              'orders': 'Number of Orders'},
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig5.update_traces(marker=dict(line=dict(width=2, color='white')))
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3: CUSTOMER INSIGHTS
# ──────────────────────────────────────────────────────────────
with tab3:
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.subheader("👤 Age Distribution")
        fig6 = px.histogram(completed, x='customer_age', nbins=30,
                            color_discrete_sequence=['#2E86AB'])
        fig6.update_layout(height=350,
                           xaxis_title="Age", yaxis_title="Count",
                           showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
    
    with col_d2:
        st.subheader("⭐ Rating Distribution")
        rating_counts = (
            completed['customer_rating'].value_counts()
            .sort_index().reset_index()
        )
        rating_counts.columns = ['rating', 'count']
        fig7 = px.bar(rating_counts, x='rating', y='count',
                      color='rating', color_continuous_scale='RdYlGn')
        fig7.update_layout(height=350, showlegend=False,
                           xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig7, use_container_width=True)
    
    with col_d3:
        st.subheader("💳 Payment Methods")
        pay = completed['payment_method'].value_counts().reset_index()
        pay.columns = ['method', 'count']
        fig8 = px.pie(pay, values='count', names='method',
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      hole=0.4)
        fig8.update_layout(height=350)
        st.plotly_chart(fig8, use_container_width=True)
    
    # Age group analysis
    st.subheader("👥 Revenue by Age Group & Gender")
    age_gender = (
        completed.groupby(['age_group', 'customer_gender'])['total_amount']
        .sum().reset_index()
    )
    fig9 = px.bar(age_gender, x='age_group', y='total_amount',
                  color='customer_gender', barmode='group',
                  color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
    fig9.update_layout(height=400,
                       xaxis_title="Age Group",
                       yaxis_title="Revenue ($)")
    st.plotly_chart(fig9, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 4: AI PREDICTIONS
# ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🤖 Machine Learning Insights")
    
    col_ml1, col_ml2 = st.columns(2)
    
    # Churn predictions
    with col_ml1:
        st.markdown("### ⚠️ Churn Risk Analysis")
        if 'churn' in data:
            churn_df = data['churn']
            risk_counts = churn_df['risk_level'].value_counts()
            
            fig_churn = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={
                    'Low Risk': '#44BBA4',
                    'Medium Risk': '#F18F01',
                    'High Risk': '#C73E1D'
                },
                hole=0.5
            )
            fig_churn.update_layout(height=350)
            st.plotly_chart(fig_churn, use_container_width=True)
            
            high_risk = (churn_df['risk_level'] == 'High Risk').sum()
            st.error(f"⚠️ **{high_risk:,} customers** at high churn risk")
            st.warning(f"💰 **${churn_df[churn_df['risk_level']=='High Risk']['monetary'].sum():,.0f}** in at-risk revenue")
        else:
            st.info("Churn data not available. Run: python ml_models/churn_predictor.py")
    
    # Customer Segments
    with col_ml2:
        st.markdown("### 🎯 Customer Segments")
        if 'segments' in data:
            seg_df = data['segments']
            if 'segment_name' in seg_df.columns:
                seg_counts = seg_df['segment_name'].value_counts()
                fig_seg = px.bar(
                    x=seg_counts.values,
                    y=seg_counts.index,
                    orientation='h',
                    color=seg_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_seg.update_layout(height=350, showlegend=False,
                                      xaxis_title="# Customers",
                                      yaxis_title="")
                st.plotly_chart(fig_seg, use_container_width=True)
            else:
                st.info("Segment names not found")
        else:
            st.info("Segmentation data not available")
    
    # Sentiment Analysis
    if 'sentiment' in data:
        st.markdown("### 💬 Customer Sentiment Analysis")
        sent_df = data['sentiment']
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sent_counts = sent_df['sentiment_label'].value_counts()
            fig_sent = px.pie(
                values=sent_counts.values,
                names=sent_counts.index,
                color=sent_counts.index,
                color_discrete_map={
                    'POSITIVE': '#44BBA4',
                    'NEUTRAL': '#F18F01',
                    'NEGATIVE': '#C73E1D'
                },
                hole=0.5
            )
            fig_sent.update_layout(height=350, title="Sentiment Distribution")
            st.plotly_chart(fig_sent, use_container_width=True)
        
        with col_s2:
            cat_sent = sent_df.groupby('category')['sentiment_score'].mean().sort_values()
            fig_cat_sent = px.bar(
                x=cat_sent.values, y=cat_sent.index,
                orientation='h',
                color=cat_sent.values,
                color_continuous_scale='RdYlGn'
            )
            fig_cat_sent.update_layout(height=350, showlegend=False,
                                       title="Avg Sentiment by Category",
                                       xaxis_title="Sentiment Score",
                                       yaxis_title="")
            st.plotly_chart(fig_cat_sent, use_container_width=True)
    
    # Anomalies
    if 'anomalies' in data:
        st.markdown("### 🚨 Anomaly Detection")
        anom_df = data['anomalies']
        st.error(f"🚨 **{len(anom_df):,} suspicious transactions** flagged for review")
        st.dataframe(
            anom_df[['order_id', 'customer_id', 'category',
                     'total_amount', 'discount_pct']].head(10),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📋 View & Download Raw Data"):
    st.dataframe(filtered.head(100), use_container_width=True)
    
    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="filtered_ecommerce_data.csv",
            mime="text/csv"
        )

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "<b>E-Commerce AI Analytics Platform</b> | "
    "Built with Python, scikit-learn, Prophet, Streamlit | "
    "<a href='https://github.com/prats-cmd/ecommerce-ai-analytics.git'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)