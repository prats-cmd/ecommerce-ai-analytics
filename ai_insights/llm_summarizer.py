"""
AI-Powered Insight Generator
Uses OpenAI GPT or template fallback to generate
executive narratives from analytics results
"""

import os
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class AIInsightGenerator:
    """Generate natural language insights from data results."""

    def __init__(self):
        self.use_openai = os.getenv("USE_OPENAI", "False").lower() == "true"
        self.client = None
        if self.use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}. Using templates.")
                self.use_openai = False

    def generate_kpi_summary(self, kpis: dict) -> str:
        prompt = f"""
        You are a senior e-commerce data analyst writing a concise executive KPI summary.
        
        Data:
        - Total Revenue: {kpis.get('revenue', 'N/A')}
        - Orders: {kpis.get('orders', 'N/A')}
        - Unique Customers: {kpis.get('customers', 'N/A')}
        - Avg Order Value: {kpis.get('aov', 'N/A')}
        - Cancellation Rate: {kpis.get('cancellation_rate', 'N/A')}
        - Return Rate: {kpis.get('return_rate', 'N/A')}
        - Avg Rating: {kpis.get('avg_rating', 'N/A')}
        
        Write 2 short paragraphs highlighting performance and concerns.
        Be specific, data-driven, and action-oriented.
        """
        return self._call_llm(prompt) or self._kpi_template(kpis)

    def generate_churn_insight(self, churn_stats: dict) -> str:
        prompt = f"""
        Write a 1-paragraph business recommendation based on this churn analysis:
        - High Risk Customers: {churn_stats.get('high_risk', 'N/A')}
        - Revenue At Risk: ${churn_stats.get('revenue_at_risk', 0):,.0f}
        - Average Days Since Last Purchase (at-risk): {churn_stats.get('avg_recency', 'N/A')}
        - Model AUC: {churn_stats.get('auc', 'N/A')}
        
        Recommend specific retention strategies with expected business impact.
        """
        return self._call_llm(prompt) or self._churn_template(churn_stats)

    def generate_forecast_insight(self, forecast_stats: dict) -> str:
        prompt = f"""
        Interpret this 30-day revenue forecast for a business executive:
        - Predicted Revenue: ${forecast_stats.get('predicted_total', 0):,.0f}
        - Daily Average: ${forecast_stats.get('daily_avg', 0):,.0f}
        - Uncertainty Range: ±${forecast_stats.get('uncertainty_avg', 0):,.0f}
        - Best Forecasted Day: {forecast_stats.get('best_day', 'N/A')}
        
        Write 1 paragraph with business implications and planning recommendations.
        """
        return self._call_llm(prompt) or self._forecast_template(forecast_stats)

    def generate_full_report(self, kpis, churn_stats, forecast_stats, sentiment_stats) -> str:
        """Generate a complete executive report."""
        sections = [
            "# 📊 AI-Generated Executive Report",
            "---",
            "## 📈 Business Performance",
            self.generate_kpi_summary(kpis),
            "",
            "## ⚠️ Customer Retention Alert",
            self.generate_churn_insight(churn_stats),
            "",
            "## 🔮 Revenue Forecast",
            self.generate_forecast_insight(forecast_stats),
            "",
            "## 💬 Customer Voice",
            f"Overall sentiment: **{sentiment_stats.get('positive_pct', 0):.1f}%** positive reviews. "
            f"Key complaint themes: {sentiment_stats.get('top_complaints', 'N/A')}",
        ]
        return "\n".join(sections)

    def _call_llm(self, prompt) -> str:
        """Call OpenAI API."""
        if not self.use_openai or not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior e-commerce data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    # ── Template Fallbacks ──────────────────────────────────
    def _kpi_template(self, kpis):
        return (
            f"The business generated {kpis.get('revenue', 'N/A')} across "
            f"{kpis.get('orders', 'N/A')} orders with an average order value of "
            f"{kpis.get('aov', 'N/A')}. The cancellation rate of "
            f"{kpis.get('cancellation_rate', 'N/A')} requires attention.\n\n"
            f"Customer satisfaction stands at {kpis.get('avg_rating', 'N/A')}/5 "
            f"across {kpis.get('customers', 'N/A')} unique customers, "
            f"indicating healthy engagement levels."
        )

    def _churn_template(self, stats):
        return (
            f"{stats.get('high_risk', 'N/A')} customers are at high churn risk, "
            f"representing ${stats.get('revenue_at_risk', 0):,.0f} in potential lost revenue. "
            f"Immediate win-back campaigns targeting customers inactive for "
            f"{stats.get('avg_recency', 'N/A')} days are strongly recommended. "
            f"A personalized discount of 15-20% with loyalty incentives could recover "
            f"an estimated 10-15% of at-risk revenue within 60 days."
        )

    def _forecast_template(self, stats):
        return (
            f"The 30-day revenue forecast projects ${stats.get('predicted_total', 0):,.0f} "
            f"(daily average: ${stats.get('daily_avg', 0):,.0f}) with an uncertainty range "
            f"of ±${stats.get('uncertainty_avg', 0):,.0f}. "
            f"Peak revenue is expected on {stats.get('best_day', 'N/A')}. "
            f"Inventory and staffing should be aligned to meet projected demand spikes."
        )


if __name__ == "__main__":
    ai = AIInsightGenerator()

    kpis = {
        "revenue": "$2,534,211",
        "orders": "8,542",
        "customers": "3,498",
        "aov": "$296.70",
        "cancellation_rate": "8.1%",
        "return_rate": "6.9%",
        "avg_rating": "3.92/5",
    }
    churn = {"high_risk": 412, "revenue_at_risk": 128500, "avg_recency": 142, "auc": 0.87}
    forecast = {"predicted_total": 215000, "daily_avg": 7167, "uncertainty_avg": 48000, "best_day": "2025-11-28"}
    sentiment = {"positive_pct": 70.3, "top_complaints": "quality, delivery, packaging"}

    report = ai.generate_full_report(kpis, churn, forecast, sentiment)
    print(report)

    with open("reports/ai_executive_report.md", "w") as f:
        f.write(report)
    logger.info("Report saved → reports/ai_executive_report.md")