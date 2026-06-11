"""
Sales Forecasting with Facebook Prophet
Predicts next 30-day revenue with confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from loguru import logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Using linear trend fallback.")


class SalesForecaster:
    """Time series forecasting pipeline."""

    def __init__(self, periods=30):
        self.periods = periods
        self.model = None
        self.forecast = None

    def prepare_data(self, df):
        """Aggregate daily revenue for Prophet format."""
        completed = df[df["order_status"].isin(["Delivered", "Shipped", "Processing"])]
        daily = (
            completed.groupby("order_date")["total_amount"]
            .sum()
            .reset_index()
            .rename(columns={"order_date": "ds", "total_amount": "y"})
        )
        daily["ds"] = pd.to_datetime(daily["ds"])
        daily = daily.sort_values("ds").reset_index(drop=True)
        return daily

    def train(self, df):
        """Train forecasting model and generate predictions."""
        data = self.prepare_data(df)
        logger.info(f"Training on {len(data)} daily data points...")

        if PROPHET_AVAILABLE:
            self.model = Prophet(
                seasonality_mode="multiplicative",
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95,
            )
            self.model.add_country_holidays(country_name="US")
            self.model.fit(data)
            future = self.model.make_future_dataframe(periods=self.periods)
            self.forecast = self.model.predict(future)
        else:
            # Fallback: simple linear trend
            self.forecast = self._linear_fallback(data)

        self.actual_data = data
        return self.forecast

    def _linear_fallback(self, data):
        """Simple rolling average fallback if Prophet not available."""
        last_date = data["ds"].max()
        avg = data["y"].rolling(7).mean().iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(1, "d"), periods=self.periods)
        noise = np.random.normal(0, avg * 0.1, self.periods)
        forecast = pd.DataFrame({
            "ds": pd.concat([data["ds"], pd.Series(future_dates)]).values,
            "yhat": list(data["y"]) + list(avg + noise),
            "yhat_lower": list(data["y"] * 0.9) + list(avg * 0.85 + noise),
            "yhat_upper": list(data["y"] * 1.1) + list(avg * 1.15 + noise),
        })
        return forecast

    def get_summary(self):
        """Return next N-day forecast summary."""
        future_only = self.forecast.tail(self.periods)
        return {
            "predicted_total": round(future_only["yhat"].sum(), 2),
            "daily_avg": round(future_only["yhat"].mean(), 2),
            "best_day": future_only.loc[future_only["yhat"].idxmax(), "ds"].strftime("%Y-%m-%d"),
            "worst_day": future_only.loc[future_only["yhat"].idxmin(), "ds"].strftime("%Y-%m-%d"),
            "uncertainty_avg": round((future_only["yhat_upper"] - future_only["yhat_lower"]).mean(), 2),
            "confidence_pct": round(
                future_only["yhat"].sum() /
                (future_only["yhat_upper"].sum() + 1) * 100, 1
            ),
        }

    def plot(self, save_dir="visuals"):
        """Plot forecast with actuals."""
        os.makedirs(save_dir, exist_ok=True)

        if PROPHET_AVAILABLE and self.model:
            fig = self.model.plot(self.forecast, figsize=(14, 6))
            plt.title(f"{self.periods}-Day Revenue Forecast (Prophet)", fontsize=14, fontweight="bold")
            plt.xlabel("Date")
            plt.ylabel("Revenue ($)")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/forecast.png", dpi=150, bbox_inches="tight")
            plt.show()

            fig2 = self.model.plot_components(self.forecast)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/forecast_components.png", dpi=150, bbox_inches="tight")
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            n_actual = len(self.actual_data)
            ax.plot(self.forecast["ds"].iloc[:n_actual],
                    self.forecast["yhat"].iloc[:n_actual],
                    color="#2E86AB", label="Actual")
            ax.plot(self.forecast["ds"].iloc[n_actual:],
                    self.forecast["yhat"].iloc[n_actual:],
                    color="#F18F01", linestyle="--", label="Forecast")
            ax.fill_between(
                self.forecast["ds"].iloc[n_actual:],
                self.forecast["yhat_lower"].iloc[n_actual:],
                self.forecast["yhat_upper"].iloc[n_actual:],
                alpha=0.2, color="#F18F01", label="95% CI"
            )
            ax.set_title(f"{self.periods}-Day Revenue Forecast", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Revenue ($)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/forecast.png", dpi=150, bbox_inches="tight")
            plt.show()

        logger.info(f"Saved → {save_dir}/forecast.png")


if __name__ == "__main__":
    df = pd.read_csv("data/ecommerce_clean.csv")
    forecaster = SalesForecaster(periods=30)
    forecast = forecaster.train(df)
    summary = forecaster.get_summary()

    print("\n" + "=" * 50)
    print("  30-DAY REVENUE FORECAST")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")

    forecaster.plot()