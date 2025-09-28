import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from arch import arch_model

TICKER = "^VIX"
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
FORECAST_HORIZON = 90
SCALE_FACTOR = 100

print(f"\n--- Downloading VIX data ({TICKER}) from {START_DATE} to {END_DATE} ---")
vix_df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
vix_df = vix_df[['Close']].rename(columns={'Close': 'y'})
vix_df.index.name = "Date"

print("\nData Sample:")
print(vix_df.head())
print("-" * 50)

# Visualization 1: Historical Prices
plt.figure(figsize=(12, 6))
plt.plot(vix_df.index, vix_df['y'], label='VIX Closing Price', color='blue')
plt.title('VIX Historical Prices (2020 - Present)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('VIX Value')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Returns calculation and visualization
vix_df['Returns'] = vix_df['y'].pct_change()

plt.figure(figsize=(12, 4))
plt.plot(vix_df.index, vix_df['Returns'], label='Daily Returns', color='orange', alpha=0.6)
plt.title('VIX Daily Percentage Returns', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

print("\nVIX Value Summary:")
print(vix_df['y'].describe())
print("-" * 50)

print("\n--- Phase 3: Prophet Modeling and Forecasting ---")

prophet_df = pd.DataFrame({
    'ds': vix_df.index.values.flatten(), 
    'y': vix_df['y'].values.astype(float).flatten() 
})
prophet_df = prophet_df.dropna()

train_df = prophet_df[:-FORECAST_HORIZON]
test_df = prophet_df[-FORECAST_HORIZON:].copy()
test_df = test_df[test_df['ds'].dt.dayofweek < 5].copy()

print("Training Prophet Model...")
m = Prophet(yearly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
m.fit(train_df)

future = m.make_future_dataframe(periods=FORECAST_HORIZON, freq='D')
future = future[future['ds'].dt.dayofweek < 5]
forecast = m.predict(future)

print("Forecasting Complete.")
print("Aligning actual and predicted data for RMSE calculation...")
results_df = test_df.merge(forecast[['ds', 'yhat']], on='ds', how='inner')

test_actual = results_df['y']
test_predicted = results_df['yhat']
rmse = np.sqrt(mean_squared_error(test_actual, test_predicted))

print(f"\nModel Evaluation ({FORECAST_HORIZON}-day Forecast):")
print(f"Prophet RMSE: {rmse:.2f}")

fig1 = m.plot(forecast)
fig1.axes[0].set_title(f"VIX {FORECAST_HORIZON}-Day Forecast (Prophet RMSE: {rmse:.2f})")
plt.show()

fig2 = m.plot_components(forecast)
plt.show()

print("Displayed Prophet Forecast Plot and Components.")

print("\n\n--- Phase 4: GARCH Modeling for Volatility ---")

vix_returns = vix_df['Returns'].dropna()
scaled_returns = vix_returns * SCALE_FACTOR
train_returns = scaled_returns[:-FORECAST_HORIZON]

print("Training GARCH(1, 1) Model (Scaled)...")
am = arch_model(train_returns, mean='Zero', vol='Garch', p=1, q=1, rescale=False)
res = am.fit(disp='off')
print(res.summary())

forecast_results = res.forecast(horizon=FORECAST_HORIZON, reindex=False)
garch_volatility_forecast = np.sqrt(forecast_results.variance.values[-1, :]) / SCALE_FACTOR

last_train_date = train_returns.index[-1]
# Fix for forecast_dates: The previous use of pd.to_datetime was redundant.
forecast_dates = pd.date_range(start=last_train_date, periods=FORECAST_HORIZON + 1, freq='B')[1:]

plt.figure(figsize=(12, 6))
plt.plot(res.conditional_volatility.index, res.conditional_volatility.values / SCALE_FACTOR,
         label='In-Sample Volatility', color='blue', alpha=0.7)

plt.plot(forecast_dates, garch_volatility_forecast,
         label=f'{FORECAST_HORIZON}-Day Volatility Forecast', color='red', linewidth=2)

plt.title('VIX Conditional Volatility Forecast (GARCH(1,1))', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Volatility ($\sigma$)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

print("\nDisplayed GARCH Forecast Plot. The red line represents the out-of-sample volatility forecast.")