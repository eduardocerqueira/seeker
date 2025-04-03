#date: 2025-04-03T17:02:27Z
#url: https://api.github.com/gists/b696eb04470c247296f47f8855e5bc09
#owner: https://api.github.com/users/carloocchiena

# Augmented Dickey-Fuller Test (ADF)

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

"""
Se p-value < 0.05 possiamo rifiutare l’ipotesi nulla e concludere che la serie è stazionaria.
Se p-value > 0.05, la serie non è stazionaria e necessita di trasformazioni.
"""

result = adfuller(df['Prezzo'])

print("ADF Statistic: {:.4f}".format(result[0]))
print("p-value: {:.4f}".format(result[1]))
for key, value in result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")
    
df['Prezzo_diff'] = df['Prezzo'].diff()
df_diff = df.dropna(subset=['Prezzo_diff'])  # remove NaNs caused by differencing

# Re-test ADF on differenced series
result_diff = adfuller(df_diff['Prezzo_diff'])

print("ADF Statistic (differenced): {:.4f}".format(result_diff[0]))
print("p-value (differenced): {:.4f}".format(result_diff[1]))

# ACF & PACF plots

plt.figure(figsize=(12, 4))
plot_acf(df_diff['Prezzo_diff'], lags=50)
plt.title("Autocorrelation (ACF) - Differenced Series")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plot_pacf(df_diff['Prezzo_diff'], lags=50)
plt.title("Partial Autocorrelation (PACF) - Differenced Series")
plt.tight_layout()
plt.show()

# SARIMA model

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Drop NaNs from differencing (if any)
daily_data = df['Prezzo'].resample('D').mean()
daily_data = daily_data.asfreq('D') # daily average
daily_data = daily_data.fillna(method='ffill')  # safety fill if any missing

subset = daily_data['2024-01-01':'2024-03-31']

# Fit a SARIMA model
model = SARIMAX(
    subset,
    order=(1, 1, 1),             # p, d, q
    seasonal_order=(1, 1, 1, 7), # P, D, Q, S (weekly seasonality)
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
print(results.summary())
results.plot_diagnostics(figsize=(15, 8))
plt.tight_layout()
plt.show()

forecast = results.get_forecast(steps=168)
forecast_df = forecast.summary_frame()

# Plot
plt.figure(figsize=(14, 5))
plt.plot(daily_data[-168:], label='Observed (last 7 days)')
plt.plot(forecast_df['mean'], label='Forecast (next 7 days)', color='orange')
plt.fill_between(forecast_df.index,
                 forecast_df['mean_ci_lower'],
                 forecast_df['mean_ci_upper'],
                 color='orange', alpha=0.3)
plt.title('SARIMA Forecast - PUN €/MWh')
plt.xlabel('Date')
plt.ylabel('€/MWh')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()