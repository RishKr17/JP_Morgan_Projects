import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

# Load the data
data = pd.read_csv('Nat_Gas.csv')

# Convert Dates to datetime and set as index
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')
print(data['Dates'].head())
data.set_index('Dates', inplace=True)
data.columns = ['Price']  # Rename Prices to Price for clarity

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Price'], marker='o')
plt.title('Natural Gas Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Decompose the time series to see trend and seasonality
decomposition = seasonal_decompose(data['Price'], model='additive', period=12)
decomposition.plot()
plt.suptitle('Time Series Decomposition')
plt.tight_layout()
plt.show()

# Prepare data for modeling
data['Month'] = data.index.month
data['Year'] = data.index.year
data['Time'] = np.arange(len(data))  # Time index for regression

# Create polynomial features for non-linear trend
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(data[['Time']])

# Fit a model with polynomial trend and monthly seasonality
# We'll use linear regression for simplicity
X = pd.get_dummies(data['Month'], prefix='month', drop_first=True)
X = pd.concat([pd.Series(X_poly[:,1], index=data.index, name='Time'), 
               pd.Series(X_poly[:,2], index=data.index, name='Time_squared'),
               X], axis=1)

model = LinearRegression()
model.fit(X, data['Price'])

# Function to estimate price for any given date
def estimate_gas_price(input_date):
    # Convert input to datetime if it's a string
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%m/%d/%Y')
    
    # For dates within the observed range, return actual price if available
    if input_date in data.index:
        return data.loc[input_date, 'Price']
    
    # For future dates, predict using our model
    time_value = len(data) + (input_date - data.index[-1]).days / 30.44  # Approximate months
    
    # Create feature vector
    month = input_date.month
    features = {
        'Time': time_value,
        'Time_squared': time_value**2
    }
    
    # Add month dummies
    for m in range(2, 13):
        features[f'month_{m}'] = 1 if month == m else 0
    
    # Convert to DataFrame for prediction
    features_df = pd.DataFrame([features])
    
    # Ensure columns are in same order as training
    features_df = features_df[X.columns]
    
    # Predict price
    predicted_price = model.predict(features_df)[0]
    return predicted_price

# Example usage
print("Price on 12/15/2022:", estimate_gas_price('12/15/2022'))
print("Price on 3/31/2025 (forecast):", estimate_gas_price('3/31/2025'))

# Visualize the forecast
future_dates = pd.date_range(start=data.index[-1] + timedelta(days=31), periods=12, freq='M')
future_prices = [estimate_gas_price(date) for date in future_dates]

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Price'], label='Historical Prices')
plt.plot(future_dates, future_prices, 'r--', label='Forecasted Prices')
plt.title('Natural Gas Price History and Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()