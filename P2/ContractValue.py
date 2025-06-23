import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

class NaturalGasPricingModel:
    def __init__(self, data_file='Nat_Gas.csv'):
        """
        Initialize the natural gas pricing model with historical data
        :param data_file: Path to CSV file containing historical prices
        """
        # Load and prepare price data
        self.data = pd.read_csv(data_file)
        self.data['Dates'] = pd.to_datetime(self.data['Dates'], format='%m/%d/%y')
        self.data.set_index('Dates', inplace=True)
        self.data.columns = ['Price']
        
        # Prepare model features
        self._prepare_model()
        
    def _prepare_model(self):
        """Internal method to prepare the forecasting model"""
        # Add time features
        self.data['Month'] = self.data.index.month
        self.data['Year'] = self.data.index.year
        self.data['Time'] = np.arange(len(self.data))
        
        # Create polynomial time features
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(self.data[['Time']])
        
        # Create month dummy variables (dropping January to avoid collinearity)
        month_dummies = pd.get_dummies(self.data['Month'], prefix='month', drop_first=True)
        
        # Combine all features
        self.X = pd.concat([
            pd.Series(X_poly[:,1], index=self.data.index, name='Time'),
            pd.Series(X_poly[:,2], index=self.data.index, name='Time_squared'),
            month_dummies
        ], axis=1)
        
        # Train the model
        self.model = LinearRegression()
        self.model.fit(self.X, self.data['Price'])
        
    def estimate_price(self, date):
        """
        Estimate gas price for any given date (historical or future)
        :param date: datetime object or string in format 'MM/DD/YYYY'
        :return: estimated price
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%m/%d/%Y')
            
        # Return actual price if available
        if date in self.data.index:
            return self.data.loc[date, 'Price']
        
        # For future dates, predict using our model
        time_value = len(self.data) + (date - self.data.index[-1]).days / 30.44
        
        # Create feature vector
        month = date.month
        features = {
            'Time': time_value,
            'Time_squared': time_value**2
        }
        
        # Add month dummies
        for m in range(2, 13):
            features[f'month_{m}'] = 1 if month == m else 0
        
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([features])
        features_df = features_df[self.X.columns]  # Ensure correct column order
        
        return self.model.predict(features_df)[0]
    
    def plot_forecast(self, forecast_months=12):
        """Plot historical prices and forecast"""
        future_dates = pd.date_range(
            start=self.data.index[-1] + timedelta(days=31),
            periods=forecast_months,
            freq='M'
        )
        future_prices = [self.estimate_price(date) for date in future_dates]
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Price'], label='Historical Prices')
        plt.plot(future_dates, future_prices, 'r--', label='Forecasted Prices')
        plt.title('Natural Gas Price History and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price ($/MMBtu)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return future_dates, future_prices

class GasStorageContract:
    def __init__(self, pricing_model):
        """
        Initialize with a pricing model
        :param pricing_model: NaturalGasPricingModel instance
        """
        self.pricing_model = pricing_model
        
    def calculate_contract_value(self, injection_dates, withdrawal_dates, 
                               injection_rates, withdrawal_rates,
                               max_volume, storage_cost_per_month,
                               injection_cost_per_mmbtu=0.02,
                               withdrawal_cost_per_mmbtu=0.02,
                               transport_cost_per_transaction=50000):
        """
        Calculate the value of a storage contract
        
        :param injection_dates: list of dates when gas is injected ('MM/DD/YYYY' or datetime)
        :param withdrawal_dates: list of dates when gas is withdrawn
        :param injection_rates: list of injection rates in MMBtu/day
        :param withdrawal_rates: list of withdrawal rates in MMBtu/day
        :param max_volume: maximum storage capacity in MMBtu
        :param storage_cost_per_month: $ cost per month for storage
        :param injection_cost_per_mmbtu: $ cost per MMBtu for injection (default: $0.02)
        :param withdrawal_cost_per_mmbtu: $ cost per MMBtu for withdrawal (default: $0.02)
        :param transport_cost_per_transaction: $ cost per transport (default: $50,000)
        :return: dictionary with all calculated values
        """
        # Convert dates to datetime objects
        injection_dates = [self._parse_date(d) for d in injection_dates]
        withdrawal_dates = [self._parse_date(d) for d in withdrawal_dates]
        
        # Sort dates chronologically
        injection_dates.sort()
        withdrawal_dates.sort()
        
        # Calculate total injection and withdrawal volumes
        total_injection = sum(injection_rates)
        total_withdrawal = sum(withdrawal_rates)
        
        # Verify storage capacity constraint
        if total_injection > max_volume:
            raise ValueError(f"Total injection volume {total_injection:,} MMBtu exceeds max storage capacity {max_volume:,} MMBtu")
        
        # Calculate storage duration in months
        storage_duration = (max(withdrawal_dates) - min(injection_dates)).days / 30.44
        
        # Get prices for all dates
        injection_prices = [self.pricing_model.estimate_price(d) for d in injection_dates]
        withdrawal_prices = [self.pricing_model.estimate_price(d) for d in withdrawal_dates]
        
        # Calculate gross spread value
        gross_value = (sum(w_p * w_r for w_p, w_r in zip(withdrawal_prices, withdrawal_rates)) -
                       sum(i_p * i_r for i_p, i_r in zip(injection_prices, injection_rates)))
        
        # Calculate costs
        storage_cost = storage_cost_per_month * storage_duration
        injection_cost = total_injection * injection_cost_per_mmbtu
        withdrawal_cost = total_withdrawal * withdrawal_cost_per_mmbtu
        transport_cost = transport_cost_per_transaction * (len(injection_dates) + len(withdrawal_dates))
        
        total_costs = storage_cost + injection_cost + withdrawal_cost + transport_cost
        
        # Net contract value
        net_value = gross_value - total_costs
        
        return {
            'gross_value': gross_value,
            'storage_cost': storage_cost,
            'injection_cost': injection_cost,
            'withdrawal_cost': withdrawal_cost,
            'transport_cost': transport_cost,
            'total_costs': total_costs,
            'net_value': net_value,
            'injection_prices': injection_prices,
            'withdrawal_prices': withdrawal_prices,
            'storage_duration_months': storage_duration
        }
    
    def _parse_date(self, date):
        """Helper method to parse dates"""
        if isinstance(date, str):
            return datetime.strptime(date, '%m/%d/%Y')
        return date
        
    def print_valuation_report(self, valuation_results):
        """Print a formatted valuation report"""
        print("\n" + "="*50)
        print("NATURAL GAS STORAGE CONTRACT VALUATION")
        print("="*50)
        
        print("\nPRICE SUMMARY:")
        print(f"Injection Prices: ${', '.join(f'{p:.2f}' for p in valuation_results['injection_prices'])}/MMBtu")
        print(f"Withdrawal Prices: ${', '.join(f'{p:.2f}' for p in valuation_results['withdrawal_prices'])}/MMBtu")
        
        print("\nVALUATION RESULTS:")
        print(f"{'Gross Spread Value:':<30} ${valuation_results['gross_value']:>12,.2f}")
        print(f"{'Total Costs:':<30} ${valuation_results['total_costs']:>12,.2f}")
        print(f"{'Net Contract Value:':<30} ${valuation_results['net_value']:>12,.2f}")
        
        print("\nCOST BREAKDOWN:")
        print(f"{'Storage Costs:':<25} ${valuation_results['storage_cost']:>12,.2f} (for {valuation_results['storage_duration_months']:.1f} months)")
        print(f"{'Injection Costs:':<25} ${valuation_results['injection_cost']:>12,.2f}")
        print(f"{'Withdrawal Costs:':<25} ${valuation_results['withdrawal_cost']:>12,.2f}")
        print(f"{'Transport Costs:':<25} ${valuation_results['transport_cost']:>12,.2f}")
        print("="*50 + "\n")

# Example usage
if __name__ == "__main__":
    print("Initializing Natural Gas Pricing Model...")
    pricing_model = NaturalGasPricingModel()
    
    # Plot historical data and forecast
    print("\nGenerating price forecast...")
    pricing_model.plot_forecast()
    
    # Initialize contract valuation model
    contract_model = GasStorageContract(pricing_model)
    
    # Example contract parameters
    print("\nCalculating example contract valuation...")
    valuation = contract_model.calculate_contract_value(
        injection_dates=['06/15/2024', '07/15/2024'],
        withdrawal_dates=['12/15/2024', '01/15/2025'],
        injection_rates=[500000, 500000],  # 500,000 MMBtu/day each
        withdrawal_rates=[600000, 400000],  # 600,000 and 400,000 MMBtu/day
        max_volume=1000000,  # 1 million MMBtu
        storage_cost_per_month=100000,  # $100K/month
        injection_cost_per_mmbtu=0.02,
        withdrawal_cost_per_mmbtu=0.02,
        transport_cost_per_transaction=50000
    )
    
    # Print detailed report
    contract_model.print_valuation_report(valuation)