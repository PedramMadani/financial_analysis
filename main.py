import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import os


# Ensure matplotlib inline if using Jupyter notebooks
# %matplotlib inline


def fetch_data(stock_symbol, start_date, end_date):
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        # Forward fill to handle weekends and holidays
        data.ffill(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean()


def calculate_ema(data, window=20):
    return data['Close'].ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return upper_band, lower_band


def prepare_data(data):
    data['SMA'] = calculate_sma(data)
    data['EMA'] = calculate_ema(data)
    data['RSI'] = calculate_rsi(data)
    macd, signal_line = calculate_macd(data)
    data['MACD'] = macd
    data['Signal_Line'] = signal_line
    upper_band, lower_band = calculate_bollinger_bands(data)
    data['Upper_Band'] = upper_band
    data['Lower_Band'] = lower_band
    data.fillna(method='bfill', inplace=True)


def split_data(data):
    features = ['Close', 'SMA', 'EMA', 'RSI', 'MACD',
                'Signal_Line', 'Upper_Band', 'Lower_Band']
    X = data[features]
    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = data['Close'].shift(-1)
    y.fillna(method='ffill', inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')


def visualize_data(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['SMA'], label='SMA', color='red', linestyle='--')
    plt.plot(data['EMA'], label='EMA', color='green', linestyle='--')
    plt.title('Stock Price with Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def calculate_maximum_drawdown(return_series):
    """
    Calculates the maximum drawdown of a return series.
    """
    cumulative_returns = (1 + return_series).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_sharpe_ratio(return_series, risk_free_rate=0.0):
    """
    Calculates the Sharpe ratio for a return series based on a risk-free rate.
    """
    mean_return = return_series.mean()
    std_return = return_series.std()
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio


def calculate_var(return_series, confidence_level=0.95):
    """
    Calculates the Value at Risk (VaR) at a specified confidence level.
    """
    if isinstance(return_series, pd.Series):
        return_series = return_series.dropna()
    var = np.percentile(return_series, 100 * (1-confidence_level))
    return var


def walk_forward_analysis(data, n_splits):
    """
    Perform walk-forward analysis using time series cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    r2_scores = []

    for train_index, test_index in tscv.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        prepare_data(train)
        prepare_data(test)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train[features])
        X_test_scaled = scaler.transform(test[features])

        y_train = train['Close'].shift(-1).fillna(method='ffill')
        y_test = test['Close'].shift(-1).fillna(method='ffill')

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        last_model = None  # This will store the last trained model
        for train_index, test_index in tscv.split(data):
            # ... [rest of the code remains the same]
            last_model = model  # Store the model trained in the current split
    return mse_scores, r2_scores, last_model


def simple_backtest(data, model, initial_investment=10000):
    """
    A simple backtest that buys a stock based on model prediction and holds.
    """
    # Assuming no transaction costs for simplicity
    data['Predicted_Signal'] = model.predict(data[features])
    data['Order'] = np.where(data['Predicted_Signal'].shift(
        1) > data['Close'].shift(1), 1, -1)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Order'].shift(1) * data['Returns']
    data['Cumulative_Strategy_Returns'] = (
        1 + data['Strategy_Returns']).cumprod()

    initial_investment *= data['Cumulative_Strategy_Returns'].iloc[-1]

    # Calculate daily returns
    data['Strategy_Daily_Return'] = data['Strategy_Returns'].fillna(0)

    # Calculate risk metrics
    max_drawdown = calculate_maximum_drawdown(data['Strategy_Daily_Return'])
    sharpe_ratio = calculate_sharpe_ratio(data['Strategy_Daily_Return'])
    var = calculate_var(data['Strategy_Daily_Return'])

    return initial_investment, max_drawdown, sharpe_ratio, var


def visualize_data(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['SMA'], label='SMA', color='red', linestyle='--')
    plt.plot(data['EMA'], label='EMA', color='green', linestyle='--')
    plt.title('Stock Price with Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Define the path for saving the plot
    output_folder = 'output'
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    plot_filename = 'stock_price_with_indicators.png'
    plot_path = os.path.join(output_folder, plot_filename)

    plt.savefig(plot_path)  # Save the figure
    plt.close()  # Close the figure to free up memory

    print(f"Plot saved to {plot_path}")


def main():
    stock_symbol = input("Enter the stock symbol (default 'AAPL'): ") or 'AAPL'
    start_date = input(
        "Enter the start date in YYYY-MM-DD format (default '2023-01-01'): ") or '2023-01-01'
    end_date = input(
        "Enter the end date in YYYY-MM-DD format (default '2023-12-31'): ") or '2023-12-31'

    print(f"Fetching data for {stock_symbol} from {start_date} to {end_date}")
    data = fetch_data(stock_symbol, start_date, end_date)

    if data is not None:
        prepare_data(data)

        # We will use the features defined in split_data for walk-forward analysis and backtesting
        global features
        features = ['Close', 'SMA', 'EMA', 'RSI', 'MACD',
                    'Signal_Line', 'Upper_Band', 'Lower_Band']

        # Walk-forward Analysis
        n_splits = 5  # Define the number of splits for cross-validation
        mse_scores, r2_scores, last_model = walk_forward_analysis(
            data, n_splits)
        print(f'Walk-forward MSE scores: {mse_scores}')
        print(f'Walk-forward R^2 scores: {r2_scores}')
        mse_scores, r2_scores, model = walk_forward_analysis(data, n_splits)
        # Backtesting with the last model trained during the walk-forward analysis
        final_investment, max_drawdown, sharpe_ratio, var = simple_backtest(
            data, last_model)
        print(f'Final investment after backtesting: {final_investment}')
        print(f'Maximum Drawdown: {max_drawdown}')
        print(f'Sharpe Ratio: {sharpe_ratio}')
        print(f'Value at Risk (VaR): {var}')

        # Output file path
        output_folder = 'output'
        output_file = 'backtest_results.txt'
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_file)

        # Now write the outputs to the file
        with open(output_path, 'w') as file:
            file.write(
                f"Final investment after backtesting: {final_investment}\n")
            file.write(f"Maximum Drawdown: {max_drawdown}\n")
            file.write(f"Sharpe Ratio: {sharpe_ratio}\n")
            file.write(f"Value at Risk (VaR): {var}\n")

        print(f"Results saved to {output_path}")

        visualize_data(data)
    else:
        print("Failed to fetch or process data.")


if __name__ == "__main__":
    main()
