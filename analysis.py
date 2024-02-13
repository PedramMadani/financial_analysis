import pandas as pd


def calculate_sma(data, window=20):
    """
    Calculates the Simple Moving Average (SMA) for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock price data.
        window (int): Window size for the moving average calculation.

    Returns:
        pd.Series: Series containing SMA values.
    """
    return data['Close'].rolling(window=window).mean()


def calculate_ema(data, window=20):
    """
    Calculates the Exponential Moving Average (EMA) for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock price data.
        window (int): Window size for the moving average calculation.

    Returns:
        pd.Series: Series containing EMA values.
    """
    return data['Close'].ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI) for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock price data.
        window (int): Window size for the RSI calculation.

    Returns:
        pd.Series: Series containing RSI values.
    """
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD) for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock price data.
        short_window (int): Window size for the short-term EMA.
        long_window (int): Window size for the long-term EMA.
        signal_window (int): Window size for the signal line calculation.

    Returns:
        tuple: Tuple containing MACD values and signal line values.
    """
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(data, window=20):
    """
    Calculates the Bollinger Bands for the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock price data.
        window (int): Window size for the moving average calculation.

    Returns:
        tuple: Tuple containing upper band and lower band values.
    """
    sma = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return upper_band, lower_band


def prepare_data(data):
    """
    Prepares the stock price data by adding technical indicators.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock price data.

    Returns:
        None
    """
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
