import yfinance as yf


def fetch_data(stock_symbol, start_date, end_date):
    """
    Fetches historical stock price data from Yahoo Finance API.

    Parameters:
        stock_symbol (str): The stock symbol to fetch data for.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched stock data.
    """
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        # Forward fill to handle weekends and holidays
        data.ffill(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
