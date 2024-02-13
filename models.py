from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime


def split_data(data):
    """
    Splits the stock price data into training and testing sets.

    Parameters:
        data (pd.DataFrame): The DataFrame containing stock price data.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test DataFrames.
    """
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
    """
    Trains a linear regression model and evaluates its performance.

    Parameters:
        X_train (pd.DataFrame): The DataFrame containing training features.
        X_test (pd.DataFrame): The DataFrame containing testing features.
        y_train (pd.Series): The Series containing training labels.
        y_test (pd.Series): The Series containing testing labels.

    Returns:
        tuple: A tuple containing mean squared error and R^2 score.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
