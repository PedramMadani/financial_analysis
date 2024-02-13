import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from your_package.models import train_and_evaluate


class TestTrainAndEvaluate(unittest.TestCase):
    def setUp(self):
        # Generate sample data for testing
        np.random.seed(0)
        self.data = pd.DataFrame({
            'Close': np.random.rand(100),
            'Feature1': np.random.rand(100),
            'Feature2': np.random.rand(100)
        })
        self.target = np.random.rand(100)

    def test_train_and_evaluate(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[['Feature1', 'Feature2']], self.target, test_size=0.2, random_state=42)

        # Train and evaluate the model
        mse_expected, r2_expected = self.train_and_evaluate_linear_regression(
            X_train, X_test, y_train, y_test)

        # Use the train_and_evaluate function
        mse_result, r2_result = train_and_evaluate(
            X_train, X_test, y_train, y_test)

        # Assert that the results match
        self.assertAlmostEqual(mse_expected, mse_result, places=5)
        self.assertAlmostEqual(r2_expected, r2_result, places=5)

    def train_and_evaluate_linear_regression(self, X_train, X_test, y_train, y_test):
        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2


if __name__ == '__main__':
    unittest.main()
