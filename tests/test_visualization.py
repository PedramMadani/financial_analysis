import unittest
import pandas as pd
import numpy as np
from your_package.visualization import visualize_data
import matplotlib.pyplot as plt


class TestVisualizationFunctions(unittest.TestCase):
    def setUp(self):
        # Generate sample data for testing
        np.random.seed(0)
        self.data = pd.DataFrame({
            'Close': np.random.rand(100),
            'SMA': np.random.rand(100),
            'EMA': np.random.rand(100)
        })

    def test_visualize_data(self):
        # Test if plot is generated without errors
        try:
            visualize_data(self.data)
            plt.close()  # Close the plot to prevent it from showing during testing
        except Exception as e:
            self.fail(f"Failed to generate plot: {e}")


if __name__ == '__main__':
    unittest.main()
