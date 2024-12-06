import unittest
import pandas as pd
from utils.preprocessing import load_data

class DataLoading(unittest.TestCase):
    def loaded_data_not_empty(self):
        """Test that loaded data is not empty"""
        df = load_data()  # Load the data
        # Check that the DataFrame is not empty
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertTrue(len(df) > 0, "DataFrame should have at least one row")
        self.assertTrue(len(df.columns) > 0, "DataFrame should have at least one column")
    
if __name__ == '__main__':
    unittest.main()
