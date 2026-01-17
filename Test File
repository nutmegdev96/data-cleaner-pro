"""
Unit tests for Data Cleaner Pro
"""
import pytest
import pandas as pd
import numpy as np
from src.cleaner import DataCleaner


class TestDataCleaner:
    """Test suite for DataCleaner class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = {
            'ID': [1, 2, 3, 4, 5],
            'Name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'Age': [25, 30, 35, None, 45],
            'Salary': [50000, 60000, 75000, 80000, 95000],
            'Email': ['alice@test.com', 'bob@test', None, 'david@test.com', 'eve@test.com']
        }
        return pd.DataFrame(data)
    
    def test_initialization(self, sample_data):
        """Test DataCleaner initialization"""
        cleaner = DataCleaner(sample_data)
        assert cleaner.df is not None
        assert cleaner.original_shape == sample_data.shape
        
    def test_load_data(self, tmp_path):
        """Test data loading functionality"""
        # Create a test CSV file
        test_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        test_data.to_csv(test_file, index=False)
        
        cleaner = DataCleaner()
        cleaner.load_data(str(test_file))
        
        assert cleaner.df is not None
        assert cleaner.df.shape == (2, 2)
        
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling"""
        cleaner = DataCleaner(sample_data.copy())
        cleaner.handle_missing_values(strategy='auto')
        
        # Check no nulls remain
        assert cleaner.df.isna().sum().sum() == 0
        
    def test_standardize_column_names(self, sample_data):
        """Test column name standardization"""
        cleaner = DataCleaner(sample_data.copy())
        cleaner.standardize_column_names('snake')
        
        # Check all column names are snake_case
        for col in cleaner.df.columns:
            assert ' ' not in col
            assert col == col.lower()
            
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        data = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],
            'B': ['x', 'x', 'y', 'y', 'z']
        })
        
        cleaner = DataCleaner(data.copy())
        cleaner.remove_duplicates()
        
        assert len(cleaner.df) == 3  # Should have 3 unique rows
        
    def test_detect_outliers(self):
        """Test outlier detection"""
        data = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        cleaner = DataCleaner(data.copy())
        outliers = cleaner.detect_outliers(method='iqr', threshold=1.5)
        
        assert 'values' in outliers
        assert outliers['values']['count'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
