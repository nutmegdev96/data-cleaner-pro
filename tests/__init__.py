"""
Test package for ecommerce-cleaner project.

This package contains all unit and integration tests for the project.
"""

__version__ = '1.1.0'
__author__ = 'nutmegdev96'
__description__ = 'Test suite for ecommerce-cleaner data processing library'

# Import key test utilities
from .conftest import (
    sample_customer_data,
    sample_transaction_data,
    sample_product_data,
    sample_website_logs,
    sample_data_with_missing,
    sample_data_with_outliers,
    mock_config
)

# Test discovery configuration
__all__ = [
    # Fixtures
    'sample_customer_data',
    'sample_transaction_data',
    'sample_product_data',
    'sample_website_logs',
    'sample_data_with_missing',
    'sample_data_with_outliers',
    'mock_config',
    
    # Test modules
    'test_cleaner',
    'test_validators',
    'test_transformers',
    'test_utils'
]

# Define test categories for selective test execution
TEST_CATEGORIES = {
    'unit': ['test_validators', 'test_utils'],
    'integration': ['test_cleaner', 'test_transformers'],
    'all': ['test_cleaner', 'test_validators', 'test_transformers', 'test_utils']
}

def get_test_modules(category='all'):
    """
    Get test modules for a specific category.
    
    Parameters:
    -----------
    category : str
        Test category: 'unit', 'integration', or 'all'
    
    Returns:
    --------
    list
        List of test module names
    """
    return TEST_CATEGORIES.get(category, TEST_CATEGORIES['all'])
