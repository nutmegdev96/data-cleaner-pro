"""
Pytest configuration for the ecommerce-cleaner project
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import random

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@pytest.fixture
def sample_customer_data():
    """Fixture: sample customer data"""
    return pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'first_name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'last_name': ['Smith', 'Johnson', 'Brown', 'Wilson'],
        'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'diana@email.com'],
        'age': [25, 30, 35, 28],
        'city': ['New York', 'London', 'Berlin', 'Paris'],
        'country': ['US', 'UK', 'DE', 'FR'],
        'join_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']),
        'total_spent': [1250.50, 890.25, 450.75, 2100.00],
        'orders_count': [8, 5, 3, 12],
        'is_premium': [True, False, False, True]
    })

@pytest.fixture
def sample_transaction_data():
    """Fixture: sample transaction data"""
    return pd.DataFrame({
        'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
        'customer_id': ['C001', 'C002', 'C001', 'C003', 'C004'],
        'product_id': ['P100', 'P101', 'P102', 'P100', 'P103'],
        'product_name': ['MacBook Pro', 'iPhone 14', 'AirPods Pro', 'MacBook Pro', 'iPad Air'],
        'category': ['Electronics', 'Electronics', 'Accessories', 'Electronics', 'Tablets'],
        'quantity': [1, 2, 1, 1, 3],
        'unit_price': [2999.99, 999.50, 249.99, 2999.99, 599.99],
        'discount': [150.00, 50.00, 10.00, 0.00, 30.00],
        'tax': [200.00, 80.00, 20.00, 200.00, 45.00],
        'shipping': [25.00, 15.00, 5.00, 25.00, 10.00],
        'total_amount': [3074.99, 1959.00, 264.99, 3224.99, 1824.97],
        'currency': ['USD', 'USD', 'USD', 'USD', 'USD'],
        'payment_method': ['credit_card', 'paypal', 'credit_card', 'debit_card', 'credit_card'],
        'payment_status': ['completed', 'completed', 'completed', 'failed', 'completed'],
        'order_date': pd.to_datetime(['2023-05-10', '2023-05-11', '2023-05-12', '2023-05-13', '2023-05-14']),
        'shipping_date': pd.to_datetime(['2023-05-11', '2023-05-12', '2023-05-13', None, '2023-05-15']),
        'delivery_date': pd.to_datetime(['2023-05-15', '2023-05-16', '2023-05-17', None, '2023-05-19'])
    })

@pytest.fixture
def sample_product_data():
    """Fixture: sample product catalog"""
    return pd.DataFrame({
        'product_id': ['P100', 'P101', 'P102', 'P103', 'P104'],
        'name': ['MacBook Pro', 'iPhone 14', 'AirPods Pro', 'iPad Air', 'Apple Watch'],
        'category': ['Electronics', 'Electronics', 'Accessories', 'Tablets', 'Wearables'],
        'subcategory': ['Laptops', 'Phones', 'Audio', 'Tablets', 'Smartwatches'],
        'brand': ['Apple', 'Apple', 'Apple', 'Apple', 'Apple'],
        'price': [2999.99, 999.50, 249.99, 599.99, 399.99],
        'cost': [1800.00, 650.00, 150.00, 350.00, 250.00],
        'stock_quantity': [50, 200, 500, 150, 300],
        'supplier': ['Supplier A', 'Supplier B', 'Supplier A', 'Supplier C', 'Supplier B'],
        'rating': [4.8, 4.6, 4.7, 4.5, 4.4],
        'is_active': [True, True, True, True, False]
    })

@pytest.fixture
def sample_website_logs():
    """Fixture: sample website clickstream logs"""
    return pd.DataFrame({
        'session_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
        'customer_id': ['C001', 'C002', 'C001', 'C003', None],
        'page_url': ['/products/laptops', '/cart', '/checkout', '/products/phones', '/home'],
        'timestamp': pd.to_datetime([
            '2023-05-10 14:25:00',
            '2023-05-11 10:10:00',
            '2023-05-12 16:45:00',
            '2023-05-13 09:30:00',
            '2023-05-14 11:20:00'
        ]),
        'duration_seconds': [45, 120, 180, 60, 30],
        'device_type': ['desktop', 'mobile', 'desktop', 'tablet', 'mobile'],
        'browser': ['Chrome', 'Safari', 'Firefox', 'Chrome', 'Edge'],
        'country': ['US', 'UK', 'US', 'DE', 'FR'],
        'converted': [True, True, False, True, False]
    })

@pytest.fixture
def sample_data_with_missing():
    """Fixture: data with missing values for testing imputation"""
    return pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'age': [25, None, 35, 28],
        'income': [50000.0, 75000.0, None, 60000.0],
        'email': ['alice@email.com', None, 'charlie@email.com', 'diana@email.com'],
        'last_purchase': pd.to_datetime(['2023-11-20', '2023-10-15', None, '2023-12-01'])
    })

@pytest.fixture
def sample_data_with_outliers():
    """Fixture: data with outliers for testing outlier detection"""
    return pd.DataFrame({
        'customer_id': [f'C{i:03d}' for i in range(100)],
        'purchase_amount': [random.uniform(10, 1000) for _ in range(99)] + [100000]  # One extreme outlier
    })

@pytest.fixture
def temp_directory(tmp_path):
    """Fixture: temporary directory for file operations"""
    return tmp_path

@pytest.fixture
def mock_config():
    """Fixture: mock configuration for testing"""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'ecommerce_test',
            'user': 'test_user',
            'password': 'test_password'
        },
        'api': {
            'base_url': 'https://api.test.com',
            'key': 'test_api_key_123',
            'timeout': 30
        },
        'paths': {
            'data_raw': './data/raw',
            'data_processed': './data/processed',
            'models': './models',
            'logs': './logs'
        },
        'features': {
            'enable_feature_scaling': True,
            'enable_categorical_encoding': True,
            'outlier_threshold': 3.0
        }
    }

@pytest.fixture
def mock_database_connection(mocker):
    """Fixture: mock database connection"""
    mock_conn = mocker.Mock()
    mock_cursor = mocker.Mock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn

@pytest.fixture
def datetime_range():
    """Fixture: generate datetime range for time series testing"""
    start_date = datetime(2023, 1, 1)
    return [start_date + timedelta(days=i) for i in range(100)]

@pytest.fixture(scope="session")
def project_root():
    """Fixture: project root directory"""
    return Path(__file__).parent.parent

@pytest.fixture
def sample_time_series_data():
    """Fixture: sample time series data"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, 100).cumsum(),
        'visitors': np.random.poisson(500, 100),
        'conversion_rate': np.random.beta(2, 8, 100),
        'revenue': np.random.normal(5000, 1000, 100).cumsum()
    })

@pytest.fixture
def sample_json_config():
    """Fixture: sample JSON configuration file content"""
    return {
        "validation_rules": {
            "customer": {
                "email_pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                "age_min": 18,
                "age_max": 120,
                "countries_allowed": ["US", "UK", "DE", "FR", "IT", "ES", "CA", "AU"]
            },
            "transaction": {
                "amount_min": 0.01,
                "amount_max": 100000,
                "currencies_allowed": ["USD", "EUR", "GBP", "CAD", "AUD"]
            }
        },
        "cleaning_rules": {
            "impute_strategy": {
                "numeric": "median",
                "categorical": "mode",
                "datetime": "forward_fill"
            },
            "outlier_method": "iqr",
            "outlier_threshold": 1.5
        }
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, tmp_path):
    """Setup test environment before each test"""
    # Set environment variables
    monkeypatch.setenv('TEST_MODE', 'True')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    
    # Create temporary data directory
    data_dir = tmp_path / 'test_data'
    data_dir.mkdir()
    monkeypatch.setattr('src.config.DATA_DIR', str(data_dir))
    
    # Clean up after test
    yield
    
    # Optional: Cleanup code here if needed
