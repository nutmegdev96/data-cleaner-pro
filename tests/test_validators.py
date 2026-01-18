"""
Test module validators.py
"""
import pytest
import pandas as pd
import numpy as np
from src.validators import DataValidator, EmailValidator

class TestDataValidator:
    """Test DataValidator"""
    
    def test_validate_email_valid(self):
        """Test email valide"""
        validator = EmailValidator()
        assert validator.validate("test@example.com") == True
        assert validator.validate("user.name@domain.co.uk") == True
    
    def test_validate_email_invalid(self):
        """Test email non valide"""
        validator = EmailValidator()
        assert validator.validate("invalid-email") == False
        assert validator.validate("@domain.com") == False
    
    def test_check_missing_values(self):
        """Test check miss values"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan],
            'B': ['a', None, 'c']
        })
        
        validator = DataValidator()
        missing = validator.check_missing_values(df)
        
        assert missing['A'] == 1
        assert missing['B'] == 1
    
    def test_validate_numeric_range(self):
        """Test validation range numeric"""
        validator = DataValidator()
        
        # Dati all'interno del range
        data = pd.Series([10, 20, 30])
        assert validator.validate_numeric_range(data, 0, 100) == True
        
        # Dati fuori range
        data = pd.Series([-10, 150])
        assert validator.validate_numeric_range(data, 0, 100) == False
    
    def test_validate_date_format(self):
        """Test validation data format"""
        validator = DataValidator()
        
        valid_dates = ['2023-01-01', '2023/12/31']
        invalid_dates = ['01-01-2023', 'invalid-date']
        
        for date in valid_dates:
            assert validator.validate_date_format(date, '%Y-%m-%d') == True
        
        for date in invalid_dates:
            assert validator.validate_date_format(date, '%Y-%m-%d') == False

class TestCustomerValidator:
    """Test specifici per validazione clienti e-commerce"""
    
    def test_validate_customer_age(self):
        """Test age client valid"""
        from src.validators import CustomerValidator
        
        validator = CustomerValidator()
        
        # Età valide
        assert validator.validate_age(18) == True
        assert validator.validate_age(100) == True
        
        # Età non valide
        assert validator.validate_age(-5) == False
        assert validator.validate_age(150) == False
    
    def test_validate_order_amount(self):
        """Test importo ordine valido"""
        from src.validators import CustomerValidator
        
        validator = CustomerValidator()
        
        assert validator.validate_order_amount(0.01) == True  # Minimo
        assert validator.validate_order_amount(10000.00) == True  # Massimo ragionevole
        assert validator.validate_order_amount(-100) == False  # Negativo
        assert validator.validate_order_amount(1000000) == False  # Troppo alto
    
    def test_validate_country_code(self):
        """Test country code valid"""
        from src.validators import CustomerValidator
        
        validator = CustomerValidator()
        
        valid_countries = ['US', 'IT', 'DE', 'FR', 'UK']
        invalid_countries = ['XX', 'USA', '', None]
        
        for country in valid_countries:
            assert validator.validate_country_code(country) == True
        
        for country in invalid_countries:
            assert validator.validate_country_code(country) == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
