"""
Configuration for Data Cleaner Pro
"""

from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    # File handling
    'default_encoding': 'utf-8',
    'csv_delimiter': ',',
    'excel_sheet': 0,
    
    # Missing value handling
    'missing_threshold': 0.3,
    'numeric_missing_strategy': 'median',
    'categorical_missing_strategy': 'mode',
    'datetime_missing_strategy': 'ffill',
    
    # Outlier detection
    'outlier_method': 'iqr',
    'outlier_threshold': 1.5,
    'outlier_treatment': 'cap',
    
    # Data type conversion
    'infer_dtypes': True,
    'convert_to_category_threshold': 50,  # Convert if unique values < threshold
    
    # Column standardization
    'column_case': 'snake',
    'remove_special_chars': True,
    
    # Validation
    'strict_validation': False,
    'log_validation_results': True,
    
    # Performance
    'use_multiprocessing': False,
    'chunk_size': 10000,
    'memory_efficient': True,
    
    # Logging
    'log_level': 'INFO',
    'log_file': 'data_cleaner.log',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}


class ConfigManager:
    """
    Configuration manager for Data Cleaner Pro
    
    #hint: Use to customize cleaning behavior without changing code
    #hint: Supports JSON config files for easy sharing
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        self.config.update(user_config)
    
    def save_config(self, config_file: str) -> None:
        """Save configuration to JSON file"""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self.config.update(updates)
    
    def get_cleaning_profile(self, profile: str = 'standard') -> Dict[str, Any]:
        """
        Get predefined cleaning profile
        
        Parameters:
        -----------
        profile : str
            Profile name: 'standard', 'strict', 'fast', 'comprehensive'
            
        Returns:
        --------
        dict
            Profile configuration
            
        #hint: Use profiles for different data quality requirements
        """
        profiles = {
            'standard': {
                'missing_threshold': 0.3,
                'outlier_method': 'iqr',
                'strict_validation': False,
            },
            'strict': {
                'missing_threshold': 0.1,
                'outlier_method': 'zscore',
                'outlier_threshold': 2.5,
                'strict_validation': True,
            },
            'fast': {
                'infer_dtypes': False,
                'use_multiprocessing': True,
                'memory_efficient': False,
            },
            'comprehensive': {
                'missing_threshold': 0.05,
                'outlier_method': 'percentile',
                'convert_to_category_threshold': 100,
                'strict_validation': True,
            }
        }
        
        return profiles.get(profile, profiles['standard'])
    
    def create_custom_profile(self, name: str, settings: Dict[str, Any]) -> None:
        """
        Create custom cleaning profile
        
        Parameters:
        -----------
        name : str
            Profile name
        settings : dict
            Profile settings
        """
        # Save to profiles directory
        profiles_dir = Path.home() / '.data_cleaner' / 'profiles'
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        profile_file = profiles_dir / f'{name}.json'
        with open(profile_file, 'w') as f:
            json.dump(settings, f, indent=2)
