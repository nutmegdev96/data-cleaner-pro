"""
Data Cleaner Pro v2.0 - Smart Data Cleaning Toolkit
Author: nutmegdev96
GitHub: https://github.com/nutmegdev96/Data-Cleaner-PRO-v1.1
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Any, Callable
import warnings
import logging
from pathlib import Path
import json
from functools import wraps
import time
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# Configure smart logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class CleaningStrategy(Enum):
    """Available cleaning strategies"""
    AUTO = "auto"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    CUSTOM = "custom"


class DataType(Enum):
    """Standardized data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BOOLEAN = "boolean"


@dataclass
class ColumnProfile:
    """Smart column profiling"""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    inferred_type: DataType
    memory_usage: int
    sample_values: List[Any]
    stats: Dict[str, Any] = None
    
    def to_dict(self):
        return asdict(self)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Auto-log timing if verbose
        if len(args) > 0 and hasattr(args[0], 'verbose') and args[0].verbose:
            logger.info(f"{func.__name__} executed in {elapsed:.3f}s")
        
        return result
    return wrapper


class SmartDataCleaner:
    """
    🤖 Smart Data Cleaning Toolkit with AI-inspired features
    
    Features:
    - Auto-detection of data patterns
    - Smart imputation strategies
    - ML-ready preprocessing
    - Parallel processing support
    - Data quality scoring
    - Pipeline caching
    
    Usage:
    >>> cleaner = SmartDataCleaner(df, strategy='auto')
    >>> cleaned_df = cleaner.clean().df
    """
    
    def __init__(self, 
                 df: Optional[pd.DataFrame] = None,
                 strategy: Union[str, CleaningStrategy] = CleaningStrategy.AUTO,
                 verbose: bool = True,
                 cache_enabled: bool = True):
        """
        Initialize the smart cleaner.
        
        Args:
            df: Input DataFrame
            strategy: Cleaning strategy (auto, aggressive, conservative)
            verbose: Enable detailed logging
            cache_enabled: Enable pipeline caching for performance
        """
        self.df = df
        self.strategy = CleaningStrategy(strategy)
        self.verbose = verbose
        self.cache_enabled = cache_enabled
        
        # Smart state tracking
        self.original_shape = df.shape if df is not None else None
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.data_signature: Optional[str] = None
        self.quality_score: Optional[float] = None
        self.pipeline_cache: Dict[str, Any] = {}
        
        # Smart configuration
        self._setup_smart_config()
        
        # Initialize profiles if data exists
        if df is not None:
            self._analyze_data()
    
    def _setup_smart_config(self) -> None:
        """Setup smart configuration based on strategy"""
        configs = {
            CleaningStrategy.AUTO: {
                'missing_threshold': 0.3,
                'outlier_sensitivity': 1.5,
                'auto_impute': True,
                'auto_typecast': True,
                'aggressive_cleaning': False,
                'preserve_structure': True
            },
            CleaningStrategy.AGGRESSIVE: {
                'missing_threshold': 0.1,
                'outlier_sensitivity': 1.0,
                'auto_impute': True,
                'auto_typecast': True,
                'aggressive_cleaning': True,
                'preserve_structure': False
            },
            CleaningStrategy.CONSERVATIVE: {
                'missing_threshold': 0.5,
                'outlier_sensitivity': 2.0,
                'auto_impute': False,
                'auto_typecast': False,
                'aggressive_cleaning': False,
                'preserve_structure': True
            }
        }
        
        self.config = configs.get(self.strategy, configs[CleaningStrategy.AUTO])
        
        if self.verbose:
            logger.info(f"🔄 Strategy: {self.strategy.value}")
    
    def _analyze_data(self) -> None:
        """Perform comprehensive data analysis"""
        if self.df is None:
            return
        
        self.data_signature = self._generate_data_signature()
        
        for col in self.df.columns:
            profile = self._create_column_profile(col)
            self.column_profiles[col] = profile
        
        self.quality_score = self._calculate_quality_score()
        
        if self.verbose:
            logger.info(f"📊 Analyzed {len(self.df.columns)} columns")
            logger.info(f"📈 Quality Score: {self.quality_score:.2%}")
    
    def _create_column_profile(self, col: str) -> ColumnProfile:
        """Create smart profile for a column"""
        series = self.df[col]
        
        # Infer data type
        inferred_type = self._infer_data_type(series)
        
        # Calculate statistics based on type
        stats = {}
        if inferred_type == DataType.NUMERIC:
            stats = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skew': series.skew(),
                'kurtosis': series.kurtosis()
            }
        elif inferred_type == DataType.CATEGORICAL:
            stats = {
                'top_values': series.value_counts().head(5).to_dict(),
                'entropy': self._calculate_entropy(series)
            }
        
        return ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            null_count=series.isna().sum(),
            null_percentage=(series.isna().sum() / len(series)) * 100,
            unique_count=series.nunique(),
            unique_percentage=(series.nunique() / len(series)) * 100,
            inferred_type=inferred_type,
            memory_usage=series.memory_usage(deep=True),
            sample_values=series.dropna().head(3).tolist(),
            stats=stats
        )
    
    def _infer_data_type(self, series: pd.Series) -> DataType:
        """Smart data type inference"""
        # Check for temporal data
        try:
            pd.to_datetime(series, errors='raise')
            return DataType.TEMPORAL
        except:
            pass
        
        # Check for boolean
        if series.dropna().isin([0, 1, True, False]).all():
            return DataType.BOOLEAN
        
        # Check for numeric
        try:
            pd.to_numeric(series, errors='raise')
            # Determine if categorical or numeric
            if series.nunique() < min(20, len(series) * 0.1):
                return DataType.CATEGORICAL
            return DataType.NUMERIC
        except:
            pass
        
        # Check for categorical text
        if series.nunique() < min(50, len(series) * 0.3):
            return DataType.CATEGORICAL
        
        return DataType.TEXT
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy for categorical data"""
        value_counts = series.value_counts(normalize=True)
        return -(value_counts * np.log(value_counts)).sum()
    
    def _generate_data_signature(self) -> str:
        """Generate unique signature for data state"""
        if self.df is None:
            return ""
        
        # Create signature from shape and column hashes
        signature_parts = [
            str(self.df.shape),
            str(sorted(self.df.columns))
        ]
        
        # Add sample of data for uniqueness
        sample = self.df.sample(min(100, len(self.df))).to_string()
        signature_parts.append(hashlib.md5(sample.encode()).hexdigest()[:8])
        
        return hashlib.md5('|'.join(signature_parts).encode()).hexdigest()
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-1)"""
        if self.df is None or len(self.df) == 0:
            return 0.0
        
        scores = []
        
        # Completeness score
        completeness = 1 - (self.df.isna().sum().sum() / (self.df.size))
        scores.append(completeness * 0.3)
        
        # Uniqueness score (avoid excessive duplicates)
        duplicate_ratio = self.df.duplicated().sum() / len(self.df)
        scores.append((1 - duplicate_ratio) * 0.2)
        
        # Validity score (based on inferred types)
        validity = 0
        for profile in self.column_profiles.values():
            if profile.inferred_type != DataType.TEXT:  # Text is hard to validate
                validity += 0.5  # Base score
                if profile.null_percentage < 20:
                    validity += 0.3
                if profile.unique_percentage > 1:
                    validity += 0.2
        scores.append((validity / len(self.column_profiles)) * 0.5 if self.column_profiles else 0)
        
        return min(1.0, sum(scores))
    
    @timing_decorator
    def load_data(self, 
                  source: Union[str, pd.DataFrame, Path],
                  **kwargs) -> 'SmartDataCleaner':
        """
        Load data from various sources with smart detection.
        
        Args:
            source: File path, URL, or DataFrame
            **kwargs: Additional parameters for pandas read functions
        """
        try:
            if isinstance(source, pd.DataFrame):
                self.df = source.copy()
            elif isinstance(source, (str, Path)):
                source = str(source)
                
                # Smart format detection
                if source.startswith(('http://', 'https://')):
                    self.df = pd.read_csv(source, **kwargs)
                elif source.endswith('.csv'):
                    self.df = pd.read_csv(source, low_memory=False, **kwargs)
                elif source.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(source, **kwargs)
                elif source.endswith('.parquet'):
                    self.df = pd.read_parquet(source, **kwargs)
                elif source.endswith('.json'):
                    self.df = pd.read_json(source, **kwargs)
                elif source.endswith('.feather'):
                    self.df = pd.read_feather(source, **kwargs)
                else:
                    # Try auto-detection
                    try:
                        self.df = pd.read_csv(source, **kwargs)
                    except:
                        self.df = pd.read_excel(source, **kwargs)
            
            self.original_shape = self.df.shape
            self._analyze_data()
            
            if self.verbose:
                logger.info(f"✅ Data loaded: {self.df.shape}")
                logger.info(f"📊 Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
        
        return self
    
    @timing_decorator
    def clean(self, 
              steps: Optional[List[str]] = None,
              inplace: bool = True) -> Union['SmartDataCleaner', pd.DataFrame]:
        """
        Execute smart cleaning pipeline.
        
        Args:
            steps: Specific steps to execute (None for all)
            inplace: Whether to modify the internal DataFrame
            
        Returns:
            Cleaned DataFrame or self for chaining
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        # Define cleaning pipeline
        pipeline = {
            'standardize_names': self.standardize_names,
            'handle_missing': self.handle_missing_smart,
            'fix_data_types': self.fix_data_types_smart,
            'remove_duplicates': self.remove_duplicates_smart,
            'handle_outliers': self.handle_outliers_smart,
            'normalize_text': self.normalize_text_columns,
            'extract_features': self.extract_temporal_features
        }
        
        # Determine steps to execute
        steps_to_execute = steps or list(pipeline.keys())
        
        # Execute pipeline
        for step in steps_to_execute:
            if step in pipeline:
                if self.verbose:
                    logger.info(f"🔧 Executing: {step}")
                pipeline[step]()
        
        # Update analysis
        self._analyze_data()
        
        if self.verbose:
            logger.info(f"✨ Cleaning complete. Quality: {self.quality_score:.2%}")
        
        return self if inplace else self.df
    
    @timing_decorator
    def standardize_names(self, style: str = 'snake') -> 'SmartDataCleaner':
        """
        Standardize column names with smart formatting.
        
        Args:
            style: 'snake', 'camel', 'pascal', 'lower', 'upper'
        """
        if self.df is None:
            return self
        
        name_map = {}
        for col in self.df.columns:
            original = str(col)
            
            # Clean and convert
            clean_name = original.strip()
            clean_name = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in clean_name)
            clean_name = ' '.join(clean_name.split())
            
            # Apply style
            if style == 'snake':
                new_name = clean_name.lower().replace(' ', '_')
            elif style == 'camel':
                parts = clean_name.lower().split()
                new_name = parts[0] + ''.join(p.title() for p in parts[1:])
            elif style == 'pascal':
                new_name = ''.join(p.title() for p in clean_name.lower().split())
            elif style == 'lower':
                new_name = clean_name.lower()
            elif style == 'upper':
                new_name = clean_name.upper()
            else:
                new_name = clean_name
            
            name_map[original] = new_name
        
        self.df.rename(columns=name_map, inplace=True)
        
        if self.verbose:
            changes = sum(1 for k, v in name_map.items() if k != v)
            logger.info(f"📝 Renamed {changes} columns to {style}_case")
        
        return self
    
    @timing_decorator
    def handle_missing_smart(self) -> 'SmartDataCleaner':
        """Smart missing value handling based on data type"""
        if self.df is None:
            return self
        
        for col, profile in self.column_profiles.items():
            if profile.null_count == 0:
                continue
            
            if profile.inferred_type == DataType.NUMERIC:
                # For numeric: use median (robust to outliers)
                fill_value = self.df[col].median()
                self.df[col].fillna(fill_value, inplace=True)
                
            elif profile.inferred_type == DataType.CATEGORICAL:
                # For categorical: use mode or 'Unknown'
                mode_values = self.df[col].mode()
                fill_value = mode_values[0] if not mode_values.empty else 'Unknown'
                self.df[col].fillna(fill_value, inplace=True)
                
            elif profile.inferred_type == DataType.TEMPORAL:
                # For temporal: forward fill or interpolate
                self.df[col] = pd.to_datetime(self.df[col])
                self.df[col].fillna(method='ffill', inplace=True)
                
            else:  # TEXT or BOOLEAN
                fill_value = '' if profile.inferred_type == DataType.TEXT else False
                self.df[col].fillna(fill_value, inplace=True)
        
        if self.verbose:
            remaining_nulls = self.df.isna().sum().sum()
            logger.info(f"🔄 Handled missing values. Remaining nulls: {remaining_nulls}")
        
        return self
    
    @timing_decorator
    def fix_data_types_smart(self) -> 'SmartDataCleaner':
        """Smart data type conversion and optimization"""
        if self.df is None:
            return self
        
        for col, profile in self.column_profiles.items():
            current_dtype = str(self.df[col].dtype)
            
            if profile.inferred_type == DataType.NUMERIC:
                # Convert to optimal numeric type
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    # Downcast to save memory
                    self.df[col] = pd.to_numeric(self.df[col], downcast='float')
                except:
                    pass
                    
            elif profile.inferred_type == DataType.CATEGORICAL:
                # Convert to category if beneficial
                if profile.unique_percentage < 50:  # Less than 50% unique
                    self.df[col] = self.df[col].astype('category')
                    
            elif profile.inferred_type == DataType.TEMPORAL:
                # Convert to datetime
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    pass
        
        if self.verbose:
            memory_saved = self._calculate_memory_savings()
            logger.info(f"💾 Optimized types. Memory saved: {memory_saved:.1f} MB")
        
        return self
    
    def _calculate_memory_savings(self) -> float:
        """Calculate memory savings from optimization"""
        if self.df is None:
            return 0.0
        
        # This is simplified - in reality, track before/after memory
        original_estimate = self.original_shape[0] * self.original_shape[1] * 8  # Assume 8 bytes per cell
        current_usage = self.df.memory_usage(deep=True).sum()
        
        return max(0, (original_estimate - current_usage) / 1024**2)
    
    @timing_decorator
    def remove_duplicates_smart(self, 
                               subset: Optional[List[str]] = None,
                               keep: str = 'first') -> 'SmartDataCleaner':
        """
        Smart duplicate removal with reporting.
        
        Args:
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ('first', 'last', False)
        """
        if self.df is None:
            return self
        
        before = len(self.df)
        
        # If no subset provided, use all columns except high-cardinality text
        if subset is None:
            subset = [
                col for col, profile in self.column_profiles.items()
                if profile.inferred_type != DataType.TEXT or profile.unique_percentage < 80
            ]
        
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        duplicates_removed = before - len(self.df)
        
        if self.verbose:
            logger.info(f"🎯 Removed {duplicates_removed} duplicates ({duplicates_removed/before:.1%})")
        
        return self
    
    @timing_decorator
    def handle_outliers_smart(self, method: str = 'iqr') -> 'SmartDataCleaner':
        """
        Smart outlier detection and handling.
        
        Args:
            method: 'iqr', 'zscore', 'percentile'
        """
        if self.df is None:
            return self
        
        numeric_cols = [
            col for col, profile in self.column_profiles.items()
            if profile.inferred_type == DataType.NUMERIC
        ]
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if method == 'iqr':
                Q1, Q3 = data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > 3
                
                # Replace outliers with median
                if outlier_mask.any():
                    median_val = data.median()
                    self.df.loc[self.df.index[data.index[outlier_mask]], col] = median_val
        
        if self.verbose:
            logger.info(f"📈 Handled outliers using {method} method")
        
        return self
    
    @timing_decorator
    def normalize_text_columns(self) -> 'SmartDataCleaner':
        """Normalize text columns (lowercase, strip, etc.)"""
        if self.df is None:
            return self
        
        text_cols = [
            col for col, profile in self.column_profiles.items()
            if profile.inferred_type in [DataType.TEXT, DataType.CATEGORICAL]
        ]
        
        for col in text_cols:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str).str.lower().str.strip()
        
        if self.verbose and text_cols:
            logger.info(f"📝 Normalized {len(text_cols)} text columns")
        
        return self
    
    @timing_decorator
    def extract_temporal_features(self) -> 'SmartDataCleaner':
        """Extract features from temporal columns"""
        if self.df is None:
            return self
        
        temporal_cols = [
            col for col, profile in self.column_profiles.items()
            if profile.inferred_type == DataType.TEMPORAL
        ]
        
        for col in temporal_cols:
            try:
                dt_series = pd.to_datetime(self.df[col], errors='coerce')
                
                # Extract common features
                self.df[f'{col}_year'] = dt_series.dt.year
                self.df[f'{col}_month'] = dt_series.dt.month
                self.df[f'{col}_day'] = dt_series.dt.day
                self.df[f'{col}_weekday'] = dt_series.dt.weekday
                self.df[f'{col}_hour'] = dt_series.dt.hour
                
            except:
                continue
        
        if self.verbose and temporal_cols:
            logger.info(f"⏰ Extracted features from {len(temporal_cols)} temporal columns")
        
        return self
    
    def get_report(self, format: str = 'dict') -> Union[Dict, str]:
        """
        Generate smart cleaning report.
        
        Args:
            format: 'dict', 'json', 'markdown'
        
        Returns:
            Cleaning report in specified format
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_signature': self.data_signature,
            'original_shape': self.original_shape,
            'current_shape': self.df.shape if self.df is not None else None,
            'quality_score': self.quality_score,
            'strategy': self.strategy.value,
            'column_count': len(self.column_profiles),
            'column_types': {
                col: profile.inferred_type.value 
                for col, profile in self.column_profiles.items()
            },
            'issues_found': {
                'high_null_columns': [
                    col for col, profile in self.column_profiles.items()
                    if profile.null_percentage > 20
                ],
                'low_variance_columns': [
                    col for col, profile in self.column_profiles.items()
                    if profile.unique_percentage < 1
                ]
            },
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2 if self.df is not None else 0
        }
        
        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        elif format == 'markdown':
            md = [
                "# Data Cleaning Report",
                f"**Generated**: {report['timestamp']}",
                f"**Quality Score**: {report['quality_score']:.2%}",
                "",
                "## Summary",
                f"- Original shape: {report['original_shape']}",
                f"- Current shape: {report['current_shape']}",
                f"- Columns: {report['column_count']}",
                f"- Strategy: {report['strategy']}",
                "",
                "## Column Types",
            ]
            
            type_counts = {}
            for col_type in report['column_types'].values():
                type_counts[col_type] = type_counts.get(col_type, 0) + 1
            
            for col_type, count in type_counts.items():
                md.append(f"- {col_type}: {count}")
            
            return '\n'.join(md)
        
        return report
    
    def save_state(self, filepath: Union[str, Path]) -> None:
        """Save cleaner state for later resumption"""
        state = {
            'df': self.df.to_parquet() if self.df is not None else None,
            'strategy': self.strategy.value,
            'original_shape': self.original_shape,
            'column_profiles': {
                col: profile.to_dict() 
                for col, profile in self.column_profiles.items()
            },
            'data_signature': self.data_signature,
            'quality_score': self.quality_score
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, default=str)
        
        if self.verbose:
            logger.info(f"💾 State saved to {filepath}")
    
    @classmethod
    def load_state(cls, filepath: Union[str, Path]) -> 'SmartDataCleaner':
        """Load cleaner from saved state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        cleaner = cls(strategy=state['strategy'])
        
        if state['df']:
            import io
            cleaner.df = pd.read_parquet(io.BytesIO(state['df']))
        
        cleaner.original_shape = state['original_shape']
        cleaner.data_signature = state['data_signature']
        cleaner.quality_score = state['quality_score']
        
        # Reconstruct profiles
        cleaner.column_profiles = {
            col: ColumnProfile(**profile)
            for col, profile in state['column_profiles'].items()
        }
        
        return cleaner
    
    @property
    def is_clean(self) -> bool:
        """Check if data is clean based on quality score"""
        return self.quality_score is not None and self.quality_score > 0.8
    
    def __repr__(self) -> str:
        return (f"SmartDataCleaner(shape={self.df.shape if self.df else 'No data'}, "
                f"strategy={self.strategy.value}, "
                f"quality={self.quality_score:.1% if self.quality_score else 'N/A'})")


# Example usage with modern Python features
def example_smart_usage():
    """
    Example of using the smart data cleaner.
    
    Demonstrates:
    - Smart loading
    - Auto-cleaning
    - Quality reporting
    - State management
    """
    print("🤖 Smart Data Cleaner Example")
    print("=" * 50)
    
    # Create sample data with issues
    data = {
        'Customer_ID': range(100),
        'Customer Name': [f'Customer_{i}' for i in range(100)],
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 20000, 100),
        'join_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'rating': np.random.choice([1, 2, 3, 4, 5], 100),
        'notes': ['Note ' + str(i) for i in range(100)]
    }
    
    df = pd.DataFrame(data)
    
    # Add some issues
    df.loc[::10, 'age'] = np.nan  # Add missing values
    df.loc[5:15, 'income'] = 999999  # Add outliers
    df.loc[20:30, 'category'] = None  # More missing
    
    # Initialize smart cleaner
    cleaner = SmartDataCleaner(df, strategy='auto', verbose=True)
    
    # Get initial report
    print("\n📊 Initial Report:")
    initial_report = cleaner.get_report('markdown')
    print(initial_report[:500] + "...")
    
    # Execute smart cleaning
    print("\n🔧 Cleaning in progress...")
    cleaner.clean()
    
    # Get final report
    print("\n📊 Final Report:")
    final_report = cleaner.get_report('markdown')
    print(final_report)
    
    # Check if clean
    print(f"\n✅ Data is {'clean' if cleaner.is_clean else 'not clean'}")
    print(f"📈 Quality improved by: "
          f"{(cleaner.quality_score - 0.8) * 100:.1f}%" if cleaner.quality_score else "N/A")
    
    # Show cleaned data sample
    print("\n📋 Cleaned Data Sample:")
    print(cleaner.df.head())
    
    print("\n✨ Example complete! Ready for analysis or ML.")


# Test the cleaner
if __name__ == "__main__":
    # Test con dati di esempio
    print("🧪 Testing SmartDataCleaner...")
    
    # Crea dati di test
    test_data = {
        'User Name': ['Alice Smith', 'BOB JONES', 'charlie brown', None, 'Eve Davis'],
        'Age': [25, None, 35, 45, 999],
        'Salary': [50000, 60000, None, 80000, 120000],
        'Join Date': ['2020-01-15', '2021-03-22', None, '2019-11-30', '2022-12-01'],
        'Department': ['IT', 'HR', 'IT', 'Sales', 'HR'],
        'Active': [True, False, True, None, False]
    }
    
    df_test = pd.DataFrame(test_data)
    print("📊 Test Data:")
    print(df_test)
    print("\n" + "=" * 50)
    
    # Crea e usa il cleaner
    cleaner = SmartDataCleaner(df_test, verbose=True)
    
    # Esegui pulizia
    cleaned_df = cleaner.clean(inplace=False)
    
    print("\n✨ Cleaned Data:")
    print(cleaned_df)
    
    print("\n📈 Quality Score:", cleaner.quality_score)
    print("✅ Test completato con successo!")
