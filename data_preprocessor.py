"""
data_preprocessor.py - Clean and prepare data for analytics/ML
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

class DataPreprocessor:
    """Clean and preprocess data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_processed = None
        self.missing_report = {}
        self.categorical_encodings = {}
    
    def handle_missing_values(self, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values - median for numerical"""
        
        print(f"\n🔧 HANDLING MISSING VALUES (strategy: {strategy})")
        print("="*60)
        
        # Get columns with missing values
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        
        if not missing_cols:
            print("✓ No missing values found")
            return self.df
        
        print(f"Found missing values in {len(missing_cols)} columns:")
        
        for col in missing_cols:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            print(f"\n  {col}: {missing_count} ({missing_pct:.1f}%)")
            
            if self.df[col].dtype in ['float64', 'int64']:
                # Numerical: use median or mean
                if strategy == 'median':
                    fill_value = self.df[col].median()
                    print(f"    → Filling with median: {fill_value:.2f}")
                else:
                    fill_value = self.df[col].mean()
                    print(f"    → Filling with mean: {fill_value:.2f}")
                
                self.df[col].fillna(fill_value, inplace=True)
                self.missing_report[col] = {'strategy': strategy, 'value': fill_value}
            
            else:
                # Categorical: use mode
                fill_value = self.df[col].mode()[0]
                print(f"    → Filling with mode: {fill_value}")
                self.df[col].fillna(fill_value, inplace=True)
                self.missing_report[col] = {'strategy': 'mode', 'value': fill_value}
        
        # Verify
        remaining_missing = self.df.isnull().sum().sum()
        print(f"\n✓ Missing values handled. Remaining: {remaining_missing}")
        print("="*60)
        
        return self.df
    
    def encode_categorical(self) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables for ML"""
        
        print(f"\n🔤 ENCODING CATEGORICAL VARIABLES")
        print("="*60)
        
        categorical_cols = self.df.select_dtypes(include='object').columns.tolist()
        
        if not categorical_cols:
            print("✓ No categorical columns found")
            return self.df, {}
        
        print(f"Found {len(categorical_cols)} categorical columns:\n")
        
        df_encoded = self.df.copy()
        
        for col in categorical_cols:
            unique_values = df_encoded[col].unique()
            n_unique = len(unique_values)
            
            # Create encoding
            encoding = {val: idx for idx, val in enumerate(unique_values)}
            df_encoded[col] = df_encoded[col].map(encoding)
            
            self.categorical_encodings[col] = encoding
            
            print(f"  {col}:")
            print(f"    Unique values: {n_unique}")
            for val, code in sorted(encoding.items(), key=lambda x: x[1])[:5]:
                print(f"      {val} → {code}")
            if n_unique > 5:
                print(f"      ... and {n_unique - 5} more")
        
        self.df_processed = df_encoded
        print(f"\n✓ Encoding complete")
        print("="*60)
        
        return df_encoded, self.categorical_encodings
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get cleaned and processed data"""
        return self.df_processed if self.df_processed is not None else self.df
    
    def get_statistics(self) -> Dict:
        """Get descriptive statistics"""
        
        return {
            'mean': self.df.describe().loc['mean'].to_dict(),
            'std': self.df.describe().loc['std'].to_dict(),
            'min': self.df.describe().loc['min'].to_dict(),
            'max': self.df.describe().loc['max'].to_dict(),
        }
    
    def print_statistics(self):
        """Print formatted statistics"""
        
        print("\n" + "="*60)
        print("📈 DESCRIPTIVE STATISTICS")
        print("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        stats_df = self.df[numerical_cols].describe().T
        
        for col in stats_df.index:
            print(f"\n{col}:")
            print(f"  Mean: {stats_df.loc[col, 'mean']:.2f}")
            print(f"  Std:  {stats_df.loc[col, 'std']:.2f}")
            print(f"  Min:  {stats_df.loc[col, 'min']:.2f}")
            print(f"  Max:  {stats_df.loc[col, 'max']:.2f}")
        
        print("\n" + "="*60 + "\n")