import pandas as pd
import os

class DataLoader:
    """Load and validate e-commerce churn data"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.original_shape = None
    
    def load(self) -> pd.DataFrame:
        """Load CSV and validate structure"""
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found: {self.filepath}")
        
        print(f"📁 Loading data from: {self.filepath}")
        
        self.df = pd.read_csv(self.filepath)
        self.original_shape = self.df.shape
        
        print(f"✓ Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        print(f"✓ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        self._validate()
        return self.df
    
    def _validate(self):
        """Check data integrity"""
        
        if 'Churn' not in self.df.columns:
            raise ValueError("❌ 'Churn' column missing from dataset")
        
        if self.df['Churn'].isnull().sum() > 0:
            raise ValueError("❌ 'Churn' column has null values")
        
        unique_churn = self.df['Churn'].unique()
        if not set(unique_churn).issubset({0, 1, 'Yes', 'No'}):
            raise ValueError(f"❌ Invalid churn values: {unique_churn}")
        
        print("✓ Data validation passed")
    
    def get_summary(self) -> dict:
        """Return data summary"""
        
        if pd.api.types.is_numeric_dtype(self.df['Churn']):
            churn_count = (self.df['Churn'] == 1).sum()
        else:
            churn_count = (self.df['Churn'] == 'Yes').sum()
        
        churn_rate = churn_count / len(self.df)
        
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'churn_count': int(churn_count),
            'churn_rate': float(churn_rate),
            'missing_values': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.to_dict()
        }
    
    def print_summary(self):
        """Print formatted data summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 DATA SUMMARY")
        print("="*60)
        print(f"Total Customers: {summary['total_rows']:,}")
        print(f"Total Features: {summary['total_columns']}")
        print(f"Churned: {summary['churn_count']:,} ({summary['churn_rate']:.1%})")
        print(f"Active: {summary['total_rows'] - summary['churn_count']:,} ({1-summary['churn_rate']:.1%})")
        
        missing = {k: v for k, v in summary['missing_values'].items() if v > 0}
        if missing:
            print("\nMissing Values:")
            for col, count in missing.items():
                pct = (count / summary['total_rows']) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("\n✓ No missing values")
        
        print("="*60 + "\n")


# ✅ Proper execution block (outside class)
if __name__ == "__main__":
    filepath = r"D:\Projects\AI CHURN\data_ecommerce_customer_churn.csv"
    
    loader = DataLoader(filepath)
    df = loader.load()
    loader.print_summary()