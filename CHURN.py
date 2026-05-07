# data_ingestion.py
import pandas as pd

def load_raw_data(filepath):
    df = pd.read_csv(filepath)
    print(df.info())
    print(df.describe())
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    return df

if __name__ == "__main__":
    df = load_raw_data(r"D:\AI CHURN\data_ecommerce_customer_churn.csv")