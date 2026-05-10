import pandas as pd
import sys

def run_eda(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print("\n--- Basic Info ---")
    df.info()
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Target Variable Distribution (churn) ---")
    print(df['churn'].value_counts(normalize=True))
    
    print("\n--- Descriptive Statistics (Numeric) ---")
    print(df.describe().T)
    
    print("\n--- Descriptive Statistics (Categorical) ---")
    print(df.describe(include=['O']).T)
    
if __name__ == '__main__':
    run_eda('d:/AMC/telecom_churn.csv')
