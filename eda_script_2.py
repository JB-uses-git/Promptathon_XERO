import pandas as pd

def run_eda(filepath):
    df = pd.read_csv(filepath)
    print("--- Min/Max/Negatives ---")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        neg_count = (df[col] < 0).sum()
        print(f"{col} - Min: {df[col].min()}, Max: {df[col].max()}, Negatives: {neg_count}")

if __name__ == '__main__':
    run_eda('d:/AMC/telecom_churn.csv')
