import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_customer_churn_100k.csv')
df['churn_bin'] = (df['Churn'] == 'Yes').astype(int)

print("=== Churn by Tenure bins ===")
df['tenure_bin'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 36, 48, 60, 72])
print(df.groupby('tenure_bin')['churn_bin'].mean())

print("\n=== Churn by MonthlyCharges bins ===")
df['mc_bin'] = pd.cut(df['MonthlyCharges'], bins=[0, 30, 60, 90, 120, 150])
print(df.groupby('mc_bin')['churn_bin'].mean())

print("\n=== Churn by Age bins ===")
df['age_bin'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70, 80])
print(df.groupby('age_bin')['churn_bin'].mean())

print("\n=== Correlations with churn ===")
num_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
for c in num_cols:
    print(f"  {c}: {df[c].corr(df['churn_bin']):.4f}")

print("\n=== Contract x MonthlyCharges interaction ===")
for contract in df['Contract'].unique():
    subset = df[df['Contract'] == contract]
    print(f"  {contract}: churn_rate={subset['churn_bin'].mean():.4f}, "
          f"avg_monthly={subset['MonthlyCharges'].mean():.1f}, "
          f"avg_tenure={subset['Tenure'].mean():.1f}")

print("\n=== TotalCharges consistency check ===")
df['expected_total'] = df['Tenure'] * df['MonthlyCharges']
df['total_diff'] = df['TotalCharges'] - df['expected_total']
print(f"  Mean diff: {df['total_diff'].mean():.2f}")
print(f"  Std diff:  {df['total_diff'].std():.2f}")
print(f"  Neg TotalCharges: {(df['TotalCharges'] < 0).sum()}")
