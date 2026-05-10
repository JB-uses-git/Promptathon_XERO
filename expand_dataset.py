"""
Dataset Expansion Pipeline: Telco 7K -> 50K
Uses SMOTE for minority class + Gaussian noise injection for majority class
to create a high-fidelity 50K dataset from the original 7K Telco churn data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings('ignore')

def expand_dataset(target_size=50000):
    print("=" * 70)
    print("  DATASET EXPANSION: Telco 7K -> 50K")
    print("=" * 70)
    
    # 1. Load original
    df = pd.read_csv("Telco_churn.csv")
    print(f"\n[1/5] Original dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
    
    # 2. Clean data
    print("\n[2/5] Cleaning data...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    # Save original IDs and categorical mappings
    original_ids = df['customerID'].values
    
    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns.drop(['customerID', 'Churn'])
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Encode target
    y = (df['Churn'] == 'Yes').astype(int).values
    X = df.drop(columns=['customerID', 'Churn'])
    feature_cols = X.columns.tolist()
    X = X.values.astype(np.float64)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Numeric range check — tenure: [{df['tenure'].min()}, {df['tenure'].max()}]")
    
    # 3. SMOTE to balance + expand
    print(f"\n[3/5] Expanding with SMOTE + noise injection...")
    
    # We want ~50k total. SMOTE creates synthetic minority samples.
    # Strategy: first SMOTE to balance, then replicate with noise to reach 50k
    
    # Step A: SMOTE to balance classes
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"  After SMOTE balance: {len(X_balanced)} rows (churn rate: {y_balanced.mean()*100:.1f}%)")
    
    # Step B: Replicate with Gaussian noise to reach target size
    n_current = len(X_balanced)
    n_needed = target_size - n_current
    
    if n_needed > 0:
        # Sample from balanced data and add noise
        rng = np.random.RandomState(42)
        indices = rng.choice(n_current, size=n_needed, replace=True)
        X_extra = X_balanced[indices].copy()
        y_extra = y_balanced[indices].copy()
        
        # Add Gaussian noise to continuous columns only
        # tenure(idx 4), MonthlyCharges(idx 17), TotalCharges(idx 18)
        continuous_idx = [4, 17, 18]  # tenure, MonthlyCharges, TotalCharges
        for ci in continuous_idx:
            col_std = X_balanced[:, ci].std()
            noise = rng.normal(0, col_std * 0.05, size=n_needed)  # 5% noise
            X_extra[:, ci] += noise
        
        # Clip to valid ranges
        X_extra[:, 4] = np.clip(X_extra[:, 4], 0, 72).astype(int)  # tenure: 0-72
        X_extra[:, 17] = np.clip(X_extra[:, 17], 18.0, 120.0)      # MonthlyCharges
        X_extra[:, 18] = np.clip(X_extra[:, 18], 0, 9000)          # TotalCharges
        
        # For categorical columns, randomly flip ~3% of values
        cat_idx = [i for i in range(X_extra.shape[1]) if i not in continuous_idx]
        for ci in cat_idx:
            unique_vals = np.unique(X_balanced[:, ci])
            if len(unique_vals) > 1:
                flip_mask = rng.random(n_needed) < 0.03
                X_extra[flip_mask, ci] = rng.choice(unique_vals, size=flip_mask.sum())
        
        X_final = np.vstack([X_balanced, X_extra])
        y_final = np.concatenate([y_balanced, y_extra])
    else:
        X_final = X_balanced
        y_final = y_balanced
    
    print(f"  After noise expansion: {len(X_final)} rows")
    
    # 4. Reconstruct DataFrame
    print(f"\n[4/5] Reconstructing dataset...")
    expanded_df = pd.DataFrame(X_final, columns=feature_cols)
    
    # Decode categoricals back to original labels
    for col in cat_cols:
        le = label_encoders[col]
        # Round encoded values and clip to valid range
        vals = np.clip(np.round(expanded_df[col].values).astype(int), 0, len(le.classes_) - 1)
        expanded_df[col] = le.inverse_transform(vals)
    
    # Fix numeric types
    expanded_df['SeniorCitizen'] = expanded_df['SeniorCitizen'].round().astype(int).clip(0, 1)
    expanded_df['tenure'] = expanded_df['tenure'].round().astype(int).clip(0, 72)
    expanded_df['MonthlyCharges'] = expanded_df['MonthlyCharges'].round(2)
    expanded_df['TotalCharges'] = expanded_df['TotalCharges'].round(2)
    
    # Add Churn column and customer IDs
    expanded_df['Churn'] = np.where(y_final == 1, 'Yes', 'No')
    expanded_df['customerID'] = [f"EXP-{i:06d}" for i in range(len(expanded_df))]
    
    # Reorder columns to match original
    col_order = ['customerID'] + feature_cols + ['Churn']
    expanded_df = expanded_df[col_order]
    
    # 5. Save
    out_path = "Telco_churn_50k.csv"
    expanded_df.to_csv(out_path, index=False)
    
    print(f"\n[5/5] Final dataset stats:")
    print(f"  Shape: {expanded_df.shape}")
    print(f"  Churn rate: {(expanded_df['Churn']=='Yes').mean()*100:.1f}%")
    print(f"  Tenure range: {expanded_df['tenure'].min()}-{expanded_df['tenure'].max()}")
    print(f"  MonthlyCharges range: ${expanded_df['MonthlyCharges'].min():.2f}-${expanded_df['MonthlyCharges'].max():.2f}")
    print(f"  Saved to: {out_path}")
    
    return expanded_df


if __name__ == "__main__":
    expand_dataset()
