"""
Phase 1: Data Exploration & Understanding
Case Study: Algorithmic Fairness & Risk Assessment in High-Stakes Financial Lending
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load main datasets
print("Loading datasets...")
train = pd.read_csv('dataset/home-credit-default-risk/application_train.csv')
test = pd.read_csv('dataset/home-credit-default-risk/application_test.csv')
bureau = pd.read_csv('dataset/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('dataset/home-credit-default-risk/bureau_balance.csv')
columns_desc = pd.read_csv('dataset/home-credit-default-risk/HomeCredit_columns_description.csv', encoding='latin-1')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Bureau shape: {bureau.shape}")
print(f"Bureau Balance shape: {bureau_balance.shape}")

# Target distribution
print("\n=== TARGET DISTRIBUTION ===")
print(train['TARGET'].value_counts())
print(f"Default rate: {train['TARGET'].mean():.4f} ({train['TARGET'].mean()*100:.2f}%)")

# Data types
print("\n=== DATA TYPES ===")
print(train.dtypes.value_counts())

# Missing values analysis
print("\n=== MISSING VALUES (Top 20) ===")
missing = train.isnull().sum()
missing_pct = (missing / len(train)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Pct': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Pct', ascending=False)
print(missing_df.head(20))

# Categorical vs Numerical
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove('TARGET')
num_cols.remove('SK_ID_CURR')

print(f"\nCategorical columns: {len(cat_cols)}")
print(f"Numerical columns: {len(num_cols)}")

# Save exploration results
missing_df.to_csv('missing_analysis.csv')
train.describe().to_csv('numerical_summary.csv')

print("\nPhase 1 Complete - Data exploration saved")