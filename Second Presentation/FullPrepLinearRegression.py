## 1. Setup and Data Loading
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

# Load the training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

## 2. Complete "Smarter" Data Preparation

# --- Step 2.1: Remove Hardcoded Outliers ---
outlier_ids = [935, 1299, 253, 314, 336, 707, 1183, 379, 496, 636, 441, 524, 692, 186, 598, 955, 49, 810, 198]
train_df = train_df[~train_df['Id'].isin(outlier_ids)]
print(f"Training data shape after removing outliers: {train_df.shape}")

# Keep test IDs and log transform the target
test_ids = test_df['Id']
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# Combine train and test data for preprocessing
all_data = pd.concat([train_df.drop('SalePrice', axis=1), test_df], ignore_index=True)
all_data = all_data.drop('Id', axis=1)

# --- Step 2.2: ADDED FEATURE - Feature Engineering ---
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBaths'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
print("\nEngineered new features: TotalSF, TotalBaths, HouseAge.")

# --- Step 2.3: Log Transform Skewed Numerical Features ---
numerical_cols = all_data.select_dtypes(include=np.number).columns
skewness = all_data[numerical_cols].apply(lambda x: skew(x.dropna()))
highly_skewed = skewness[skewness > 0.75].index
all_data[highly_skewed] = np.log1p(all_data[highly_skewed])
print(f"Applied log transform to {len(highly_skewed)} skewed numerical features.")

# --- Step 2.4: Impute and Encode ---
all_data = pd.get_dummies(all_data, drop_first=True)
all_data = all_data.fillna(all_data.median())

# --- Step 2.5: Separate back into Training and Test Sets ---
X_processed = all_data.iloc[:len(train_df)]
X_test_processed = all_data.iloc[len(train_df):]
print(f"\nData preparation v2 (full) complete.")

## 3. Feature Scaling
# Scaling is important for regularized models like Ridge.
scaler = StandardScaler()
scaler.fit(X_processed)
X_scaled = scaler.transform(X_processed)
X_test_scaled = scaler.transform(X_test_processed)
print("Features scaled using StandardScaler.")

## 4. Model Training and Prediction
model = Ridge(alpha=15, random_state=42)
y = train_df['SalePrice']
model.fit(X_scaled, y)
print("\nRidge model trained successfully on v2 (full) prepared data.")

predictions_log = model.predict(X_test_scaled)
final_predictions = np.expm1(predictions_log)

## 5. Create Submission File
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
submission_df.to_csv('submission_ridge_v2_full_prep.csv', index=False)
print("\nSubmission file 'submission_ridge_v2_full_prep.csv' created.")