## 1. Setup and Data Loading
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Load the training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Keep test IDs and log transform the target
test_ids = test_df['Id']
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# Separate target and combine for preprocessing
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
all_data = pd.concat([X, test_df], ignore_index=True)
all_data = all_data.drop('Id', axis=1)

# --- SMARTER STEP: Feature Engineering ---
# Based on our results, we are adding this directly to the baseline prep.
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBaths'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
print("\nEngineered new features: TotalSF, TotalBaths, HouseAge.")


## 2. Bare Minimum Data Preparation (Applied to engineered data)
numerical_cols = all_data.select_dtypes(include=np.number).columns
for col in numerical_cols:
    median_val = all_data[col].median()
    all_data[col] = all_data[col].fillna(median_val)

all_data = pd.get_dummies(all_data, drop_first=True)

X_processed = all_data.iloc[:len(train_df)]
X_test_processed = all_data.iloc[len(train_df):]
print(f"\nData preparation v2 (full) complete.")


## 3. Model Training and Prediction
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X_processed, y)
print("\nXGBoost model trained on v2 (full) prepared data.")

predictions_log = model.predict(X_test_processed)
final_predictions = np.expm1(predictions_log)

## 4. Create Submission File
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
submission_df.to_csv('submission_xgboost_v2_full_prep.csv', index=False)
print("\nSubmission file 'submission_xgboost_v2_full_prep.csv' created.")