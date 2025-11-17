## 1. Setup and Data Loading
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Load the training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Keep the test IDs for the final submission file
test_ids = test_df['Id']

# --- Guideline from Video: Log Transform the Target Variable ---
# This helps normalize the skewed sale price distribution.
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# Separate target variable from features
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# Combine train and test data for consistent preprocessing
all_data = pd.concat([X, test_df], ignore_index=True)
# Drop Id column as it's not a feature
all_data = all_data.drop('Id', axis=1)

print("Combined data for preprocessing.")
print(f"Shape: {all_data.shape}")

## 2. Bare Minimum Data Preparation

# --- Step 2.1: Impute Missing Numerical Values ---
# We'll fill any missing numbers with the median of their respective column.
numerical_cols = all_data.select_dtypes(include=np.number).columns
for col in numerical_cols:
    median_val = all_data[col].median()
    all_data[col] = all_data[col].fillna(median_val)

print("Missing numerical values imputed with the median.")

# --- Step 2.2: One-Hot Encode Categorical Features ---
# This is the simplest way to convert all text-based columns into numbers the model can use.
all_data = pd.get_dummies(all_data, drop_first=True)

print("Categorical features converted to numerical using one-hot encoding.")

# --- Step 2.3: Separate back into Training and Test Sets ---
X_processed = all_data.iloc[:len(train_df)]
X_test_processed = all_data.iloc[len(train_df):]

print("\nData preparation complete.")
print(f"Final training features shape: {X_processed.shape}")
print(f"Final test features shape: {X_test_processed.shape}")

## 3. Model Training and Prediction (Scaling is NOT needed for XGBoost)

# --- Step 3.1: Train the XGBoost Regressor Model ---
# We'll use some standard baseline parameters for XGBoost.
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)

# Train the model on the entire prepared training dataset
model.fit(X_processed, y)

print("\nXGBoost model trained successfully.")

# --- Step 3.2: Make Predictions on the Test Set ---
predictions_log = model.predict(X_test_processed)

# --- Step 3.3: Inverse Transform Predictions ---
# We must convert the predictions back from the log scale to the original dollar scale.
final_predictions = np.expm1(predictions_log)

print("Predictions generated and converted back to original scale.")

## 4. Create Submission File

# Create a DataFrame for the submission
submission_df = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_predictions
})

# Save the DataFrame to a .csv file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' has been created successfully! 🚀")
print("Top 5 rows of the submission file:")
print(submission_df.head())