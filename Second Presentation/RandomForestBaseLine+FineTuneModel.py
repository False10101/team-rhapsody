import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# =============================================================================
# STEP 0: INITIAL SETUP & TRAIN-VALIDATION SPLIT
# =============================================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

test_ids = test_df['Id']

train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 0A: Apply log transform to the target variable (SalePrice)
y_train = np.log1p(y_train)
y_val = np.log1p(y_val)

X_train = X_train.copy()
X_val = X_val.copy()
test_df_processed = test_df.copy()

# =============================================================================
# STEP 0.5: FEATURE ENGINEERING (FROM VIDEO)
# =============================================================================
for df in [X_train, X_val, test_df_processed]:
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Update column lists after creating new features
NUMERICAL_COLS = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'HalfBath', 'BedroomAbvGrd', 'KitchenAbvGrd', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea', 'MiscVal', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'HouseAge', 'TotalSF'
]
ORDINAL_COLS = [
    'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
    'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'
]
NOMINAL_COLS = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'GarageType', 'MiscFeature', 'MoSold',
    'YrSold', 'SaleType', 'SaleCondition'
]

# =============================================================================
# STEP 1 & 2: IMPUTE AND MAP NUMERICAL/ORDINAL FEATURES
# =============================================================================
for col in NUMERICAL_COLS:
    if col in X_train.columns:
        median_val = X_train[col].median()
        for df in [X_train, X_val, test_df_processed]:
            df[col] = df[col].fillna(median_val)

ordinal_mappings = {
    'LotShape':     {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0}, 'Utilities':    {'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0},
    'LandSlope':    {'Gtl': 2, 'Mod': 1, 'Sev': 0}, 'ExterQual':    {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'ExterCond':    {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}, 'BsmtQual':     {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'BsmtCond':     {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}, 'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0},
    'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}, 'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0},
    'HeatingQC':    {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}, 'CentralAir':   {'Y': 1, 'N': 0},
    'Electrical':   {'SBrkr': 4, 'FuseA': 3, 'FuseF': 2, 'FuseP': 1, 'Mix': 0}, 'KitchenQual':  {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'Functional':   {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0, 'NA': 0}, 'FireplaceQu':  {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}, 'GarageQual':   {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'GarageCond':   {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}, 'PavedDrive':   {'Y': 2, 'P': 1, 'N': 0},
    'PoolQC':       {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0}, 'Fence':        {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0},
}
for col in ORDINAL_COLS:
    if col in X_train.columns:
        if X_train[col].dtype == 'object':
            for df in [X_train, X_val, test_df_processed]:
                df[col] = df[col].fillna('NA').str.strip()
                df[col] = df[col].map(ordinal_mappings.get(col, {})).fillna(0)
        else:
            median_val = X_train[col].median()
            for df in [X_train, X_val, test_df_processed]:
                df[col] = df[col].fillna(median_val)

# =============================================================================
# STEP 3: PROCESS NOMINAL FEATURES
# =============================================================================
for col in NOMINAL_COLS:
    if col in X_train.columns and X_train[col].dtype == 'object':
        mode_val = X_train[col].mode()[0]
        for df in [X_train, X_val, test_df_processed]:
            df[col] = df[col].fillna(mode_val)

# Encode nominal features into simple integers for the tree model
print("Applying integer encoding for tree-based models...")
for col in NOMINAL_COLS:
    if col in X_train.columns and X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype('category')
        known_categories = X_train[col].cat.categories
        X_val[col] = pd.Categorical(X_val[col], categories=known_categories, ordered=False)
        test_df_processed[col] = pd.Categorical(test_df_processed[col], categories=known_categories, ordered=False)
        X_train[col] = X_train[col].cat.codes
        X_val[col] = X_val[col].cat.codes
        test_df_processed[col] = test_df_processed[col].cat.codes
        X_val[col] = X_val[col].replace(-1, X_train[col].mode()[0])
        test_df_processed[col] = test_df_processed[col].replace(-1, X_train[col].mode()[0])

# =============================================================================
# STEP 4: FINAL CLEANUP
# =============================================================================
# Drop sparse columns and cap outliers
threshold = 0.6
non_null_ratio = X_train.notnull().mean()
cols_to_drop = non_null_ratio[non_null_ratio < threshold].index.tolist()
for df in [X_train, X_val, test_df_processed]:
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

tmp = X_train.copy()
tmp["SalePrice"] = y_train
corr = tmp.corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False)
top_corr_cols = corr.index[1:21]

for col in top_corr_cols:
    if col in X_train.columns:
      lower_cap = X_train[col].quantile(0.01)
      upper_cap = X_train[col].quantile(0.99)
      for df in [X_train, X_val, test_df_processed]:
          df[col] = np.clip(df[col], lower_cap, upper_cap)

print("\n--- Data prep for Tree Models complete ---")
print(f"Final training shape: {X_train.shape}")
print(f"Final validation shape: {X_val.shape}")
print(f"Final testing shape:  {test_df_processed.shape}")
print("\nFinal Processed Training Data Head:")
print(X_train.head())



# Use the same baseline code to compare against
print("--- Training Standard Random Forest (Baseline) ---")
rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)
train_rmse_rf = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(y_train_pred_rf)))
val_rmse_rf = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_val_pred_rf)))
print(f"Validation RMSE: ${val_rmse_rf:,.2f}")
print("-" * 50)


# =============================================================================
# MODEL 2: RANDOM FOREST (IMPROVED FINE-TUNING)
# =============================================================================
print("\n--- Running an Improved Fine-Tuning Search for Random Forest ---")

# A wider, more strategic parameter grid to explore
# We're including the default-like options (max_depth=None, min_samples_leaf=1)
param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Set up GridSearchCV with the new grid
rf_regressor = RandomForestRegressor(n_jobs=-1, random_state=42)
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, # Using cv=3 to speed it up
                           scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model
best_rf_model = grid_search.best_estimator_
print(f"\nBest fine-tuning parameters found: {grid_search.best_params_}")

# Make predictions with the new best model
y_train_pred_rf_tuned = best_rf_model.predict(X_train)
y_val_pred_rf_tuned = best_rf_model.predict(X_val)

# --- Calculate RMSE for the new fine-tuned model ---
train_rmse_rf_tuned = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(y_train_pred_rf_tuned)))
val_rmse_rf_tuned = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_val_pred_rf_tuned)))

print(f"\nNew Fine-Tuned Train RMSE: ${train_rmse_rf_tuned:,.2f}")
print(f"New Fine-Tuned Validation RMSE: ${val_rmse_rf_tuned:,.2f}")
print("-" * 50)