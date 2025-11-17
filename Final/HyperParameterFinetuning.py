# --- 1. LOAD LIBRARIES ---
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna # For hyperparameter tuning
import warnings # Import warnings

# --- 2. LOAD DATA ---
# (This section is identical to your script)
warnings.filterwarnings('ignore')
print("Loading data...")
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print ("Data is loaded!")
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# --- 3. DATA PROCESSING AND FEATURE ENGINEERING ---
# (This section is identical to your script)
print("Starting data processing...")
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)

# Convert numerical categories to strings
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

# Fill NaNs
features['Functional'] = features['Functional'].fillna('Typ')
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['KitchenQual'] = features['KitchenQual'].fillna("TA")
features["PoolQC"] = features["PoolQC"].fillna("None")
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features['MasVnrType'] = features['MasVnrType'].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'):
    features[col] = features[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    features[col] = features[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
objects = [i for i in features.columns if features[i].dtype == object]
features.update(features[objects].fillna('None'))
numerics = [i for i in features.columns if features[i].dtype != object]
features.update(features[numerics].fillna(0))

# --- SKEWNESS TRANSFORM ---
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = [i for i in features.columns if features[i].dtype in numeric_dtypes]
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
for i in skew_index:
    if len(features[i].unique()) == 1: continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lambda_val = boxcox_normmax(features[i] + 1)
            features[i] = boxcox1p(features[i], lambda_val)
        except Exception:
            features[i] = np.log1p(features[i])

# --- FEATURE ENGINEERING ---
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] + features['1stFlrSF'] + features['2ndFlrSF'])
features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) + features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))
features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] + features['EnclosedPorch'] + features['ScreenPorch'] + features['WoodDeckSF'])
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# --- FINAL PREP ---
final_features = pd.get_dummies(features).reset_index(drop=True)
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94: overfit.append(i)
overfit = list(overfit)
if 'MSZoning_C (all)' in X.columns: overfit.append('MSZoning_C (all)')
X = X.drop(overfit, axis=1)
X_sub = X_sub.drop(overfit, axis=1)
print(f"Final training data shape: {X.shape}")
print("Data processing finished.")

# --- 4. HYPERPARAMETER TUNING ---

# Setup K-Folds (consistent for all models)
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# --- 4A: GridSearchCV (Linear Models & SVR) ---
print("\n--- Starting GridSearchCV for Linear Models & SVR ---")

# Note: We use the base models (Ridge, Lasso) NOT RidgeCV, LassoCV
# GridSearchCV will do the work of finding the best alpha.

# 1. Lasso
pipe_lasso = make_pipeline(RobustScaler(), Lasso(max_iter=10000000, random_state=42))
param_lasso = {
    'lasso__alpha': [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
}
grid_lasso = GridSearchCV(pipe_lasso, param_lasso, cv=kfolds, scoring='neg_mean_squared_error', n_jobs=-1)
grid_lasso.fit(X, y)
rmse_lasso = np.sqrt(-grid_lasso.best_score_)
print(f"Lasso Tuned! Best RMSE: {rmse_lasso:.6f}")
print(f"Best Params: {grid_lasso.best_params_}\n")

# 2. Ridge
pipe_ridge = make_pipeline(RobustScaler(), Ridge(random_state=42))
param_ridge = {
    'ridge__alpha': [14.5, 14.8, 15, 15.2, 15.5]
}
grid_ridge = GridSearchCV(pipe_ridge, param_ridge, cv=kfolds, scoring='neg_mean_squared_error', n_jobs=-1)
grid_ridge.fit(X, y)
rmse_ridge = np.sqrt(-grid_ridge.best_score_)
print(f"Ridge Tuned! Best RMSE: {rmse_ridge:.6f}")
print(f"Best Params: {grid_ridge.best_params_}\n")

# 3. SVR
pipe_svr = make_pipeline(RobustScaler(), SVR())
param_svr = {
    'svr__C': [15, 20, 25],
    'svr__epsilon': [0.007, 0.008, 0.009],
    'svr__gamma': [0.0002, 0.0003, 0.0004]
}
grid_svr = GridSearchCV(pipe_svr, param_svr, cv=kfolds, scoring='neg_mean_squared_error', n_jobs=-1)
grid_svr.fit(X, y)
rmse_svr = np.sqrt(-grid_svr.best_score_)
print(f"SVR Tuned! Best RMSE: {rmse_svr:.6f}")
print(f"Best Params: {grid_svr.best_params_}\n")


# --- 4B: Optuna (Boosting Models) ---
print("\n--- Starting Optuna for Boosting Models ---")
# We create an "objective" function for each model
# Optuna will try to find params that maximize the score (so we use neg_mean_squared_error)
optuna.logging.set_verbosity(optuna.logging.WARNING) # Quiets the output

# 1. GradientBoostingRegressor (GBR)
def objective_gbr(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 8, 16),
    }
    model = GradientBoostingRegressor(loss='huber', random_state=42, **params)
    score = cross_val_score(model, X, y, cv=kfolds, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score

study_gbr = optuna.create_study(direction='maximize')
# NOTE: n_trials is low for a quick demo. Use 100+ for a real search.
study_gbr.optimize(objective_gbr, n_trials=10) 
rmse_gbr = np.sqrt(-study_gbr.best_value)
print(f"GBR Tuned! Best RMSE: {rmse_gbr:.6f}")
print(f"Best Params: {study_gbr.best_params}\n")

# 2. XGBRegressor (XGB)
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
        'gamma': trial.suggest_float('gamma', 0.0, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 0.1, log=True)
    }
    model = XGBRegressor(objective='reg:linear', nthread=-1, seed=27, **params)
    score = cross_val_score(model, X, y, cv=kfolds, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=10) # NOTE: Low n_trials for demo
rmse_xgb = np.sqrt(-study_xgb.best_value)
print(f"XGBoost Tuned! Best RMSE: {rmse_xgb:.6f}")
print(f"Best Params: {study_xgb.best_params}\n")

# 3. LGBMRegressor (LGBM)
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 3000, 6000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 4, 10),
        'max_bin': trial.suggest_int('max_bin', 150, 250),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 0.4),
    }
    model = LGBMRegressor(objective='regression', verbose=-1, bagging_seed=7, feature_fraction_seed=7, **params)
    score = cross_val_score(model, X, y, cv=kfolds, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=10) # NOTE: Low n_trials for demo
rmse_lgbm = np.sqrt(-study_lgbm.best_value)
print(f"LightGBM Tuned! Best RMSE: {rmse_lgbm:.6f}")
print(f"Best Params: {study_lgbm.best_params}\n")

# --- 5. FINAL TUNING SUMMARY ---
print("\n--- Hyperparameter Tuning Complete ---")
print(f"Lasso Best RMSE:     {rmse_lasso:.6f}")
print(f"Ridge Best RMSE:     {rmse_ridge:.6f}")
print(f"SVR Best RMSE:       {rmse_svr:.6f}")
print(f"GBR Best RMSE:       {rmse_gbr:.6f}")
print(f"XGBoost Best RMSE:   {rmse_xgb:.6f}")
print(f"LightGBM Best RMSE:  {rmse_lgbm:.6f}")