# --- 1. LOAD LIBRARIES ---
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings # Import warnings

# --- 2. LOAD DATA ---
warnings.filterwarnings('ignore')
print("Loading data...")

# These paths are correct for the Kaggle environment
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print ("Data is loaded!")

# Drop 'Id'
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# --- 3. DATA PROCESSING AND FEATURE ENGINEERING ---
print("Starting data processing...")

# Remove outliers
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)

# Log transform target
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)

# Concat features
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

# Fill remaining NaNs
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
print(f"Applying Box-Cox transform to {len(skew_index)} skewed features...")

for i in skew_index:
    if len(features[i].unique()) == 1:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lambda_val = boxcox_normmax(features[i] + 1)
            features[i] = boxcox1p(features[i], lambda_val)
        except Exception:
            features[i] = np.log1p(features[i])
# --- END SKEWNESS TRANSFORM ---

# Drop features
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

# Create new features
features['YrBltAndRemod'] = features['YearBuilt'] + features['YearRemodAdd']
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                features['1stFlrSF'] + features['2ndFlrSF'])
features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))
features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                             features['EnclosedPorch'] + features['ScreenPorch'] +
                             features['WoodDeckSF'])
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Dummify
final_features = pd.get_dummies(features).reset_index(drop=True)
print(f"Final feature shape: {final_features.shape}")

# Split back to X and X_sub
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]

# Remove outliers from X and y
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

# Remove overfit features
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
if 'MSZoning_C (all)' in X.columns:
    overfit.append('MSZoning_C (all)')

X = X.drop(overfit, axis=1)
X_sub = X_sub.drop(overfit, axis=1)
print(f"Final training/submission shapes: {X.shape}, {y.shape}, {X_sub.shape}")

# --- 4. MODEL DEFINITION ---
print("Defining models...")
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

# Base Models
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=10000000, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=10000000, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                 min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
lightgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01,
                       n_estimators=5000, max_bin=200, bagging_fraction=0.75,
                       bagging_freq=5, bagging_seed=7, feature_fraction=0.2,
                       feature_fraction_seed=7, verbose=-1)
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                max_depth=3, min_child_weight=0,
                                gamma=0, subsample=0.7, colsample_bytree=0.7,
                                objective='reg:linear', nthread=-1,
                                scale_pos_weight=1, seed=27, reg_alpha=0.00006)

# Define a simple linear meta-model
lasso_meta = make_pipeline(RobustScaler(),
                            LassoCV(max_iter=10000000, alphas=alphas2,
                                    random_state=42, cv=kfolds))

# Stack 1: XGBoost as voter
stack_gen_xgb = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                  meta_regressor=xgboost,
                                  use_features_in_secondary=True)

# Stack 2: Lasso as voter
stack_gen_lasso = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                    meta_regressor=lasso_meta,
                                    use_features_in_secondary=True)

# --- 5. MODEL TRAINING ---
print('START Fit')

print('Stack 1 (XGB Meta)')
stack_gen_xgb_model = stack_gen_xgb.fit(np.array(X), np.array(y))

print('Stack 2 (Lasso Meta)')
stack_gen_lasso_model = stack_gen_lasso.fit(np.array(X), np.array(y))

print('All stack models fitted.')

