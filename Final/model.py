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