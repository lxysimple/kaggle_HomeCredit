
# ============================ 导入配置 ============================
import sys
from pathlib import Path
import subprocess
import os
import gc
from glob import glob

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

ROOT = '/kaggle/input/home-credit-credit-risk-model-stability'

from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

from catboost import CatBoostClassifier, Pool
import joblib
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

# ============================ 导入配置 ============================



df_train = pd.read_csv('/kaggle/input/home-credit-lgb-cat-ensemble/train.csv')

_, cat_cols = to_pandas(df_train)

sample = pd.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/sample_submission.csv")
device='gpu'
#n_samples=200000
n_est=6000
DRY_RUN = True if sample.shape[0] == 10 else False   
if DRY_RUN:
# if False:
    device='cpu'
    df_train = df_train.iloc[:50000]
    #n_samples=10000
    n_est=600
print(device)

y = df_train["target"]
weeks = df_train["WEEK_NUM"]
try:
    df_train= df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
except:
    print("这个代码已经执行过1次了！")
cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

df_train[cat_cols] = df_train[cat_cols].astype(str)
df_test[cat_cols] = df_test[cat_cols].astype(str)


# df_train = copy.deepcopy(df_train_copy)

# ============================ 数据清理 ============================
"""
对cat_cols列外的所有列进行数据清理，即把nan和inf换成该列的均值
"""

# 找到除cat_cols列外的所有列
non_cat_cols = df_train.columns.difference(cat_cols) 
print('df_train.shape: ', df_train.shape)
print('df_train[cat_cols].shape: ', df_train[cat_cols].shape)
print('df_train[non_cat_cols].shape: ', df_train[non_cat_cols].shape)
# 求1列均值时，遇到nan/inf会自动忽略
mean_values = df_train[non_cat_cols].mean()# 找到所有列的均值
# 如果该列都是nan/inf，均值为inf，则令均值为0
mean_values = mean_values.replace([np.inf, -np.inf], 0)

for column in non_cat_cols:   
    # 将nan换成该列的均值，或者0
    df_train[column] = df_train[column].fillna(mean_values[column])
    # 将+-无穷值替换为该列均值
    df_train[column].replace([np.inf,-np.inf], mean_values[column], inplace=True)
    
# print('df_train: ',df_train[non_cat_cols])
    


"""
对cat_cols列进行编码，保存113个编码器
"""
print('len(cat_cols): ', len(cat_cols))
# 定义113个编码器
label_encoders = [LabelEncoder() for i in range(df_train.shape[1])]

# print(df_train[cat_cols])

# 对每列进行一个编码
for i in range(len(cat_cols)):
    df_encoded = label_encoders[i].fit_transform(df_train[cat_cols[i]])
    df_train[cat_cols[i]] = df_encoded
# ============================ 数据清理 ============================
    
# ============================ print ============================
""" 查看分类器的映射字典 """

print(label_encoders[0].classes_)
print(label_encoders[1].classes_)
print(label_encoders[2].classes_)
print(label_encoders[3].classes_)
print(label_encoders[5].classes_)


""" 看一下数值列数据清理和非数值列打标签后大体是啥样 """

print('np.max(df_train[cat_cols]): ', np.max(df_train[cat_cols]))
print('np.min(df_train[cat_cols]): ', np.min(df_train[cat_cols]))

print('np.max(df_train[non_cat_cols]): ', np.max(df_train[non_cat_cols]))
print('np.min(df_train[non_cat_cols]): ', np.min(df_train[non_cat_cols]))

print('np.max(df_train): ', np.max(df_train))
print('np.min(df_train): ', np.min(df_train))
# ============================ print ============================














