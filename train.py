
# ======================================== 导入配置 =====================================
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

import torch

import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)

# ======================================== 导入配置 =====================================


def to_pandas(df_data, cat_cols=None):
    
    if not isinstance(df_data, pd.DataFrame):
        df_data = df_data.to_pandas()
    
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols

df_train = pd.read_csv('/home/xyli/kaggle/train.csv')

_, cat_cols = to_pandas(df_train)

# sample = pd.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/sample_submission.csv")
device='gpu'
#n_samples=200000
n_est=6000
# DRY_RUN = True if sample.shape[0] == 10 else False   
# if DRY_RUN:
# if True:
if False: 
    device= 'gpu' # 'cpu'
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
# df_test[cat_cols] = df_test[cat_cols].astype(str)


# df_train = copy.deepcopy(df_train_copy)

# ======================================== 清理数据 =====================================
# """
# 对cat_cols列外的所有列进行数据清理，即把nan和inf换成该列的均值
# """

# # 找到除cat_cols列外的所有列
# non_cat_cols = df_train.columns.difference(cat_cols) 
# print('df_train.shape: ', df_train.shape)
# print('df_train[cat_cols].shape: ', df_train[cat_cols].shape)
# print('df_train[non_cat_cols].shape: ', df_train[non_cat_cols].shape)
# # 求1列均值时，遇到nan/inf会自动忽略
# mean_values = df_train[non_cat_cols].mean()# 找到所有列的均值
# # 如果该列都是nan/inf，均值为inf，则令均值为0
# mean_values = mean_values.replace([np.inf, -np.inf], 0)

# for column in non_cat_cols:   
#     # 将nan换成该列的均值，或者0
#     df_train[column] = df_train[column].fillna(mean_values[column])
#     # 将+-无穷值替换为该列均值
#     df_train[column].replace([np.inf,-np.inf], mean_values[column], inplace=True)
    
# # print('df_train: ',df_train[non_cat_cols])
    


# """
# 对cat_cols列进行编码，保存113个编码器
# """
# print('len(cat_cols): ', len(cat_cols))
# # 定义113个编码器
# label_encoders = [LabelEncoder() for i in range(df_train.shape[1])]

# # print(df_train[cat_cols])

# # 对每列进行一个编码
# for i in range(len(cat_cols)):
#     df_encoded = label_encoders[i].fit_transform(df_train[cat_cols[i]])
#     df_train[cat_cols[i]] = df_encoded
# ======================================== 清理数据 =====================================
    
# ======================================== print =====================================
# """ 查看分类器的映射字典 """

# print(label_encoders[0].classes_)
# print(label_encoders[1].classes_)
# print(label_encoders[2].classes_)
# print(label_encoders[3].classes_)
# print(label_encoders[5].classes_)


# """ 看一下数值列数据清理和非数值列打标签后大体是啥样 """

# print('np.max(df_train[cat_cols]): ', np.max(df_train[cat_cols]))
# print('np.min(df_train[cat_cols]): ', np.min(df_train[cat_cols]))

# print('np.max(df_train[non_cat_cols]): ', np.max(df_train[non_cat_cols]))
# print('np.min(df_train[non_cat_cols]): ', np.min(df_train[non_cat_cols]))

# print('np.max(df_train): ', np.max(df_train))
# print('np.min(df_train): ', np.min(df_train))
# ======================================== print =====================================


# ======================================== 训练3树模型 =====================================
# %%time

fitted_models_cat = []
fitted_models_lgb = []
fitted_models_xgb = []
fitted_models_rf = []

cv_scores_cat = []
cv_scores_lgb = []
cv_scores_xgb = []
cv_scores_rf = []

fold = 1
for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

    # X_train(≈40000,386), y_train(≈40000)
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]    
        
    # ======================================
    train_pool = Pool(X_train, y_train,cat_features=cat_cols)
    val_pool = Pool(X_valid, y_valid,cat_features=cat_cols)
    
    # train_pool = Pool(X_train, y_train)
    # val_pool = Pool(X_valid, y_valid)

#     clf = CatBoostClassifier(
#         eval_metric='AUC',
#         task_type='GPU',
#         learning_rate=0.03,
#         iterations=n_est, # n_est
# #         early_stopping_rounds = 500,
#     )
    clf = CatBoostClassifier(
        eval_metric='AUC',
        task_type='GPU',
        learning_rate=0.05, # 0.03
        iterations=n_est, # n_est
        max_depth = 10,
        n_estimators = 2000,
        colsample_bylevel = 0.8,    
        reg_lambda = 10,
        num_leaves = 64,
#         early_stopping_rounds = 500,
    )

    random_seed=3107
    clf.fit(
        train_pool, 
        eval_set=val_pool,
        verbose=300,
#         # 保证调试的时候不需要重新训练
#         save_snapshot = True, 
#         snapshot_file = '/kaggle/working/catboost.cbsnapshot',
#         snapshot_interval = 10
    )
    clf.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/catboost_fold{fold}.cbm')
    fitted_models_cat.append(clf)
    y_pred_valid = clf.predict_proba(X_valid)[:,1]
    auc_score = roc_auc_score(y_valid, y_pred_valid)
    cv_scores_cat.append(auc_score)
    # ==================================
    
    # ==================================
    # 一些列是很多单词，将这些单词变为唯一标号，该列就能进行多类别分类了
#     X_train[cat_cols] = X_train[cat_cols].astype("category") 
#     X_valid[cat_cols] = X_valid[cat_cols].astype("category")
    
#     bst = XGBClassifier(
#         n_estimators=2000, # 2000颗树
#         max_depth=10,  # 10
#         learning_rate=0.05, 
#         objective='binary:logistic', # 最小化的目标函数，利用它优化模型
#         metric= "auc", # 利用它选best model
#         device= 'gpu',
#         early_stopping_rounds=100, 
#         enable_categorical=True, # 使用分类转换算法
#         tree_method="hist", # 使用直方图算法加速
#         reg_alpha = 0.1, # L1正则化0.1
#         reg_lambda = 10, # L2正则化10
#         max_leaves = 64, # 64
#     )
#     bst.fit(
#         X_train, 
#         y_train, 
#         eval_set=[(X_valid, y_valid)],
#         verbose=300,
#     )
#     fitted_models_xgb.append(bst)
#     y_pred_valid = bst.predict_proba(X_valid)[:,1]
#     auc_score = roc_auc_score(y_valid, y_pred_valid)
#     cv_scores_xgb.append(auc_score)
#     print(f'fold:{fold},auc_score:{auc_score}')
    # ===============================
    
    # ===============================
#     X_train[cat_cols] = X_train[cat_cols].astype("category")
#     X_valid[cat_cols] = X_valid[cat_cols].astype("category")
#     params = {
#         "boosting_type": "gbdt",
#         "objective": "binary",
#         "metric": "auc",
#         "max_depth": 10,  
#         "learning_rate": 0.05,
#         "n_estimators": 2000,  
#         # 则每棵树在构建时会随机选择 80% 的特征进行训练，剩下的 20% 特征将不参与训练，从而增加模型的泛化能力和稳定性
#         "colsample_bytree": 0.8, 
#         "colsample_bynode": 0.8, # 控制每个节点的特征采样比例
#         "verbose": -1,
#         "random_state": 42,
#         "reg_alpha": 0.1,
#         "reg_lambda": 10,
#         "extra_trees":True,
#         'num_leaves':64,
#         "device": 'gpu', # gpu
#         'gpu_use_dp' : True # 转化float为64精度
#     }
#     model = lgb.LGBMClassifier(**params)
# #     model = lgb.Booster(model_file=f"/kaggle/input/credit-models/lgbm_fold{fold}.txt")
#     model.fit(
#         X_train, y_train,
#         eval_set = [(X_valid, y_valid)],
#         callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
# #         init_model = f"/kaggle/input/credit-models/lgbm_fold{fold}.txt",
#     )
#     model.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
#     # 二次优化
# #     params['learning_rate'] = 0.01
# #     model2 = lgb.LGBMClassifier(**params)
# #     model2.fit(
# #         X_train, y_train,
# #         eval_set = [(X_valid, y_valid)],
# #         callbacks = [lgb.log_evaluation(200), lgb.early_stopping(500)],
# #         init_model = model,
# #     )
#     fitted_models_lgb.append(model)
#     y_pred_valid = model.predict_proba(X_valid)[:,1]
#     auc_score = roc_auc_score(y_valid, y_pred_valid)
#     cv_scores_lgb.append(auc_score)
#     print()
#     print("分隔符")
#     print()
    # ===========================
    
    fold = fold+1

print("CV AUC scores: ", cv_scores_cat)
print("Mean CV AUC score: ", np.mean(cv_scores_cat))

print("CV AUC scores: ", cv_scores_lgb)
print("Mean CV AUC score: ", np.mean(cv_scores_lgb))

print("CV AUC scores: ", cv_scores_xgb)
print("Mean CV AUC score: ", np.mean(cv_scores_xgb))

# ======================================== 训练3树模型 =====================================









