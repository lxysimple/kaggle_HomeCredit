
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

from torch.nn.modules.loss import _WeightedLoss

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
# ======================================== 清理数据 =====================================
    
# ======================================== print =====================================
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
# ======================================== print =====================================

# ======================================== nn模型 =====================================
import torch.nn as nn
from torch.nn.parallel import DataParallel

all_feat_cols = [i for i in range(386)]
target_cols = [i for i in range(1)]

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2) # 0.2

        dropout_rate = 0.2 # 0.1>0.2
        hidden_size = 256 # 386>256
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))
        
        
        # ================================
        self.dense41 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm41 = nn.BatchNorm1d(hidden_size)
        self.dropout41 = nn.Dropout(dropout_rate)

        self.dense42 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm42 = nn.BatchNorm1d(hidden_size)
        self.dropout42 = nn.Dropout(dropout_rate)

        self.dense6 = nn.Linear(4*hidden_size, len(target_cols))
        # ================================

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)


        x = torch.cat([x3, x4], 1)



        # # my code
        # x41 = self.dense41(x)
        # x41 = self.batch_norm41(x41)
        # x41 = self.LeakyReLU(x41)
        # x41 = self.dropout41(x41) 
        # x = torch.cat([x4, x41], 1)
        # # my code
        # x42 = self.dense42(x)
        # x42 = self.batch_norm42(x42)
        # x42 = self.LeakyReLU(x42)
        # x42 = self.dropout42(x42) 
        # x = torch.cat([x41, x42], 1)

        # x = self.dense5(x)

        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.dense6(x)

        
        x = x.squeeze()
        
        return x
    
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2) # 0.2

        dropout_rate = 0.2 # 0.1>0.2
        hidden_size = 256 # 386>256
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))
        
        
        # ================================
        self.denses = []
        self.batch_norms = []
        self.dropouts = []
        for i in range(10):
            dense = nn.Linear(hidden_size+hidden_size, hidden_size)
            self.denses.append(dense)
        for i in range(10):
            batch_norm = nn.BatchNorm1d(hidden_size)
            self.batch_norms.append(batch_norm)
        for i in range(10):
            dropout = nn.Dropout(dropout_rate)
            self.dropouts.append(dropout)
        


        self.dense41 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm41 = nn.BatchNorm1d(hidden_size)
        self.dropout41 = nn.Dropout(dropout_rate)

        self.dense42 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm42 = nn.BatchNorm1d(hidden_size)
        self.dropout42 = nn.Dropout(dropout_rate)

        self.dense6 = nn.Linear(14*hidden_size, len(target_cols))
        # ================================

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)


        x = torch.cat([x3, x4], 1)



        # # my code
        # x41 = self.dense41(x)
        # x41 = self.batch_norm41(x41)
        # x41 = self.LeakyReLU(x41)
        # x41 = self.dropout41(x41) 
        # x = torch.cat([x4, x41], 1)
        # # my code
        # x42 = self.dense42(x)
        # x42 = self.batch_norm42(x42)
        # x42 = self.LeakyReLU(x42)
        # x42 = self.dropout42(x42) 
        # x = torch.cat([x41, x42], 1)

        x_res = []
        x_res.append(x1)
        x_res.append(x2)
        x_res.append(x3)
        x_res.append(x4)

        x_pre = x4
        for i in range(10):
            x_i = self.denses[i](x)
            x_i = self.batch_norms[i](x_i)
            x_i = self.dropouts[i](x_i)
            x_res.append(x_i)
            x = torch.cat([x_pre, x_i], 1)
            x_pre = x_i
            
        # x = self.dense5(x)

        # x = torch.cat([x1, x2, x3, x4, x41, x42], 1)
            
        x = torch.cat(x_res, 1)
        x = self.dense6(x)

        x = x.squeeze()
        
        return x

from torch.utils.data import Dataset

class MarketDataset:
    def __init__(self, features, label):
        
        self.features = features
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
        }



class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)

        with torch.no_grad():
            outputs = model(features)
        
        preds.append(outputs.detach().cpu().numpy())
#         preds.append(outputs.sigmoid().detach().cpu().numpy())

#     print(len(preds))
    preds = np.concatenate(preds).reshape(-1, 1)


    return preds

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        features = data['features'].to(device)
        label = data['label'].to(device)
        outputs = model(features)

        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


# ======================================== nn模型 =====================================

# ======================================== nn模型训练 =====================================

from torch.utils.data import DataLoader
import torch
import time
import torch.nn.functional as F



fold = 1
for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

    # from IPython import embed
    # embed()

    # X_train(≈40000,386), y_train(≈40000)
    X_train, y_train = df_train.iloc[idx_train].values, y.iloc[idx_train].values 
    X_valid, y_valid = df_train.iloc[idx_valid].values, y.iloc[idx_valid].values


    
    # 定义dataset与dataloader
    train_set = MarketDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=15000, shuffle=True, num_workers=7)
    valid_set = MarketDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_set, batch_size=15000, shuffle=False, num_workers=7)

    # print(valid_set[0])

    
    print(f'Fold{fold}:') 
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    model = Model2()
    model = model.cuda()
    model = DataParallel(model)

    # lr = 1e-3 weight_decay=1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = None
#     loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = SmoothBCEwLogits(smoothing=0.005) # 0.005

    for epoch in range(20):
            start_time = time.time()
            train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)
            valid_pred = inference_fn(model, valid_loader, device)
            valid_auc = roc_auc_score(y_valid, valid_pred)
            print(f"FOLD{fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
                    f"roc_auc_score={valid_auc:.5f} "
                    f"time: {(time.time() - start_time) / 60:.2f}min")

    fold = fold+1


# fold_index = int(len(y)*0.8)

# # 深拷贝前4折训练部分数据，df转numpy，下标索引从0开始
# train_set_raw = df_train[0:fold_index].copy()
# train_set_raw = train_set_raw.reset_index(drop=True).to_numpy()
# train_y_raw = y[0:fold_index].copy()
# train_y_raw = train_y_raw.reset_index(drop=True).to_numpy()
# valid_set_raw = df_train[fold_index:].copy()
# valid_set_raw = valid_set_raw.reset_index(drop=True).to_numpy()
# valid_y_raw = y[fold_index:].copy()
# valid_y_raw = valid_y_raw.reset_index(drop=True).to_numpy()

# # def model_nn_train(train_set_raw, train_y_raw):

# # 定义dataset与dataloader
# train_set = MarketDataset(train_set_raw, train_y_raw)
# train_loader = DataLoader(train_set, batch_size=8192, shuffle=True, num_workers=1)
# valid_set = MarketDataset(valid_set_raw, valid_y_raw)
# valid_loader = DataLoader(valid_set, batch_size=8192, shuffle=False, num_workers=1)

# # print(valid_set[0])

# for _fold in range(1):
#     print(f'Fold{_fold}:')
#     torch.cuda.empty_cache()
#     device = torch.device("cuda")

#     model = Model()
#     model = model.cuda()
#     model = DataParallel(model)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     scheduler = None
# #     loss_fn = nn.BCEWithLogitsLoss()
#     loss_fn = SmoothBCEwLogits(smoothing=0.005) # 0.005

#     for epoch in range(20):
#             start_time = time.time()
#             train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)
#             valid_pred = inference_fn(model, valid_loader, device)
#             valid_auc = roc_auc_score(valid_y_raw, valid_pred)
#             print(f"FOLD{_fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
#                       f"roc_auc_score={valid_auc:.5f} "
#                       f"time: {(time.time() - start_time) / 60:.2f}min")

# ======================================== nn模型训练 =====================================

# ======================================== 训练3树模型 =====================================
# # %%time

# fitted_models_cat = []
# fitted_models_lgb = []
# fitted_models_xgb = []
# fitted_models_rf = []

# cv_scores_cat = []
# cv_scores_lgb = []
# cv_scores_xgb = []
# cv_scores_rf = []

# fold = 1
# for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

#     # X_train(≈40000,386), y_train(≈40000)
#     X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
#     X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]    
        
#     # ======================================
#     # train_pool = Pool(X_train, y_train,cat_features=cat_cols)
#     # val_pool = Pool(X_valid, y_valid,cat_features=cat_cols)
    
# #     train_pool = Pool(X_train, y_train)
# #     val_pool = Pool(X_valid, y_valid)

# #     clf = CatBoostClassifier(
# #         eval_metric='AUC',
# #         task_type='GPU',
# #         learning_rate=0.03, # 0.03
# #         iterations=n_est, # n_est
# # #         early_stopping_rounds = 500,
# #     )
# #     # clf = CatBoostClassifier(
# #     #     eval_metric='AUC',
# #     #     task_type='GPU',
# #     #     learning_rate=0.05, # 0.03
# #     #     # iterations=n_est, # n_est iterations与n_estimators二者只能有一
# #     #     grow_policy = 'Lossguide',
# #     #     max_depth = 10,
# #     #     n_estimators = 2000,   
# #     #     reg_lambda = 10,
# #     #     num_leaves = 64,
# #     #     early_stopping_rounds = 100,
# #     # )

# #     random_seed=3107
# #     clf.fit(
# #         train_pool, 
# #         eval_set=val_pool,
# #         verbose=300,
# # #         # 保证调试的时候不需要重新训练
# # #         save_snapshot = True, 
# # #         snapshot_file = '/kaggle/working/catboost.cbsnapshot',
# # #         snapshot_interval = 10
# #     )
# #     clf.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/catboost_fold{fold}.cbm')
# #     fitted_models_cat.append(clf)
# #     y_pred_valid = clf.predict_proba(X_valid)[:,1]
# #     auc_score = roc_auc_score(y_valid, y_pred_valid)
# #     cv_scores_cat.append(auc_score)
#     # ==================================
    
#     # ==================================
#     # 一些列是很多单词，将这些单词变为唯一标号，该列就能进行多类别分类了
# #     X_train[cat_cols] = X_train[cat_cols].astype("category") 
# #     X_valid[cat_cols] = X_valid[cat_cols].astype("category")
    
# #     bst = XGBClassifier(
# #         n_estimators=2000, # 2000颗树
# #         max_depth=10,  # 10
# #         learning_rate=0.05, 
# #         objective='binary:logistic', # 最小化的目标函数，利用它优化模型
# #         metric= "auc", # 利用它选best model
# #         device= 'gpu',
# #         early_stopping_rounds=100, 
# #         enable_categorical=True, # 使用分类转换算法
# #         tree_method="hist", # 使用直方图算法加速
# #         reg_alpha = 0.1, # L1正则化0.1
# #         reg_lambda = 10, # L2正则化10
# #         max_leaves = 64, # 64
# #     )
# #     bst.fit(
# #         X_train, 
# #         y_train, 
# #         eval_set=[(X_valid, y_valid)],
# #         verbose=300,
# #     )
# #     fitted_models_xgb.append(bst)
# #     y_pred_valid = bst.predict_proba(X_valid)[:,1]
# #     auc_score = roc_auc_score(y_valid, y_pred_valid)
# #     cv_scores_xgb.append(auc_score)
# #     print(f'fold:{fold},auc_score:{auc_score}')
#     # ===============================
    
#     # ===============================
#     # X_train[cat_cols] = X_train[cat_cols].astype("category")
#     # X_valid[cat_cols] = X_valid[cat_cols].astype("category")
#     # params = {
#     #     "boosting_type": "gbdt",
#     #     "objective": "binary",
#     #     "metric": "auc",
#     #     "max_depth": 10,  
#     #     "learning_rate": 0.05,
#     #     "n_estimators": 2000,  
#     #     # 则每棵树在构建时会随机选择 80% 的特征进行训练，剩下的 20% 特征将不参与训练，从而增加模型的泛化能力和稳定性
#     #     "colsample_bytree": 0.8, 
#     #     "colsample_bynode": 0.8, # 控制每个节点的特征采样比例
#     #     "verbose": -1,
#     #     "random_state": 42,
#     #     "reg_alpha": 0.1,
#     #     "reg_lambda": 10,
#     #     "extra_trees":True,
#     #     'num_leaves':64,
#     #     "device": 'gpu', # gpu
#     #     'gpu_use_dp' : True # 转化float为64精度
#     # }

#     # # 一次训练
#     # model = lgb.LGBMClassifier(**params)
#     # model.fit(
#     #     X_train, y_train,
#     #     eval_set = [(X_valid, y_valid)],
#     #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
#     #     # init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
#     # )
#     # model2 = model
#     # model.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    
#     # # 二次优化
#     # params['learning_rate'] = 0.01
#     # model2 = lgb.LGBMClassifier(**params)
#     # model2.fit(
#     #     X_train, y_train,
#     #     eval_set = [(X_valid, y_valid)],
#     #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(200)],
#     #     init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
#     # )
#     # model2.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
#     # fitted_models_lgb.append(model2)
#     # y_pred_valid = model2.predict_proba(X_valid)[:,1]
#     # auc_score = roc_auc_score(y_valid, y_pred_valid)
#     # cv_scores_lgb.append(auc_score)
#     # print()
#     # print("分隔符")
#     # print()
    # ===========================
    
    fold = fold+1

print("CV AUC scores: ", cv_scores_cat)
print("Mean CV AUC score: ", np.mean(cv_scores_cat))

print("CV AUC scores: ", cv_scores_lgb)
print("Mean CV AUC score: ", np.mean(cv_scores_lgb))

print("CV AUC scores: ", cv_scores_xgb)
print("Mean CV AUC score: ", np.mean(cv_scores_xgb))

# ======================================== 训练3树模型 =====================================









