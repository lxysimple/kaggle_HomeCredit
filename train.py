
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

from functools import lru_cache


# ======================================== 导入配置 =====================================


def to_pandas(df_data, cat_cols=None):
    
    if not isinstance(df_data, pd.DataFrame):
        df_data = df_data.to_pandas()
    
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


# file_path = '/home/xyli/kaggle/kaggle_HomeCredit/train2.csv'
# # 打开文件并逐行读取
# with open(file_path, 'r', encoding='utf-8') as file:
#     row_count = sum(1 for line in file)
# print("行数:", row_count)


# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train419.csv', nrows=5)
# df_train = pd.read_csv('/home/xyli/kaggle/train.csv', nrows=50001)
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train2.csv')
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train419.csv')
df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train389.csv')

# """ 确保df_train[cat_cols]中每列字典都有nan值 """
# new_row = pd.DataFrame([[np.nan] * len(df_train.columns)], columns=df_train.columns)
# # 将新行添加到DataFrame中
# df_train = pd.concat([df_train, new_row], ignore_index=True)

_, cat_cols = to_pandas(df_train)

# sample = pd.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/sample_submission.csv")
device='gpu'
#n_samples=200000
n_est=12000 # 6000
# DRY_RUN = True if sample.shape[0] == 10 else False   
# if DRY_RUN:
if True:
# if False: 
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
mean_values = mean_values.replace([np.inf, -np.inf, np.nan], 0)

for column in non_cat_cols:   
    # # 将nan换成该列的均值，或者0
    # df_train[column] = df_train[column].fillna(mean_values[column])
    
    # 将nan换成0
    df_train[column] = df_train[column].fillna(0)
    # 将+-无穷值替换为0
    df_train[column].replace([np.inf,-np.inf], 0, inplace=True)
    
# print('df_train: ',df_train[non_cat_cols])
    


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



# # 因为最后一行全是nan，只是为了让编码器学习nan，所以现在就可以去掉了
# y = y[:-1]
# df_train = df_train[:-1]
# weeks = weeks[:-1]
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

# ======================================== 其他模型 =====================================

# from sklearn.metrics import mean_squared_error as MSE 
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import ElasticNet

# fold = 1
# for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

#     # from IPython import embed
#     # embed()

#     # X_train(≈40000,386), y_train(≈40000)
#     X_train, y_train = df_train.iloc[idx_train].values, y.iloc[idx_train].values 
#     X_valid, y_valid = df_train.iloc[idx_valid].values, y.iloc[idx_valid].values

#     # model_1 = Ridge(alpha=1.0) # 0.80
#     model_1 = ElasticNet(random_state=0) # 0.74
#     model_1.fit(X_train, y_train)

    

#     valid_pred = model_1.predict(X_valid) 
#     valid_auc = roc_auc_score(y_valid, valid_pred)
#     print('valid_auc: ', valid_auc)

#     fold = fold + 1  
# ======================================== 其他模型 =====================================


# ======================================== nn模型 =====================================
import torch.nn as nn
from torch.nn.parallel import DataParallel

all_feat_cols = [i for i in range(273)] # 386 386-113(cat_cols)=273
target_cols = [i for i in range(1)]

# class Attention(nn.Module):
#     def __init__(self, in_features, hidden_dim):
#         super(Attention, self).__init__()
#         self.linear1 = nn.Linear(in_features, hidden_dim*4, bias=False)
#         self.linear2 = nn.Linear(hidden_dim*4, in_features, bias=False)
#         self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
#         self.sigmoid = nn.Sigmoid()    
#     def forward(self, x):
#         # # 进行平均池化, (b,w)->(1,w)
#         # out = torch.mean(x, axis=-2,  keepdim=True)

#         # 输入特征经过线性层和激活函数
#         # out = F.relu(self.linear1(x))
#         out = self.LeakyReLU(self.linear1(x))

#         # 再经过一个线性层得到注意力权重
#         # attn_weights = F.softmax(self.linear2(out), dim=1)
#         attn_weights = self.sigmoid(self.linear2(out))
#         # 使用注意力权重加权得到加权后的特征
#         return attn_weights * x
        
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

        self.dense3 = nn.Linear(len(all_feat_cols)+2*hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(len(all_feat_cols)+3*hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(2*hidden_size, len(target_cols))



        hidden_size2 = 512 # 386>256
        self.dense21 = nn.Linear(len(all_feat_cols), hidden_size2)
        self.batch_norm21 = nn.BatchNorm1d(hidden_size2)
        self.dropout21 = nn.Dropout(dropout_rate)

        self.dense22 = nn.Linear(hidden_size2+len(all_feat_cols), hidden_size2)
        self.batch_norm22 = nn.BatchNorm1d(hidden_size2)
        self.dropout22 = nn.Dropout(dropout_rate)

        self.dense23 = nn.Linear(len(all_feat_cols)+2*hidden_size2, hidden_size2)
        self.batch_norm23 = nn.BatchNorm1d(hidden_size2)
        self.dropout23 = nn.Dropout(dropout_rate)

        self.dense24 = nn.Linear(len(all_feat_cols)+3*hidden_size2, hidden_size2)
        self.batch_norm24 = nn.BatchNorm1d(hidden_size2)
        self.dropout24 = nn.Dropout(dropout_rate)

        self.dense25 = nn.Linear(2*hidden_size2 + 2*hidden_size, len(target_cols))
        # self.dense51 = nn.Linear(2*hidden_size, hidden_size//8)
        # self.dense52 = nn.Linear(2*hidden_size, hidden_size//2)
        # self.batch_norm51 = nn.BatchNorm1d(hidden_size//8)
        # self.dropout51 = nn.Dropout(dropout_rate)
        # self.batch_norm52 = nn.BatchNorm1d(hidden_size//2)
        # self.dropout52 = nn.Dropout(dropout_rate)

        # self.dense5 = nn.Linear(hidden_size//8+hidden_size//2, len(target_cols)) 
        # ================================

        # self.denses = nn.ModuleList()
        # for i in range(50):
        #     self.dense = nn.Linear(len(all_feat_cols), hidden_size)
        #     self.denses.append(self.dense)

        # self.batch_norms = nn.ModuleList()
        # for i in range(50):
        #     self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        #     self.batch_norms.append(self.batch_norm4)

        # self.denses2 = nn.ModuleList()
        # for i in range(50):
        #     self.dense = nn.Linear(hidden_size, 1)
        #     self.denses2.append(self.dense)   

        self.denses = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(5):
            self.dense = nn.Linear((i+4)*hidden_size+386, hidden_size) 
            self.denses.append(self.dense)
        for i in range(5):
            self.batch_norm = nn.BatchNorm1d(hidden_size)
            self.batch_norms.append(self.batch_norm)
        for i in range(5):
            self.dropout = nn.Dropout(dropout_rate)
            self.dropouts.append(self.dropout)

        self.dense41 = nn.Linear(len(all_feat_cols)+4*hidden_size, hidden_size)
        self.batch_norm41 = nn.BatchNorm1d(hidden_size)
        self.dropout41 = nn.Dropout(dropout_rate)

        # self.dense42 = nn.Linear(6*hidden_size, hidden_size)
        # self.batch_norm42 = nn.BatchNorm1d(hidden_size)
        # self.dropout42 = nn.Dropout(dropout_rate)

        # self.attention1 = Attention(hidden_size, hidden_size)
        # self.attention2 = Attention(hidden_size, hidden_size)
        # self.attention3 = Attention(hidden_size, hidden_size)
        # self.attention4 = Attention(2*hidden_size, 2*hidden_size)

        self.dense6 = nn.Linear(2*hidden_size, len(target_cols))
        # ================================

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        # x_clone = x.clone()
        # x_res = []
        # for i in range(50):
        #     x_i = self.denses[i](x)
        #     x_i = self.batch_norms[i](x_i)
        #     x_i = self.denses2[i](x_i)
        #     x_res.append(x_i)
            
        # for i in range(50):
        #     x = torch.cat((x, x_res[i]), dim=1)


        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)
        # x1 = self.attention1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)
        # x2 = self.attention2(x2)

        x = torch.cat([x, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)
        # x3 = self.attention3(x3)

        x = torch.cat([x, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)
        # x4 = self.attention4(x4)
        
        x = torch.cat([x, x4], 1)


        # x = x_clone
        # x21 = self.dense21(x)
        # x21 = self.batch_norm21(x21)
        # # x = F.relu(x)
        # # x = self.PReLU(x)
        # x21 = self.LeakyReLU(x21)
        # x21 = self.dropout1(x21)
        # # x1 = self.attention1(x1)

        # x = torch.cat([x, x21], 1)

        # x22 = self.dense22(x)
        # x22 = self.batch_norm22(x22)
        # # x = F.relu(x)
        # # x = self.PReLU(x)
        # x22 = self.LeakyReLU(x22)
        # x22 = self.dropout22(x22)
        # # x2 = self.attention2(x2)

        # x = torch.cat([x, x22], 1)

        # x23 = self.dense23(x)
        # x23 = self.batch_norm23(x23)
        # # x = F.relu(x)
        # # x = self.PReLU(x)
        # x23 = self.LeakyReLU(x23)
        # x23 = self.dropout23(x23)
        # # x3 = self.attention3(x3)

        # x = torch.cat([x, x23], 1)

        # x24 = self.dense24(x)
        # x24 = self.batch_norm24(x24)
        # # x = F.relu(x)
        # # x = self.PReLU(x)
        # x24 = self.LeakyReLU(x24)
        # x24 = self.dropout24(x24)
        # # x4 = self.attention4(x4)
        
        # x = torch.cat([x, x24], 1)



        # # my code
        # x41 = self.dense41(x)
        # x41 = self.batch_norm41(x41)
        # x41 = self.LeakyReLU(x41)
        # x41 = self.dropout41(x41) 
        # x = torch.cat([x, x41], 1)
        # # my code
        # x42 = self.dense42(x)
        # x42 = self.batch_norm42(x42)
        # x42 = self.LeakyReLU(x42)
        # x42 = self.dropout42(x42) 
        # x = torch.cat([x, x42], 1)

        # x_res = []
        # x_res.append(x1)
        # x_res.append(x2)
        # x_res.append(x3)
        # x_res.append(x4)
        # x_res.append(x41)
        # x_res.append(x42)

        # x_pre = x4
        # for i in range(5):

        #     x_i = self.denses[i](x)
        #     x_i = self.batch_norms[i](x_i)
        #     x_i = self.LeakyReLU(x_i)
        #     x_i = self.dropouts[i](x_i)
        
        #     x_res.append(x_i)
        #     # x = torch.cat([x_pre, x_i], 1)
        #     x = torch.cat([x, x_i], 1)
        #     # x_pre = x_i

        
        # x = torch.cat([x3, x4, x23, x24], 1)
        x = torch.cat([x3, x4], 1)
        # x = torch.cat([x2, x3], 1)
        # x = torch.cat([x4, x41], 1)
        
        # x = self.attention4(x)

        # x51 = self.dense51(x)
        # x51 = self.batch_norm51(x51)
        # x51 = self.LeakyReLU(x51)
        # x51 = self.dropout51(x51)

        # x52 = self.dense52(x)
        # x52 = self.batch_norm52(x52)
        # x52 = self.LeakyReLU(x52)
        # x52 = self.dropout52(x52)

        # x = torch.cat([x51, x52], 1)
        # x = self.dense25(x)
        x = self.dense5(x)

        # x = torch.cat([x1, x2, x3, x4, x41, x42], 1)
            
        # x = torch.cat(x_res[-2:], 1)
        # x = self.dense6(x)

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

        final_loss += loss.item()

    if scheduler:
        scheduler.step()
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
    X_train, y_train = df_train[non_cat_cols].iloc[idx_train].values, y.iloc[idx_train].values 
    X_valid, y_valid = df_train[non_cat_cols].iloc[idx_valid].values, y.iloc[idx_valid].values


    
    # 定义dataset与dataloader
    train_set = MarketDataset(X_train, y_train)
    # batch_size=15000
    train_loader = DataLoader(train_set, batch_size=15000, shuffle=True, num_workers=7)
    valid_set = MarketDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_set, batch_size=15000, shuffle=False, num_workers=7)

    # print(valid_set[0])

    
    print(f'Fold{fold}:') 
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    model = Model2()
    
    try:
        model.load_state_dict(torch.load(f'/home/xyli/kaggle/kaggle_HomeCredit/best_nn_fold{fold}.pt'))
        print('发现可用baseline, 开始加载')   
    except:
        print('未发现可用模型, 从0训练')
    model = model.cuda()
    model = DataParallel(model)

    # lr = 1e-3 weight_decay=1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    # adam的优化版本
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = None

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, 
    #     milestones=[20,40], 
    #     gamma=0.1,
    #     last_epoch=-1
    # )

#     loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = SmoothBCEwLogits(smoothing=0.005) # 0.005

    best_train_loss = 999.0
    best_valid_auc = -1
    for epoch in range(20):
        start_time = time.time()
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)
        valid_pred = inference_fn(model, valid_loader, device)
        valid_auc = roc_auc_score(y_valid, valid_pred)
        print(
            f"FOLD{fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
            f"roc_auc_score={valid_auc:.5f} "
            f"time: {(time.time() - start_time) / 60:.2f}min "
            f"lr: {optimizer.param_groups[0]['lr']}"
        )
        with open("log.txt", "a") as f:
            print(
                f"FOLD{fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
                f"roc_auc_score={valid_auc:.5f} "
                f"time: {(time.time() - start_time) / 60:.2f}min "
                f"lr: {optimizer.param_groups[0]['lr']}", file=f
            )

        if train_loss < best_train_loss and valid_auc > best_valid_auc:
            best_train_loss = train_loss
            best_valid_auc = valid_auc
            torch.save(model.module.state_dict(), f"./best_nn_fold{fold}.pt") 
            print(
                f"best_nn_fold{fold}.pt "
                f"best_train_loss: {best_train_loss} "
                f"best_valid_auc: {best_valid_auc} "
            )
            with open("log.txt", "a") as f:
                print(
                    f"best_nn_fold{fold}.pt "
                    f"best_train_loss: {best_train_loss} "
                    f"best_valid_auc: {best_valid_auc} ", file=f
                )
            
    fold = fold+1

# ======================================== nn模型训练 =====================================

# ======================================== 训练3树模型 =====================================
# # %%time

fitted_models_cat = []
fitted_models_lgb = []
fitted_models_xgb = []
fitted_models_rf = []
fitted_models_cat_dw = []
fitted_models_cat_lg = []
fitted_models_lgb_dart = []
fitted_models_lgb_rf = []


cv_scores_cat = []
cv_scores_lgb = []
cv_scores_xgb = []
cv_scores_rf = []
cv_scores_cat_dw = []
cv_scores_cat_lg = []
cv_scores_lgb_dart = []
cv_scores_lgb_rf = []

fold = 1
for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

    # X_train(≈40000,386), y_train(≈40000)
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]    
        
    # ======================================
#     train_pool = Pool(X_train, y_train, cat_features=cat_cols)
#     val_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

     
#     clf = CatBoostClassifier(
#         grow_policy = 'Lossguide', 
#         eval_metric='AUC',
#         task_type='GPU',
#         learning_rate=0.03, # 0.03
#         iterations=n_est, # n_est
# #         early_stopping_rounds = 500,
#     )
#     # clf = CatBoostClassifier(
#     #     eval_metric='AUC',
#     #     task_type='GPU',
#     #     learning_rate=0.05, # 0.03
#     #     # iterations=n_est, # n_est iterations与n_estimators二者只能有一
#     #     grow_policy = 'Lossguide',
#     #     max_depth = 10,
#     #     n_estimators = 2000,   
#     #     reg_lambda = 10,
#     #     num_leaves = 64,
#     #     early_stopping_rounds = 100,
#     # )

#     random_seed=3107
#     clf.fit(
#         train_pool, 
#         eval_set=val_pool,
#         verbose=300,
# #         # 保证调试的时候不需要重新训练
# #         save_snapshot = True, 
# #         snapshot_file = '/kaggle/working/catboost.cbsnapshot',
# #         snapshot_interval = 10
#     )
#     clf.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/catboost_lg_fold{fold}.cbm')
#     fitted_models_cat_lg.append(clf)
#     y_pred_valid = clf.predict_proba(X_valid)[:,1]
#     auc_score = roc_auc_score(y_valid, y_pred_valid)
#     cv_scores_cat_lg.append(auc_score)
    
    # ==================================


    # ======================================
#     train_pool = Pool(X_train, y_train, cat_features=cat_cols)
#     val_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

     
#     clf = CatBoostClassifier(
#         grow_policy = 'Depthwise', 
#         eval_metric='AUC',
#         task_type='GPU',
#         learning_rate=0.03, # 0.03
#         iterations=n_est, # n_est
# #         early_stopping_rounds = 500,
#     )
#     # clf = CatBoostClassifier(
#     #     eval_metric='AUC',
#     #     task_type='GPU',
#     #     learning_rate=0.05, # 0.03
#     #     # iterations=n_est, # n_est iterations与n_estimators二者只能有一
#     #     grow_policy = 'Lossguide',
#     #     max_depth = 10,
#     #     n_estimators = 2000,   
#     #     reg_lambda = 10,
#     #     num_leaves = 64,
#     #     early_stopping_rounds = 100,
#     # )

#     random_seed=3107
#     clf.fit(
#         train_pool, 
#         eval_set=val_pool,
#         verbose=300,
# #         # 保证调试的时候不需要重新训练
# #         save_snapshot = True, 
# #         snapshot_file = '/kaggle/working/catboost.cbsnapshot',
# #         snapshot_interval = 10
#     )
#     clf.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/catboost_dw_fold{fold}.cbm')
#     fitted_models_cat_dw.append(clf)
#     y_pred_valid = clf.predict_proba(X_valid)[:,1]
#     auc_score = roc_auc_score(y_valid, y_pred_valid)
#     cv_scores_cat_dw.append(auc_score)
    
    # ==================================





    # ======================================
#     train_pool = Pool(X_train, y_train,cat_features=cat_cols)
#     val_pool = Pool(X_valid, y_valid,cat_features=cat_cols)
    
# #     train_pool = Pool(X_train, y_train)
# #     val_pool = Pool(X_valid, y_valid)
     
#     clf = CatBoostClassifier(
#         eval_metric='AUC',
#         task_type='GPU',
#         learning_rate=0.03, # 0.03
#         iterations=12000, # n_est
# #         early_stopping_rounds = 500,
#     )
#     # clf = CatBoostClassifier(
#     #     eval_metric='AUC',
#     #     task_type='GPU',
#     #     learning_rate=0.05, # 0.03
#     #     # iterations=n_est, # n_est iterations与n_estimators二者只能有一
#     #     grow_policy = 'Lossguide',
#     #     max_depth = 10,
#     #     n_estimators = 2000,   
#     #     reg_lambda = 10,
#     #     num_leaves = 64,
#     #     early_stopping_rounds = 100,
#     # )

#     random_seed=3107
#     clf.fit(
#         train_pool, 
#         eval_set=val_pool,
#         verbose=300,
# #         # 保证调试的时候不需要重新训练
# #         save_snapshot = True, 
# #         snapshot_file = '/kaggle/working/catboost.cbsnapshot',
# #         snapshot_interval = 10
#     )
#     clf.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/catboost_fold{fold}.cbm')
#     fitted_models_cat.append(clf)
#     y_pred_valid = clf.predict_proba(X_valid)[:,1]
#     auc_score = roc_auc_score(y_valid, y_pred_valid)
#     print('auc_score: ', auc_score)
#     cv_scores_cat.append(auc_score)
    
    # ==================================
    
    # ==================================
    # # 一些列是很多单词，将这些单词变为唯一标号，该列就能进行多类别分类了
    # X_train[cat_cols] = X_train[cat_cols].astype("category") 
    # X_valid[cat_cols] = X_valid[cat_cols].astype("category")
    
    # # bst = XGBClassifier(
    # #     n_estimators=2000, # 2000颗树
    # #     max_depth=10,  # 10
    # #     learning_rate=0.05, 
    # #     objective='binary:logistic', # 最小化的目标函数，利用它优化模型
    # #     eval_metric= "auc", # 利用它选best model
    # #     device= 'gpu',
    # #     grow_policy = 'lossguide',
    # #     early_stopping_rounds=100, 
    # #     enable_categorical=True, # 使用分类转换算法
    # #     tree_method="hist", # 使用直方图算法加速
    # #     reg_alpha = 0.1, # L1正则化0.1
    # #     reg_lambda = 10, # L2正则化10
    # #     max_leaves = 64, # 64
    # # )
    # bst = XGBClassifier(
    #     n_estimators = n_est,
    #     learning_rate=0.03, 
    #     eval_metric= "auc", # 利用它选best model
    #     device= 'gpu',
    #     grow_policy = 'lossguide',
    #     enable_categorical=True, # 使用分类转换算法
    # )

    # bst.fit(
    #     X_train, 
    #     y_train, 
    #     eval_set=[(X_valid, y_valid)],
    #     verbose=300,
    # )
    # fitted_models_xgb.append(bst)
    # y_pred_valid = bst.predict_proba(X_valid)[:,1]
    # auc_score = roc_auc_score(y_valid, y_pred_valid)
    # cv_scores_xgb.append(auc_score)
    # print(f'fold:{fold},auc_score:{auc_score}')
    # ===============================
    
    # ===============================
    # X_train[cat_cols] = X_train[cat_cols].astype("category")
    # X_valid[cat_cols] = X_valid[cat_cols].astype("category")
    # params = {
    #     "boosting_type": "gbdt",
    #     "objective": "binary",
    #     "metric": "auc",
    #     "max_depth": 10,  
    #     "learning_rate": 0.05,
    #     "n_estimators": 2000,  
    #     # 则每棵树在构建时会随机选择 80% 的特征进行训练，剩下的 20% 特征将不参与训练，从而增加模型的泛化能力和稳定性
    #     "colsample_bytree": 0.8, 
    #     "colsample_bynode": 0.8, # 控制每个节点的特征采样比例
    #     "verbose": -1,
    #     "random_state": 42,
    #     "reg_alpha": 0.1,
    #     "reg_lambda": 10,
    #     "extra_trees":True,
    #     'num_leaves':64,
    #     "device": 'gpu', # gpu
    #     'gpu_use_dp' : True # 转化float为64精度
    # }

    # # 一次训练
    # model = lgb.LGBMClassifier(**params)
    # model.fit(
    #     X_train, y_train,
    #     eval_set = [(X_valid, y_valid)],
    #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
    #     # init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
    # )
    # model.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    # model2 = model

    # # 二次优化
    # params['learning_rate'] = 0.01
    # model2 = lgb.LGBMClassifier(**params)
    # model2.fit(
    #     X_train, y_train,
    #     eval_set = [(X_valid, y_valid)],
    #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(200)],
    #     init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset8/lgbm_fold{fold}.txt",
    # )
    # model2.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    

    # fitted_models_lgb.append(model2)
    # y_pred_valid = model2.predict_proba(X_valid)[:,1]
    # auc_score = roc_auc_score(y_valid, y_pred_valid)
    # print('auc_score: ', auc_score)
    # cv_scores_lgb.append(auc_score)
    # print()
    # print("分隔符")
    # print()
    # ===========================

    # ===============================
    # X_train[cat_cols] = X_train[cat_cols].astype("category")
    # X_valid[cat_cols] = X_valid[cat_cols].astype("category")
    # params = {
    #     "boosting_type": "dart",
    #     "objective": "binary",
    #     "metric": "auc",
    #     # "max_depth": 10,  
    #     "learning_rate": 0.05,
    #     "n_estimators": 2000,  
    #     # 则每棵树在构建时会随机选择 80% 的特征进行训练，剩下的 20% 特征将不参与训练，从而增加模型的泛化能力和稳定性
    #     # "colsample_bytree": 0.8, 
    #     # "colsample_bynode": 0.8, # 控制每个节点的特征采样比例
    #     "verbose": -1,
    #     "random_state": 42,
    #     # "reg_alpha": 0.1,
    #     # "reg_lambda": 10,
    #     # "extra_trees":True,
    #     # 'num_leaves':64,
    #     "device": 'gpu', # gpu
    #     'gpu_use_dp' : True # 转化float为64精度
    # }

    # # 一次训练
    # model = lgb.LGBMClassifier(**params)
    # model.fit(
    #     X_train, y_train,
    #     eval_set = [(X_valid, y_valid)],
    #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
    #     # init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
    # )
    # model2 = model
    # model.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_dart_fold{fold}.txt')
    

    # # 二次优化
    # # params['learning_rate'] = 0.01
    # # model2 = lgb.LGBMClassifier(**params)
    # # model2.fit(
    # #     X_train, y_train,
    # #     eval_set = [(X_valid, y_valid)],
    # #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(200)],
    # #     init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
    # # )
    # # model2.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    
    
    # fitted_models_lgb_dart.append(model2)
    # y_pred_valid = model2.predict_proba(X_valid)[:,1]
    # auc_score = roc_auc_score(y_valid, y_pred_valid)
    # cv_scores_lgb_dart.append(auc_score)
    # print()
    # print("分隔符")
    # print()
    # ===========================
    


    # ===============================
    # X_train[cat_cols] = X_train[cat_cols].astype("category")
    # X_valid[cat_cols] = X_valid[cat_cols].astype("category")
    # params = {
    #     "boosting_type": "rf",
    #     "objective": "binary",
    #     "metric": "auc",
    #     "max_depth": 10,  
    #     "learning_rate": 0.05,
    #     "n_estimators": 2000,  # rf与早停无关，与n_estimators有关
    #     # 则每棵树在构建时会随机选择 80% 的特征进行训练，剩下的 20% 特征将不参与训练，从而增加模型的泛化能力和稳定性
    #     "colsample_bytree": 0.8, 
    #     "colsample_bynode": 0.8, # 控制每个节点的特征采样比例
    #     "verbose": -1,
    #     "random_state": 42,
    #     "reg_alpha": 0.1,
    #     "reg_lambda": 10,
    #     "extra_trees":True,
    #     'num_leaves':64,
    #     "device": 'gpu', # gpu
    #     'gpu_use_dp' : True # 转化float为64精度
    # }

    # # 一次训练
    # model = lgb.LGBMClassifier(**params)
    # model.fit(
    #     X_train, y_train,
    #     eval_set = [(X_valid, y_valid)],
    #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
    #     # init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
    # )
    # model2 = model
    # # model.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    

    # # # 二次优化
    # # # params['learning_rate'] = 0.01
    # # # model2 = lgb.LGBMClassifier(**params)
    # # # model2.fit(
    # # #     X_train, y_train,
    # # #     eval_set = [(X_valid, y_valid)],
    # # #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(200)],
    # # #     init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
    # # # )
    # # # model2.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    
    
    # fitted_models_lgb_rf.append(model2)
    # y_pred_valid = model2.predict_proba(X_valid)[:,1]
    # auc_score = roc_auc_score(y_valid, y_pred_valid)
    # cv_scores_lgb_rf.append(auc_score)
    # print()
    # print("分隔符")
    # print()
    # ===========================

    # ===========================

    # # train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    # # val_pool = Pool(X_valid, y_valid, cat_features=cat_cols)
    # # clf = CatBoostClassifier()
    # # # clf.load_model(f"/home/xyli/kaggle/kaggle_HomeCredit/catboost_dw_fold{fold}.cbm")
    # # clf.load_model(f"/home/xyli/kaggle/kaggle_HomeCredit/catboost_lg_fold{fold}.cbm")
    # # y_pred_valid = clf.predict_proba(X_valid)[:,1]
    # # auc_score = roc_auc_score(y_valid, y_pred_valid)
    # # print('auc_score: ', auc_score)



    # X_train[cat_cols] = X_train[cat_cols].astype("category")
    # X_valid[cat_cols] = X_valid[cat_cols].astype("category")
    # model = lgb.LGBMClassifier()
    # model = lgb.Booster(model_file=f"/home/xyli/kaggle/kaggle_HomeCredit/lgbm_dart_fold{fold}.txt")
    # y_pred_valid = model.predict(X_valid)
    # auc_score = roc_auc_score(y_valid, y_pred_valid)
    # print('auc_score: ', auc_score)
    # ===========================


    fold = fold+1

print("CV AUC scores: ", cv_scores_cat)
print("Mean CV AUC score: ", np.mean(cv_scores_cat))

print("CV AUC scores: ", cv_scores_lgb)
print("Mean CV AUC score: ", np.mean(cv_scores_lgb))

print("CV AUC scores: ", cv_scores_xgb)
print("Mean CV AUC score: ", np.mean(cv_scores_xgb))

print("CV AUC scores: ", cv_scores_cat_dw)
print("Mean CV AUC score: ", np.mean(cv_scores_cat_dw))

print("CV AUC scores: ", cv_scores_cat_lg)
print("Mean CV AUC score: ", np.mean(cv_scores_cat_lg))

print("CV AUC scores: ", cv_scores_lgb_dart)
print("Mean CV AUC score: ", np.mean(cv_scores_lgb_dart))

print("CV AUC scores: ", cv_scores_lgb_rf)
print("Mean CV AUC score: ", np.mean(cv_scores_lgb_rf))

# ======================================== 训练3树模型 =====================================









