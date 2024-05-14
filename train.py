
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

# ======================================== 读入df_train =====================================

# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train419.csv', nrows=5)
# df_train = pd.read_csv('/home/xyli/kaggle/train.csv', nrows=50001)
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train832.csv')
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train832.csv', nrows=50000)
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train389FE.csv')
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train419.csv')
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train389.csv')





class Pipeline:

    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df



    def handle_dates2(df):
        for col in df.columns:
            if col[-1] in ("D",):
                # 可能默认替换表达式中第1个列名吧
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
                df = df.with_columns(pl.col(col).dt.total_days()) # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def handle_dates(df:pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:  
            if col.endswith('D'):
                df = df.with_columns(pl.col(col) - pl.col('date_decision'))
                df = df.with_columns(pl.col(col).dt.total_days().cast(pl.Int32))

        df = df.with_columns([pl.col('date_decision').dt.year().alias('year').cast(pl.Int16), pl.col('date_decision').dt.month().alias('month').cast(pl.UInt8), pl.col('date_decision').dt.weekday().alias('week_num').cast(pl.UInt8)])

        return df.drop('date_decision', 'MONTH', 'WEEK_NUM')
        # return df.drop('date_decision', 'MONTH')

    def filter_cols2(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
#                 if isnull > 0.9:
                # if isnull == 1:
#                 if isnull > 0.99:
                    df = df.drop(col)
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                # if freq > 200:
#                 if (freq == 1) | (freq > 400):
#                 if (freq == 1):
                    df = df.drop(col)
        
        return df
    
    def filter_cols(df:pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col not in ['case_id', 'year', 'month', 'week_num', 'target']:
                null_pct = df[col].is_null().mean()

                # if null_pct > 0.7:
                if null_pct > 0.95:
                # if null_pct == 1: # 839
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ['case_id', 'year', 'month', 'week_num', 'target']) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq > 200) | (freq == 1):
                    df = df.drop(col)

        return df

    
class Aggregator:
    #Please add or subtract features yourself, be aware that too many features will take up too much space.
    def num_expr(df):
        # P是逾期天数，A是借贷数目
        # 感觉1个均值即可
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的
        
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的
        
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]

        # my code
        expr_std = [pl.std(col).alias(f"std_{col}") for col in cols]
        expr_sum = [pl.sum(col).alias(f"sum_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        # expr_product = [pl.product(col).alias(f"product_{col}") for col in cols]

        # 0.754300 排列顺序
        # return expr_max + expr_min + expr_last + expr_first + expr_mean

        return expr_mean # Mean AUC=0.741610 433
        # return expr_max + expr_mean + expr_var # notebookv8
        # return expr_max +expr_last+expr_mean # 829+386 
        # return expr_max +expr_last+expr_mean+expr_var # 829+386 + notebookv8
        # return expr_max +expr_last+expr_mean+expr_min # 433+829+386 
    
    
    def date_expr(df):
        # D是借贷日期
        # 感觉1个均值就行
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的

        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的
        
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        # 0.754300 排列顺序
        # return  expr_max + expr_min  +  expr_last + expr_first + expr_mean

        return expr_mean # Mean AUC=0.741610 433
        # return expr_max + expr_mean + expr_var # notebookv8
        # return  expr_max +expr_last+expr_mean # 829+386
        # return  expr_max +expr_last+expr_mean+expr_var # 829+386+notebookv8 
        # return expr_max +expr_last+expr_mean +expr_min# 433+829+386 

    
    def str_expr(df):
        # M是地址编号
        # 1个人最多也就几个地址吧，感觉取2个就可以了
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的 
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的
        expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]

        # my code
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_mode = [pl.col(col).drop_nulls().mode().first().alias(f'mode_{col}') for col in cols]

        # 0.754300 排列顺序
        # return  expr_max + expr_min + expr_last + expr_first + expr_count
        
        return expr_max + expr_last + expr_first # Mean AUC=0.741610 433
        # return expr_max # notebookv8
        # return  expr_max +expr_last # 829+386
        # return expr_last + expr_first+expr_max # 829+386+433

    def other_expr(df):
        # T、L代表各种杂七杂八的信息
        # 这一块可做特征工程提分，但粗略的来说1个均值更合适
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的
        

        # my code 
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_std = [pl.std(col).alias(f"std_{col}") for col in cols]
        expr_sum = [pl.sum(col).alias(f"sum_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        # expr_product = [pl.product(col).alias(f"product_{col}") for col in cols]

        # 0.754300 排列顺序
        # return  expr_max + expr_min + expr_last + expr_first


        return expr_mean # Mean AUC=0.741610 433
        # return expr_max # notebookv8
        # return  expr_max +expr_last # 829+386
        # return  expr_max +expr_last # 829+386+notebookv8
        # return  expr_max +expr_last +expr_mean+expr_min # 829+386+433


    
    def count_expr(df):
        
        # 其他一个case_id对应多条信息的，由于不知道具体是啥意思，所以统计特征用mean是比较好的感觉
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols] 
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的

        # my code 
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        # 0.755666 排列顺序
        # return  expr_max + expr_min + expr_last + expr_first + expr_count

        return expr_mean # Mean AUC=0.741610 433
        # return  expr_max # notebookv8
        # return  expr_max +expr_last # 829+386
        # return  expr_max +expr_last # 829+386+notebookv8
        # return  expr_max +expr_last+expr_mean+expr_min # 829+386+433
    
    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs

def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1,2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df)) 
    return df

def read_files(regex_path, depth=None):
    chunks = []
    
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)
    
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df

def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base
        .with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
    )

    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    
    # for i, df in enumerate(depth_0):
    #     df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

    # df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df






ROOT            = Path("/home/xyli/kaggle")

TRAIN_DIR       = ROOT / "parquet_files" / "train"
TEST_DIR        = ROOT / "parquet_files" / "test"

print('开始读取数据!')

data_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
        read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_other_1.parquet", 1),
        read_file(TRAIN_DIR / "train_person_1.parquet", 1),
        read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
        read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
        read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),

        # 829+386
        read_file(TRAIN_DIR / "train_applprev_2.parquet", 2),
        read_file(TRAIN_DIR / "train_person_2.parquet", 2)
    ]
}

class SchemaGen:
    @staticmethod
    def change_dtypes(df:pl.LazyFrame) -> pl.LazyFrame:
        for col in df.columns:
            if col == 'case_id':
                df = df.with_columns(pl.col(col).cast(pl.UInt32).alias(col))
            elif col in ['WEEK_NUM', 'num_group1', 'num_group2']:
                df = df.with_columns(pl.col(col).cast(pl.UInt16).alias(col))
            elif col == 'date_decision' or col[-1] == 'D':
                df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
            elif col[-1] in ['P', 'A']:
                df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
            elif col[-1] in ('M',):
                    df = df.with_columns(pl.col(col).cast(pl.String));
        return df


    @staticmethod
    def scan_files(glob_path: str, depth: int = None) -> pl.LazyFrame:
        chunks: list[pl.LazyFrame] = []
        for path in glob(str(glob_path)):
            # 增加low_memory=True + rechunk=True 将会导致一些数据被局部打乱
            # 导致一些高质量数据被分配在5w个中
            # 使得数据分布更加均匀
            df: pl.LazyFrame = pl.scan_parquet(path, low_memory=True, rechunk=True).pipe(SchemaGen.change_dtypes)
            print(f'File {Path(path).stem} loaded into memory.')
            
            if depth in (1, 2):
                exprs: list[pl.Series] = Aggregator.get_exprs(df)
                df = df.group_by('case_id').agg(exprs)

                del exprs
                gc.collect()
                
            chunks.append(df)

        df: pl.LazyFrame = pl.concat(chunks, how='vertical_relaxed')
        
        del chunks
        gc.collect()
                
        df = df.unique(subset=['case_id'])
    
        return df
    
    
    @staticmethod
    def join_dataframes(df_base: pl.LazyFrame, depth_0: list[pl.LazyFrame], depth_1: list[pl.LazyFrame], depth_2: list[pl.LazyFrame]) -> pl.DataFrame:

        df_base = ( # 829+386
            df_base
            .with_columns(
                month_decision = pl.col("date_decision").dt.month(),
                weekday_decision = pl.col("date_decision").dt.weekday(),
            )
        )


        for (i, df) in enumerate(depth_0 + depth_1 + depth_2):
            df_base = df_base.join(df, how='left', on='case_id', suffix=f'_{i}')

        return df_base.collect()

class Utility:
    
    def reduce_memory_usage(df:pl.DataFrame, name) -> pl.DataFrame:
        print(f'Memory usage of dataframe \'{name}\' is {round(df.estimated_size("mb"), 2)} MB.')

        int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        float_types = [pl.Float32, pl.Float64]

        for col in df.columns:
            col_type = df[col].dtype
            if (col_type in int_types + float_types):
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min is not None and c_max is not None:
                    if col_type in int_types:
                        if c_min >= 0:
                            if c_min >= np.iinfo(np.uint8).min and c_max <= np.iinfo(np.uint8).max:
                                df = df.with_columns(df[col].cast(pl.UInt8))
                            elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                                df = df.with_columns(df[col].cast(pl.UInt16))
                            elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                                df = df.with_columns(df[col].cast(pl.UInt32))
                            elif c_min >= np.iinfo(np.uint64).min and c_max <= np.iinfo(np.uint64).max:
                                df = df.with_columns(df[col].cast(pl.UInt64))
                        else:
                            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                                df = df.with_columns(df[col].cast(pl.Int8))
                            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                                df = df.with_columns(df[col].cast(pl.Int16))
                            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                                df = df.with_columns(df[col].cast(pl.Int32))
                            elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                                df = df.with_columns(df[col].cast(pl.Int64))
                    elif col_type in float_types:
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df = df.with_columns(df[col].cast(pl.Float32))

        print(f'Memory usage of dataframe \'{name}\' became {round(df.estimated_size("mb"), 4)} MB.')

        return df


    def to_pandas(df:pl.DataFrame, cat_cols:list[str]=None) -> (pd.DataFrame, list[str]):
        df:pd.DataFrame = df.to_pandas()

        if cat_cols is None:
            cat_cols = list(df.select_dtypes('object').columns)

        df[cat_cols] = df[cat_cols].astype('str')

        return df, cat_cols


data_store:dict = {
    'df_base': SchemaGen.scan_files(TRAIN_DIR / 'train_base.parquet'),
    'depth_0': [
        SchemaGen.scan_files(TRAIN_DIR / 'train_static_cb_0.parquet'),
        SchemaGen.scan_files(TRAIN_DIR / 'train_static_0_*.parquet'),
    ],
    'depth_1': [
        SchemaGen.scan_files(TRAIN_DIR / 'train_applprev_1_*.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_tax_registry_a_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_tax_registry_b_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_tax_registry_c_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_a_1_*.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_b_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_other_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_person_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_deposit_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_debitcard_1.parquet', 1),
    ],
    'depth_2': [
        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_a_2_*.parquet', 2),
        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_b_2.parquet', 2),
   
        # 829+386
        SchemaGen.scan_files(TRAIN_DIR / 'train_applprev_2.parquet', 2), 
        SchemaGen.scan_files(TRAIN_DIR / 'train_person_2.parquet', 2), 
    ]
}
print('读取数据完毕！')


df_train = feature_eng(**data_store)
df_train:pl.LazyFrame = SchemaGen.join_dataframes(**data_store)\
.pipe(Pipeline.filter_cols).pipe(Pipeline.handle_dates).pipe(Utility.reduce_memory_usage, 'df_train')


# print("train data shape:\t", df_train.shape)

del data_store
gc.collect()

# df_train = df_train.pipe(Pipeline.filter_cols)
# df_train = df_train.pipe(Pipeline.handle_dates)


# print("train data shape:\t", df_train.shape)
df_train, cat_cols = to_pandas(df_train)

# df_train = reduce_mem_usage(df_train)
# df_train, cat_cols = to_pandas(df_train)
# df_train = reduce_mem_usage(df_train, 'df_train')
print("train data shape:\t", df_train.shape)




""" 可理解为相关性处理，去掉相关性大致相同的列 """ 

# nums=df_train.select_dtypes(exclude='category').columns
# from itertools import combinations, permutations
# #df_train=df_train[nums]
# # 计算nums列（数值列）是否是nan的一个对应掩码矩阵
# nans_df = df_train[nums].isna()
# nans_groups={}
# for col in nums:
#     # 统计每列是nan的个数
#     cur_group = nans_df[col].sum()
#     try: 
#         nans_groups[cur_group].append(col)
#     except: # 可默认永不执行
#         nans_groups[cur_group]=[col]
# del nans_df; x=gc.collect()

# def reduce_group(grps):
#     use = []
#     for g in grps:
#         mx = 0; vx = g[0]
#         for gg in g:
#             n = df_train[gg].nunique()
#             if n>mx:
#                 mx = n
#                 vx = gg
#             #print(str(gg)+'-'+str(n),', ',end='')
#         use.append(vx)
#         #print()
#     # print('Use these',use)
#     return use

# def group_columns_by_correlation(matrix, threshold=0.95):
#     # 计算列之间的相关性
#     correlation_matrix = matrix.corr()

#     # 分组列
#     groups = []
#     remaining_cols = list(matrix.columns)
#     while remaining_cols:
#         col = remaining_cols.pop(0)
#         group = [col]
#         correlated_cols = [col]
#         for c in remaining_cols:
#             if correlation_matrix.loc[col, c] >= threshold:
#                 group.append(c)
#                 correlated_cols.append(c)
#         groups.append(group)
#         remaining_cols = [c for c in remaining_cols if c not in correlated_cols]
    
#     return groups

# uses=[]
# for k,v in nans_groups.items():
#     if len(v)>1:
#             Vs = nans_groups[k] # 是按照每列nunique的个数来分组的
#             #cross_features=list(combinations(Vs, 2))
#             #make_corr(Vs)
#             grps= group_columns_by_correlation(df_train[Vs], threshold=0.8)
#             use=reduce_group(grps)
#             uses=uses+use
#             #make_corr(use)
#     else:
#         uses=uses+v
#     # print('####### NAN count =',k)
# print(uses)
# print(len(uses))
# # 选则[处理后数值列+非数值列]做最终列
# uses=uses+list(df_train.select_dtypes(include='category').columns)
# print(len(uses))
# df_train=df_train[uses]



print("train data shape:\t", df_train.shape)


# print('cat_cols: ', cat_cols)
# print('df_train.columns: ', list(df_train.columns))
# df_train.to_csv('/home/xyli/kaggle/kaggle_HomeCredit/train389FE.csv', index=False)
# import sys
# sys.exit() 

# df_train = df_train[:50000]

# ======================================== 读入df_train =====================================


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




# """ 确保df_train[cat_cols]中每列字典都有nan值 """
# new_row = pd.DataFrame([[np.nan] * len(df_train.columns)], columns=df_train.columns)
# # 将新行添加到DataFrame中
# df_train = pd.concat([df_train, new_row], ignore_index=True)



# sample = pd.read_csv("/kaggle/input/home-credit-credit-risk-model-stability/sample_submission.csv")
device='gpu'
#n_samples=200000
n_est=12000 # 6000
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
# weeks = df_train["week_num"]
try:
    # df_train= df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
    df_train= df_train.drop(columns=["target", "case_id"])
except:
    print("这个代码已经执行过1次了！")

# 具备分层和分组功能
cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
# cv = StratifiedGroupKFold(n_splits=5, shuffle=True)


# 找到除cat_cols列外的所有列
non_cat_cols = df_train.columns.difference(cat_cols) 
print('cat_cols:')
print('len(cat_cols):',len(cat_cols))
print(cat_cols)
print('df_train.columns')
print("len(list(df_train.columns)): ", len(list(df_train.columns)))
print(list(df_train.columns))


# ======================================== 特征列分类 =====================================

# 386个
df_train_386 = ['month_decision', 'weekday_decision', 'credamount_770A', 'applicationcnt_361L', 'applications30d_658L', 'applicationscnt_1086L', 'applicationscnt_464L', 'applicationscnt_867L', 'clientscnt_1022L', 'clientscnt_100L', 'clientscnt_1071L', 'clientscnt_1130L', 'clientscnt_157L', 'clientscnt_257L', 'clientscnt_304L', 'clientscnt_360L', 'clientscnt_493L', 'clientscnt_533L', 'clientscnt_887L', 'clientscnt_946L', 'deferredmnthsnum_166L', 'disbursedcredamount_1113A', 'downpmt_116A', 'homephncnt_628L', 'isbidproduct_1095L', 'mobilephncnt_593L', 'numactivecreds_622L', 'numactivecredschannel_414L', 'numactiverelcontr_750L', 'numcontrs3months_479L', 'numnotactivated_1143L', 'numpmtchanneldd_318L', 'numrejects9m_859L', 'sellerplacecnt_915L', 'max_mainoccupationinc_384A', 'max_birth_259D', 'max_num_group1_9', 'birthdate_574D', 'dateofbirth_337D', 'days180_256L', 'days30_165L', 'days360_512L', 'firstquarter_103L', 'fourthquarter_440L', 'secondquarter_766L', 'thirdquarter_1082L', 'max_debtoutstand_525A', 'max_debtoverdue_47A', 'max_refreshdate_3813885D', 'mean_refreshdate_3813885D', 'pmtscount_423L', 'pmtssum_45A', 'responsedate_1012D', 'responsedate_4527233D', 'actualdpdtolerance_344P', 'amtinstpaidbefduel24m_4187115A', 'numinstlswithdpd5_4187116L', 'annuitynextmonth_57A', 'currdebt_22A', 'currdebtcredtyperange_828A', 'numinstls_657L', 'totalsettled_863A', 'mindbddpdlast24m_3658935P', 'avgdbddpdlast3m_4187120P', 'mindbdtollast24m_4525191P', 'avgdpdtolclosure24_3658938P', 'avginstallast24m_3658937A', 'maxinstallast24m_3658928A', 'avgmaxdpdlast9m_3716943P', 'avgoutstandbalancel6m_4187114A', 'avgpmtlast12m_4525200A', 'cntincpaycont9m_3716944L', 'cntpmts24_3658933L', 'commnoinclast6m_3546845L', 'maxdpdfrom6mto36m_3546853P', 'datefirstoffer_1144D', 'datelastunpaid_3546854D', 'daysoverduetolerancedd_3976961L', 'numinsttopaygr_769L', 'dtlastpmtallstes_4499206D', 'eir_270L', 'firstclxcampaign_1125D', 'firstdatedue_489D', 'lastactivateddate_801D', 'lastapplicationdate_877D', 'mean_creationdate_885D', 'max_num_group1', 'last_num_group1', 'max_num_group2_14', 'last_num_group2_14', 'lastapprcredamount_781A', 'lastapprdate_640D', 'lastdelinqdate_224D', 'lastrejectcredamount_222A', 'lastrejectdate_50D', 'maininc_215A', 'mastercontrelectronic_519L', 'mastercontrexist_109L', 'maxannuity_159A', 'maxdebt4_972A', 'maxdpdlast24m_143P', 'maxdpdlast3m_392P', 'maxdpdtolerance_374P', 'maxdbddpdlast1m_3658939P', 'maxdbddpdtollast12m_3658940P', 'maxdbddpdtollast6m_4187119P', 'maxdpdinstldate_3546855D', 'maxdpdinstlnum_3546846P', 'maxlnamtstart6m_4525199A', 'maxoutstandbalancel12m_4187113A', 'numinstpaidearly_338L', 'numinstpaidearly5d_1087L', 'numinstpaidlate1d_3546852L', 'numincomingpmts_3546848L', 'numinstlsallpaid_934L', 'numinstlswithdpd10_728L', 'numinstlswithoutdpd_562L', 'numinstpaid_4499208L', 'numinstpaidearly3d_3546850L', 'numinstregularpaidest_4493210L', 'numinstpaidearly5dest_4493211L', 'sumoutstandtotalest_4493215A', 'numinstpaidlastcontr_4325080L', 'numinstregularpaid_973L', 'pctinstlsallpaidearl3d_427L', 'pctinstlsallpaidlate1d_3546856L', 'pctinstlsallpaidlat10d_839L', 'pctinstlsallpaidlate4d_3546849L', 'pctinstlsallpaidlate6d_3546844L', 'pmtnum_254L', 'posfpd10lastmonth_333P', 'posfpd30lastmonth_3976960P', 'posfstqpd30lastmonth_3976962P', 'price_1097A', 'sumoutstandtotal_3546847A', 'totaldebt_9A', 'mean_actualdpd_943P', 'max_annuity_853A', 'mean_annuity_853A', 'max_credacc_credlmt_575A', 'max_credamount_590A', 'max_downpmt_134A', 'mean_credacc_credlmt_575A', 'mean_credamount_590A', 'mean_downpmt_134A', 'max_currdebt_94A', 'mean_currdebt_94A', 'max_mainoccupationinc_437A', 'mean_mainoccupationinc_437A', 'mean_maxdpdtolerance_577P', 'max_outstandingdebt_522A', 'mean_outstandingdebt_522A', 'last_actualdpd_943P', 'last_annuity_853A', 'last_credacc_credlmt_575A', 'last_credamount_590A', 'last_downpmt_134A', 'last_currdebt_94A', 'last_mainoccupationinc_437A', 'last_maxdpdtolerance_577P', 'last_outstandingdebt_522A', 'max_approvaldate_319D', 'mean_approvaldate_319D', 'max_dateactivated_425D', 'mean_dateactivated_425D', 'max_dtlastpmt_581D', 'mean_dtlastpmt_581D', 'max_dtlastpmtallstes_3545839D', 'mean_dtlastpmtallstes_3545839D', 'max_employedfrom_700D', 'max_firstnonzeroinstldate_307D', 'mean_firstnonzeroinstldate_307D', 'last_approvaldate_319D', 'last_creationdate_885D', 'last_dateactivated_425D', 'last_dtlastpmtallstes_3545839D', 'last_employedfrom_700D', 'last_firstnonzeroinstldate_307D', 'max_byoccupationinc_3656910L', 'max_childnum_21L', 'max_pmtnum_8L', 'last_pmtnum_8L', 'max_pmtamount_36A', 'last_pmtamount_36A', 'max_processingdate_168D', 'last_processingdate_168D', 'max_num_group1_5', 'mean_credlmt_230A', 'mean_credlmt_935A', 'mean_pmts_dpd_1073P', 'max_dpdmaxdatemonth_89T', 'max_dpdmaxdateyear_596T', 'max_pmts_dpd_303P', 'mean_dpdmax_757P', 'max_dpdmaxdatemonth_442T', 'max_dpdmaxdateyear_896T', 'mean_pmts_dpd_303P', 'mean_instlamount_768A', 'mean_monthlyinstlamount_332A', 'max_monthlyinstlamount_674A', 'mean_monthlyinstlamount_674A', 'mean_outstandingamount_354A', 'mean_outstandingamount_362A', 'mean_overdueamount_31A', 'mean_overdueamount_659A', 'max_numberofoverdueinstls_725L', 'mean_overdueamountmax2_14A', 'mean_totaloutstanddebtvalue_39A', 'mean_dateofcredend_289D', 'mean_dateofcredstart_739D', 'max_lastupdate_1112D', 'mean_lastupdate_1112D', 'max_numberofcontrsvalue_258L', 'max_numberofoverdueinstlmax_1039L', 'max_overdueamountmaxdatemonth_365T', 'max_overdueamountmaxdateyear_2T', 'mean_pmts_overdue_1140A', 'max_pmts_month_158T', 'max_pmts_year_1139T', 'mean_overdueamountmax2_398A', 'max_dateofcredend_353D', 'max_dateofcredstart_181D', 'mean_dateofcredend_353D', 'max_numberofoverdueinstlmax_1151L', 'mean_overdueamountmax_35A', 'max_overdueamountmaxdatemonth_284T', 'max_overdueamountmaxdateyear_994T', 'mean_pmts_overdue_1152A', 'max_residualamount_488A', 'mean_residualamount_856A', 'max_totalamount_6A', 'mean_totalamount_6A', 'mean_totalamount_996A', 'mean_totaldebtoverduevalue_718A', 'mean_totaloutstanddebtvalue_668A', 'max_numberofcontrsvalue_358L', 'max_dateofrealrepmt_138D', 'mean_dateofrealrepmt_138D', 'max_lastupdate_388D', 'mean_lastupdate_388D', 'max_numberofoverdueinstlmaxdat_148D', 'mean_numberofoverdueinstlmaxdat_641D', 'mean_overdueamountmax2date_1002D', 'max_overdueamountmax2date_1142D', 'last_refreshdate_3813885D', 'max_nominalrate_281L', 'max_nominalrate_498L', 'max_numberofinstls_229L', 'max_numberofinstls_320L', 'max_numberofoutstandinstls_520L', 'max_numberofoutstandinstls_59L', 'max_numberofoverdueinstls_834L', 'max_periodicityofpmts_1102L', 'max_periodicityofpmts_837L', 'last_num_group1_6', 'last_mainoccupationinc_384A', 'last_birth_259D', 'max_empl_employedfrom_271D', 'last_personindex_1023L', 'last_persontype_1072L', 'max_collater_valueofguarantee_1124L', 'max_collater_valueofguarantee_876L', 'max_pmts_month_706T', 'max_pmts_year_507T', 'last_pmts_month_158T', 'last_pmts_year_1139T', 'last_pmts_month_706T', 'last_pmts_year_507T', 'max_num_group1_13', 'max_num_group2_13', 'last_num_group2_13', 'max_num_group1_15', 'max_num_group2_15', 'description_5085714M', 'education_1103M', 'education_88M', 'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L', 'credtype_322L', 'disbursementtype_67L', 'inittransactioncode_186L', 'lastapprcommoditycat_1041M', 'lastcancelreason_561M', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'twobodfilling_608L', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'last_cancelreason_3545846M', 'last_education_1138M', 'last_postype_4733339M', 'last_rejectreason_755M', 'last_rejectreasonclient_4145042M', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_status_219L', 'last_credtype_587L', 'last_familystate_726L', 'last_inittransactioncode_279L', 'last_isbidproduct_390L', 'last_status_219L', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'last_classificationofcontr_13M', 'last_classificationofcontr_400M', 'last_contractst_545M', 'last_contractst_964M', 'last_description_351M', 'last_financialinstitution_382M', 'last_financialinstitution_591M', 'last_purposeofcred_426M', 'last_purposeofcred_874M', 'last_subjectrole_182M', 'last_subjectrole_93M', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'last_education_927M', 'last_empladdr_district_926M', 'last_empladdr_zipcode_114M', 'last_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_incometype_1044T', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'last_contaddr_matchlist_1032L', 'last_contaddr_smempladdr_334L', 'last_incometype_1044T', 'last_relationshiptoclient_642T', 'last_role_1084L', 'last_safeguarantyflag_411L', 'last_sex_738L', 'last_type_25L', 'max_collater_typofvalofguarant_298M', 'max_collater_typofvalofguarant_407M', 'max_collaterals_typeofguarante_359M', 'max_collaterals_typeofguarante_669M', 'max_subjectroles_name_541M', 'max_subjectroles_name_838M', 'last_collater_typofvalofguarant_298M', 'last_collater_typofvalofguarant_407M', 'last_collaterals_typeofguarante_359M', 'last_collaterals_typeofguarante_669M', 'last_subjectroles_name_541M', 'last_subjectroles_name_838M', 'max_cacccardblochreas_147M', 'last_cacccardblochreas_147M', 'max_conts_type_509L', 'last_conts_type_509L', 'max_conts_role_79M', 'max_empls_economicalst_849M', 'max_empls_employer_name_740M', 'last_conts_role_79M', 'last_empls_economicalst_849M', 'last_empls_employer_name_740M']
# 113个
cat_cols_386 = ['description_5085714M', 'education_1103M', 'education_88M', 'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L', 'credtype_322L', 'disbursementtype_67L', 'inittransactioncode_186L', 'lastapprcommoditycat_1041M', 'lastcancelreason_561M', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'twobodfilling_608L', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'last_cancelreason_3545846M', 'last_education_1138M', 'last_postype_4733339M', 'last_rejectreason_755M', 'last_rejectreasonclient_4145042M', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_status_219L', 'last_credtype_587L', 'last_familystate_726L', 'last_inittransactioncode_279L', 'last_isbidproduct_390L', 'last_status_219L', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'last_classificationofcontr_13M', 'last_classificationofcontr_400M', 'last_contractst_545M', 'last_contractst_964M', 'last_description_351M', 'last_financialinstitution_382M', 'last_financialinstitution_591M', 'last_purposeofcred_426M', 'last_purposeofcred_874M', 'last_subjectrole_182M', 'last_subjectrole_93M', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'last_education_927M', 'last_empladdr_district_926M', 'last_empladdr_zipcode_114M', 'last_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_incometype_1044T', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'last_contaddr_matchlist_1032L', 'last_contaddr_smempladdr_334L', 'last_incometype_1044T', 'last_relationshiptoclient_642T', 'last_role_1084L', 'last_safeguarantyflag_411L', 'last_sex_738L', 'last_type_25L', 'max_collater_typofvalofguarant_298M', 'max_collater_typofvalofguarant_407M', 'max_collaterals_typeofguarante_359M', 'max_collaterals_typeofguarante_669M', 'max_subjectroles_name_541M', 'max_subjectroles_name_838M', 'last_collater_typofvalofguarant_298M', 'last_collater_typofvalofguarant_407M', 'last_collaterals_typeofguarante_359M', 'last_collaterals_typeofguarante_669M', 'last_subjectroles_name_541M', 'last_subjectroles_name_838M', 'max_cacccardblochreas_147M', 'last_cacccardblochreas_147M', 'max_conts_type_509L', 'last_conts_type_509L', 'max_conts_role_79M', 'max_empls_economicalst_849M', 'max_empls_employer_name_740M', 'last_conts_role_79M', 'last_empls_economicalst_849M', 'last_empls_employer_name_740M']
# 273个
non_cat_cols_386 = [i for i in df_train_386 if i not in cat_cols_386]
print('len(cat_cols_386) : ', len(cat_cols_386))
print('len(non_cat_cols_386): ', len(non_cat_cols_386))


# 829个
df_train_829 = ['month_decision', 'weekday_decision', 'assignmentdate_238D', 'assignmentdate_4527235D', 'assignmentdate_4955616D', 'birthdate_574D', 'contractssum_5085716L', 'dateofbirth_337D', 'dateofbirth_342D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'description_5085714M', 'education_1103M', 'education_88M', 'firstquarter_103L', 'for3years_128L', 'for3years_504L', 'for3years_584L', 'formonth_118L', 'formonth_206L', 'formonth_535L', 'forquarter_1017L', 'forquarter_462L', 'forquarter_634L', 'fortoday_1092L', 'forweek_1077L', 'forweek_528L', 'forweek_601L', 'foryear_618L', 'foryear_818L', 'foryear_850L', 'fourthquarter_440L', 'maritalst_385M', 'maritalst_893M', 'numberofqueries_373L', 'pmtaverage_3A', 'pmtaverage_4527227A', 'pmtaverage_4955615A', 'pmtcount_4527229L', 'pmtcount_4955617L', 'pmtcount_693L', 'pmtscount_423L', 'pmtssum_45A', 'requesttype_4525192L', 'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D', 'riskassesment_302T', 'riskassesment_940T', 'secondquarter_766L', 'thirdquarter_1082L', 'actualdpdtolerance_344P', 'amtinstpaidbefduel24m_4187115A', 'annuity_780A', 'annuitynextmonth_57A', 'applicationcnt_361L', 'applications30d_658L', 'applicationscnt_1086L', 'applicationscnt_464L', 'applicationscnt_629L', 'applicationscnt_867L', 'avgdbddpdlast24m_3658932P', 'avgdbddpdlast3m_4187120P', 'avgdbdtollast24m_4525197P', 'avgdpdtolclosure24_3658938P', 'avginstallast24m_3658937A', 'avglnamtstart24m_4525187A', 'avgmaxdpdlast9m_3716943P', 'avgoutstandbalancel6m_4187114A', 'avgpmtlast12m_4525200A', 'bankacctype_710L', 'cardtype_51L', 'clientscnt12m_3712952L', 'clientscnt3m_3712950L', 'clientscnt6m_3712949L', 'clientscnt_100L', 'clientscnt_1022L', 'clientscnt_1071L', 'clientscnt_1130L', 'clientscnt_136L', 'clientscnt_157L', 'clientscnt_257L', 'clientscnt_304L', 'clientscnt_360L', 'clientscnt_493L', 'clientscnt_533L', 'clientscnt_887L', 'clientscnt_946L', 'cntincpaycont9m_3716944L', 'cntpmts24_3658933L', 'commnoinclast6m_3546845L', 'credamount_770A', 'credtype_322L', 'currdebt_22A', 'currdebtcredtyperange_828A', 'datefirstoffer_1144D', 'datelastinstal40dpd_247D', 'datelastunpaid_3546854D', 'daysoverduetolerancedd_3976961L', 'deferredmnthsnum_166L', 'disbursedcredamount_1113A', 'disbursementtype_67L', 'downpmt_116A', 'dtlastpmtallstes_4499206D', 'eir_270L', 'equalitydataagreement_891L', 'equalityempfrom_62L', 'firstclxcampaign_1125D', 'firstdatedue_489D', 'homephncnt_628L', 'inittransactionamount_650A', 'inittransactioncode_186L', 'interestrate_311L', 'interestrategrace_34L', 'isbidproduct_1095L', 'isbidproductrequest_292L', 'isdebitcard_729L', 'lastactivateddate_801D', 'lastapplicationdate_877D', 'lastapprcommoditycat_1041M', 'lastapprcredamount_781A', 'lastapprdate_640D', 'lastcancelreason_561M', 'lastdelinqdate_224D', 'lastdependentsnum_448L', 'lastotherinc_902A', 'lastotherlnsexpense_631A', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectcredamount_222A', 'lastrejectdate_50D', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastrepayingdate_696D', 'lastst_736L', 'maininc_215A', 'mastercontrelectronic_519L', 'mastercontrexist_109L', 'maxannuity_159A', 'maxannuity_4075009A', 'maxdbddpdlast1m_3658939P', 'maxdbddpdtollast12m_3658940P', 'maxdbddpdtollast6m_4187119P', 'maxdebt4_972A', 'maxdpdfrom6mto36m_3546853P', 'maxdpdinstldate_3546855D', 'maxdpdinstlnum_3546846P', 'maxdpdlast12m_727P', 'maxdpdlast24m_143P', 'maxdpdlast3m_392P', 'maxdpdlast6m_474P', 'maxdpdlast9m_1059P', 'maxdpdtolerance_374P', 'maxinstallast24m_3658928A', 'maxlnamtstart6m_4525199A', 'maxoutstandbalancel12m_4187113A', 'maxpmtlast3m_4525190A', 'mindbddpdlast24m_3658935P', 'mindbdtollast24m_4525191P', 'mobilephncnt_593L', 'monthsannuity_845L', 'numactivecreds_622L', 'numactivecredschannel_414L', 'numactiverelcontr_750L', 'numcontrs3months_479L', 'numincomingpmts_3546848L', 'numinstlallpaidearly3d_817L', 'numinstls_657L', 'numinstlsallpaid_934L', 'numinstlswithdpd10_728L', 'numinstlswithdpd5_4187116L', 'numinstlswithoutdpd_562L', 'numinstmatpaidtearly2d_4499204L', 'numinstpaid_4499208L', 'numinstpaidearly3d_3546850L', 'numinstpaidearly3dest_4493216L', 'numinstpaidearly5d_1087L', 'numinstpaidearly5dest_4493211L', 'numinstpaidearly5dobd_4499205L', 'numinstpaidearly_338L', 'numinstpaidearlyest_4493214L', 'numinstpaidlastcontr_4325080L', 'numinstpaidlate1d_3546852L', 'numinstregularpaid_973L', 'numinstregularpaidest_4493210L', 'numinsttopaygr_769L', 'numinsttopaygrest_4493213L', 'numinstunpaidmax_3546851L', 'numinstunpaidmaxest_4493212L', 'numnotactivated_1143L', 'numpmtchanneldd_318L', 'numrejects9m_859L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'payvacationpostpone_4187118D', 'pctinstlsallpaidearl3d_427L', 'pctinstlsallpaidlat10d_839L', 'pctinstlsallpaidlate1d_3546856L', 'pctinstlsallpaidlate4d_3546849L', 'pctinstlsallpaidlate6d_3546844L', 'pmtnum_254L', 'posfpd10lastmonth_333P', 'posfpd30lastmonth_3976960P', 'posfstqpd30lastmonth_3976962P', 'price_1097A', 'sellerplacecnt_915L', 'sellerplacescnt_216L', 'sumoutstandtotal_3546847A', 'sumoutstandtotalest_4493215A', 'totaldebt_9A', 'totalsettled_863A', 'totinstallast1m_4525188A', 'twobodfilling_608L', 'typesuite_864L', 'validfrom_1069D', 'max_actualdpd_943P', 'max_annuity_853A', 'max_credacc_actualbalance_314A', 'max_credacc_credlmt_575A', 'max_credacc_maxhisbal_375A', 'max_credacc_minhisbal_90A', 'max_credamount_590A', 'max_currdebt_94A', 'max_downpmt_134A', 'max_mainoccupationinc_437A', 'max_maxdpdtolerance_577P', 'max_outstandingdebt_522A', 'max_revolvingaccount_394A', 'last_actualdpd_943P', 'last_annuity_853A', 'last_credacc_actualbalance_314A', 'last_credacc_credlmt_575A', 'last_credacc_maxhisbal_375A', 'last_credacc_minhisbal_90A', 'last_credamount_590A', 'last_currdebt_94A', 'last_downpmt_134A', 'last_mainoccupationinc_437A', 'last_maxdpdtolerance_577P', 'last_outstandingdebt_522A', 'last_revolvingaccount_394A', 'mean_actualdpd_943P', 'mean_annuity_853A', 'mean_credacc_actualbalance_314A', 'mean_credacc_credlmt_575A', 'mean_credacc_maxhisbal_375A', 'mean_credacc_minhisbal_90A', 'mean_credamount_590A', 'mean_currdebt_94A', 'mean_downpmt_134A', 'mean_mainoccupationinc_437A', 'mean_maxdpdtolerance_577P', 'mean_outstandingdebt_522A', 'mean_revolvingaccount_394A', 'max_approvaldate_319D', 'max_creationdate_885D', 'max_dateactivated_425D', 'max_dtlastpmt_581D', 'max_dtlastpmtallstes_3545839D', 'max_employedfrom_700D', 'max_firstnonzeroinstldate_307D', 'last_approvaldate_319D', 'last_creationdate_885D', 'last_dateactivated_425D', 'last_dtlastpmt_581D', 'last_dtlastpmtallstes_3545839D', 'last_employedfrom_700D', 'last_firstnonzeroinstldate_307D', 'mean_approvaldate_319D', 'mean_creationdate_885D', 'mean_dateactivated_425D', 'mean_dtlastpmt_581D', 'mean_dtlastpmtallstes_3545839D', 'mean_employedfrom_700D', 'mean_firstnonzeroinstldate_307D', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'last_cancelreason_3545846M', 'last_education_1138M', 'last_postype_4733339M', 'last_rejectreason_755M', 'last_rejectreasonclient_4145042M', 'max_byoccupationinc_3656910L', 'max_childnum_21L', 'max_credacc_status_367L', 'max_credacc_transactions_402L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_pmtnum_8L', 'max_status_219L', 'max_tenor_203L', 'last_byoccupationinc_3656910L', 'last_childnum_21L', 'last_credacc_status_367L', 'last_credacc_transactions_402L', 'last_credtype_587L', 'last_familystate_726L', 'last_inittransactioncode_279L', 'last_isbidproduct_390L', 'last_isdebitcard_527L', 'last_pmtnum_8L', 'last_status_219L', 'last_tenor_203L', 'max_num_group1', 'last_num_group1', 'max_amount_4527230A', 'last_amount_4527230A', 'mean_amount_4527230A', 'max_recorddate_4527225D', 'last_recorddate_4527225D', 'mean_recorddate_4527225D', 'max_num_group1_3', 'last_num_group1_3', 'max_amount_4917619A', 'last_amount_4917619A', 'mean_amount_4917619A', 'max_deductiondate_4917603D', 'last_deductiondate_4917603D', 'mean_deductiondate_4917603D', 'max_num_group1_4', 'last_num_group1_4', 'max_pmtamount_36A', 'last_pmtamount_36A', 'mean_pmtamount_36A', 'max_processingdate_168D', 'last_processingdate_168D', 'mean_processingdate_168D', 'max_num_group1_5', 'last_num_group1_5', 'max_credlmt_230A', 'max_credlmt_935A', 'max_debtoutstand_525A', 'max_debtoverdue_47A', 'max_dpdmax_139P', 'max_dpdmax_757P', 'max_instlamount_768A', 'max_instlamount_852A', 'max_monthlyinstlamount_332A', 'max_monthlyinstlamount_674A', 'max_outstandingamount_354A', 'max_outstandingamount_362A', 'max_overdueamount_31A', 'max_overdueamount_659A', 'max_overdueamountmax2_14A', 'max_overdueamountmax2_398A', 'max_overdueamountmax_155A', 'max_overdueamountmax_35A', 'max_residualamount_488A', 'max_residualamount_856A', 'max_totalamount_6A', 'max_totalamount_996A', 'max_totaldebtoverduevalue_178A', 'max_totaldebtoverduevalue_718A', 'max_totaloutstanddebtvalue_39A', 'max_totaloutstanddebtvalue_668A', 'last_credlmt_230A', 'last_credlmt_935A', 'last_debtoutstand_525A', 'last_debtoverdue_47A', 'last_dpdmax_139P', 'last_dpdmax_757P', 'last_instlamount_768A', 'last_instlamount_852A', 'last_monthlyinstlamount_332A', 'last_monthlyinstlamount_674A', 'last_outstandingamount_354A', 'last_outstandingamount_362A', 'last_overdueamount_31A', 'last_overdueamount_659A', 'last_overdueamountmax2_14A', 'last_overdueamountmax2_398A', 'last_overdueamountmax_155A', 'last_overdueamountmax_35A', 'last_residualamount_488A', 'last_residualamount_856A', 'last_totalamount_6A', 'last_totalamount_996A', 'last_totaldebtoverduevalue_178A', 'last_totaldebtoverduevalue_718A', 'last_totaloutstanddebtvalue_39A', 'last_totaloutstanddebtvalue_668A', 'mean_credlmt_230A', 'mean_credlmt_935A', 'mean_debtoutstand_525A', 'mean_debtoverdue_47A', 'mean_dpdmax_139P', 'mean_dpdmax_757P', 'mean_instlamount_768A', 'mean_instlamount_852A', 'mean_monthlyinstlamount_332A', 'mean_monthlyinstlamount_674A', 'mean_outstandingamount_354A', 'mean_outstandingamount_362A', 'mean_overdueamount_31A', 'mean_overdueamount_659A', 'mean_overdueamountmax2_14A', 'mean_overdueamountmax2_398A', 'mean_overdueamountmax_155A', 'mean_overdueamountmax_35A', 'mean_residualamount_488A', 'mean_residualamount_856A', 'mean_totalamount_6A', 'mean_totalamount_996A', 'mean_totaldebtoverduevalue_178A', 'mean_totaldebtoverduevalue_718A', 'mean_totaloutstanddebtvalue_39A', 'mean_totaloutstanddebtvalue_668A', 'max_dateofcredend_289D', 'max_dateofcredend_353D', 'max_dateofcredstart_181D', 'max_dateofcredstart_739D', 'max_dateofrealrepmt_138D', 'max_lastupdate_1112D', 'max_lastupdate_388D', 'max_numberofoverdueinstlmaxdat_148D', 'max_numberofoverdueinstlmaxdat_641D', 'max_overdueamountmax2date_1002D', 'max_overdueamountmax2date_1142D', 'max_refreshdate_3813885D', 'last_dateofcredend_289D', 'last_dateofcredend_353D', 'last_dateofcredstart_181D', 'last_dateofcredstart_739D', 'last_dateofrealrepmt_138D', 'last_lastupdate_1112D', 'last_lastupdate_388D', 'last_numberofoverdueinstlmaxdat_148D', 'last_numberofoverdueinstlmaxdat_641D', 'last_overdueamountmax2date_1002D', 'last_overdueamountmax2date_1142D', 'last_refreshdate_3813885D', 'mean_dateofcredend_289D', 'mean_dateofcredend_353D', 'mean_dateofcredstart_181D', 'mean_dateofcredstart_739D', 'mean_dateofrealrepmt_138D', 'mean_lastupdate_1112D', 'mean_lastupdate_388D', 'mean_numberofoverdueinstlmaxdat_148D', 'mean_numberofoverdueinstlmaxdat_641D', 'mean_overdueamountmax2date_1002D', 'mean_overdueamountmax2date_1142D', 'mean_refreshdate_3813885D', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'last_classificationofcontr_13M', 'last_classificationofcontr_400M', 'last_contractst_545M', 'last_contractst_964M', 'last_description_351M', 'last_financialinstitution_382M', 'last_financialinstitution_591M', 'last_purposeofcred_426M', 'last_purposeofcred_874M', 'last_subjectrole_182M', 'last_subjectrole_93M', 'max_annualeffectiverate_199L', 'max_annualeffectiverate_63L', 'max_contractsum_5085717L', 'max_dpdmaxdatemonth_442T', 'max_dpdmaxdatemonth_89T', 'max_dpdmaxdateyear_596T', 'max_dpdmaxdateyear_896T', 'max_interestrate_508L', 'max_nominalrate_281L', 'max_nominalrate_498L', 'max_numberofcontrsvalue_258L', 'max_numberofcontrsvalue_358L', 'max_numberofinstls_229L', 'max_numberofinstls_320L', 'max_numberofoutstandinstls_520L', 'max_numberofoutstandinstls_59L', 'max_numberofoverdueinstlmax_1039L', 'max_numberofoverdueinstlmax_1151L', 'max_numberofoverdueinstls_725L', 'max_numberofoverdueinstls_834L', 'max_overdueamountmaxdatemonth_284T', 'max_overdueamountmaxdatemonth_365T', 'max_overdueamountmaxdateyear_2T', 'max_overdueamountmaxdateyear_994T', 'max_periodicityofpmts_1102L', 'max_periodicityofpmts_837L', 'max_prolongationcount_1120L', 'max_prolongationcount_599L', 'last_annualeffectiverate_199L', 'last_contractsum_5085717L', 'last_dpdmaxdatemonth_442T', 'last_dpdmaxdatemonth_89T', 'last_dpdmaxdateyear_596T', 'last_dpdmaxdateyear_896T', 'last_interestrate_508L', 'last_nominalrate_281L', 'last_nominalrate_498L', 'last_numberofcontrsvalue_258L', 'last_numberofcontrsvalue_358L', 'last_numberofinstls_229L', 'last_numberofinstls_320L', 'last_numberofoutstandinstls_520L', 'last_numberofoutstandinstls_59L', 'last_numberofoverdueinstlmax_1039L', 'last_numberofoverdueinstlmax_1151L', 'last_numberofoverdueinstls_725L', 'last_numberofoverdueinstls_834L', 'last_overdueamountmaxdatemonth_284T', 'last_overdueamountmaxdatemonth_365T', 'last_overdueamountmaxdateyear_2T', 'last_overdueamountmaxdateyear_994T', 'last_periodicityofpmts_1102L', 'last_periodicityofpmts_837L', 'last_prolongationcount_1120L', 'max_num_group1_6', 'last_num_group1_6', 'max_amount_1115A', 'max_credlmt_1052A', 'max_credlmt_228A', 'max_credlmt_3940954A', 'max_debtpastduevalue_732A', 'max_debtvalue_227A', 'max_dpd_550P', 'max_dpd_733P', 'max_dpdmax_851P', 'max_installmentamount_644A', 'max_installmentamount_833A', 'max_instlamount_892A', 'max_maxdebtpduevalodued_3940955A', 'max_overdueamountmax_950A', 'max_pmtdaysoverdue_1135P', 'max_residualamount_1093A', 'max_residualamount_127A', 'max_residualamount_3940956A', 'max_totalamount_503A', 'max_totalamount_881A', 'last_amount_1115A', 'last_credlmt_1052A', 'last_credlmt_228A', 'last_credlmt_3940954A', 'last_debtpastduevalue_732A', 'last_debtvalue_227A', 'last_dpd_550P', 'last_dpd_733P', 'last_dpdmax_851P', 'last_installmentamount_644A', 'last_installmentamount_833A', 'last_instlamount_892A', 'last_maxdebtpduevalodued_3940955A', 'last_overdueamountmax_950A', 'last_pmtdaysoverdue_1135P', 'last_residualamount_1093A', 'last_residualamount_127A', 'last_residualamount_3940956A', 'last_totalamount_503A', 'last_totalamount_881A', 'mean_amount_1115A', 'mean_credlmt_1052A', 'mean_credlmt_228A', 'mean_credlmt_3940954A', 'mean_debtpastduevalue_732A', 'mean_debtvalue_227A', 'mean_dpd_550P', 'mean_dpd_733P', 'mean_dpdmax_851P', 'mean_installmentamount_644A', 'mean_installmentamount_833A', 'mean_instlamount_892A', 'mean_maxdebtpduevalodued_3940955A', 'mean_overdueamountmax_950A', 'mean_pmtdaysoverdue_1135P', 'mean_residualamount_1093A', 'mean_residualamount_127A', 'mean_residualamount_3940956A', 'mean_totalamount_503A', 'mean_totalamount_881A', 'max_contractdate_551D', 'max_contractmaturitydate_151D', 'max_lastupdate_260D', 'last_contractdate_551D', 'last_contractmaturitydate_151D', 'last_lastupdate_260D', 'mean_contractdate_551D', 'mean_contractmaturitydate_151D', 'mean_lastupdate_260D', 'max_classificationofcontr_1114M', 'max_contractst_516M', 'max_contracttype_653M', 'max_credor_3940957M', 'max_periodicityofpmts_997M', 'max_pmtmethod_731M', 'max_purposeofcred_722M', 'max_subjectrole_326M', 'max_subjectrole_43M', 'last_classificationofcontr_1114M', 'last_contractst_516M', 'last_contracttype_653M', 'last_credor_3940957M', 'last_periodicityofpmts_997M', 'last_pmtmethod_731M', 'last_purposeofcred_722M', 'last_subjectrole_326M', 'last_subjectrole_43M', 'max_credquantity_1099L', 'max_credquantity_984L', 'max_dpdmaxdatemonth_804T', 'max_dpdmaxdateyear_742T', 'max_interesteffectiverate_369L', 'max_interestrateyearly_538L', 'max_numberofinstls_810L', 'max_overdueamountmaxdatemonth_494T', 'max_overdueamountmaxdateyear_432T', 'max_periodicityofpmts_997L', 'max_pmtnumpending_403L', 'last_credquantity_1099L', 'last_credquantity_984L', 'last_dpdmaxdatemonth_804T', 'last_dpdmaxdateyear_742T', 'last_interesteffectiverate_369L', 'last_interestrateyearly_538L', 'last_numberofinstls_810L', 'last_overdueamountmaxdatemonth_494T', 'last_overdueamountmaxdateyear_432T', 'last_periodicityofpmts_997L', 'last_pmtnumpending_403L', 'max_num_group1_7', 'last_num_group1_7', 'max_amtdebitincoming_4809443A', 'max_amtdebitoutgoing_4809440A', 'max_amtdepositbalance_4809441A', 'max_amtdepositincoming_4809444A', 'max_amtdepositoutgoing_4809442A', 'last_amtdebitincoming_4809443A', 'last_amtdebitoutgoing_4809440A', 'last_amtdepositbalance_4809441A', 'last_amtdepositincoming_4809444A', 'last_amtdepositoutgoing_4809442A', 'mean_amtdebitincoming_4809443A', 'mean_amtdebitoutgoing_4809440A', 'mean_amtdepositbalance_4809441A', 'mean_amtdepositincoming_4809444A', 'mean_amtdepositoutgoing_4809442A', 'max_num_group1_8', 'last_num_group1_8', 'max_mainoccupationinc_384A', 'last_mainoccupationinc_384A', 'mean_mainoccupationinc_384A', 'max_birth_259D', 'max_birthdate_87D', 'max_empl_employedfrom_271D', 'last_birth_259D', 'last_birthdate_87D', 'last_empl_employedfrom_271D', 'mean_birth_259D', 'mean_birthdate_87D', 'mean_empl_employedfrom_271D', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'last_education_927M', 'last_empladdr_district_926M', 'last_empladdr_zipcode_114M', 'last_language1_981M', 'max_childnum_185L', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_gender_992L', 'max_housetype_905L', 'max_housingtype_772L', 'max_incometype_1044T', 'max_isreference_387L', 'max_maritalst_703L', 'max_personindex_1023L', 'max_persontype_1072L', 'max_persontype_792L', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_role_993L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'last_contaddr_matchlist_1032L', 'last_contaddr_smempladdr_334L', 'last_empl_employedtotal_800L', 'last_empl_industry_691L', 'last_familystate_447L', 'last_gender_992L', 'last_housetype_905L', 'last_incometype_1044T', 'last_isreference_387L', 'last_maritalst_703L', 'last_personindex_1023L', 'last_persontype_1072L', 'last_persontype_792L', 'last_relationshiptoclient_415T', 'last_relationshiptoclient_642T', 'last_remitter_829L', 'last_role_1084L', 'last_role_993L', 'last_safeguarantyflag_411L', 'last_sex_738L', 'last_type_25L', 'max_num_group1_9', 'last_num_group1_9', 'max_amount_416A', 'last_amount_416A', 'mean_amount_416A', 'max_contractenddate_991D', 'max_openingdate_313D', 'last_contractenddate_991D', 'last_openingdate_313D', 'mean_contractenddate_991D', 'mean_openingdate_313D', 'max_num_group1_10', 'last_num_group1_10', 'max_last180dayaveragebalance_704A', 'max_last180dayturnover_1134A', 'max_last30dayturnover_651A', 'last_last180dayaveragebalance_704A', 'last_last180dayturnover_1134A', 'last_last30dayturnover_651A', 'mean_last180dayaveragebalance_704A', 'mean_last180dayturnover_1134A', 'mean_last30dayturnover_651A', 'max_openingdate_857D', 'last_openingdate_857D', 'mean_openingdate_857D', 'max_num_group1_11', 'last_num_group1_11', 'max_pmts_dpdvalue_108P', 'max_pmts_pmtsoverdue_635A', 'last_pmts_dpdvalue_108P', 'last_pmts_pmtsoverdue_635A', 'mean_pmts_dpdvalue_108P', 'mean_pmts_pmtsoverdue_635A', 'max_pmts_date_1107D', 'last_pmts_date_1107D', 'mean_pmts_date_1107D', 'max_num_group1_12', 'max_num_group2', 'last_num_group1_12', 'last_num_group2', 'max_pmts_dpd_1073P', 'max_pmts_dpd_303P', 'max_pmts_overdue_1140A', 'max_pmts_overdue_1152A', 'last_pmts_dpd_1073P', 'last_pmts_dpd_303P', 'last_pmts_overdue_1140A', 'last_pmts_overdue_1152A', 'mean_pmts_dpd_1073P', 'mean_pmts_dpd_303P', 'mean_pmts_overdue_1140A', 'mean_pmts_overdue_1152A', 'max_collater_typofvalofguarant_298M', 'max_collater_typofvalofguarant_407M', 'max_collaterals_typeofguarante_359M', 'max_collaterals_typeofguarante_669M', 'max_subjectroles_name_541M', 'max_subjectroles_name_838M', 'last_collater_typofvalofguarant_298M', 'last_collater_typofvalofguarant_407M', 'last_collaterals_typeofguarante_359M', 'last_collaterals_typeofguarante_669M', 'last_subjectroles_name_541M', 'last_subjectroles_name_838M', 'max_collater_valueofguarantee_1124L', 'max_collater_valueofguarantee_876L', 'max_pmts_month_158T', 'max_pmts_month_706T', 'max_pmts_year_1139T', 'max_pmts_year_507T', 'last_collater_valueofguarantee_1124L', 'last_collater_valueofguarantee_876L', 'last_pmts_month_158T', 'last_pmts_month_706T', 'last_pmts_year_1139T', 'last_pmts_year_507T', 'max_num_group1_13', 'max_num_group2_13', 'last_num_group1_13', 'last_num_group2_13', 'max_cacccardblochreas_147M', 'last_cacccardblochreas_147M', 'max_conts_type_509L', 'max_credacc_cards_status_52L', 'last_conts_type_509L', 'last_credacc_cards_status_52L', 'max_num_group1_14', 'max_num_group2_14', 'last_num_group1_14', 'last_num_group2_14', 'max_empls_employedfrom_796D', 'mean_empls_employedfrom_796D', 'max_conts_role_79M', 'max_empls_economicalst_849M', 'max_empls_employer_name_740M', 'last_conts_role_79M', 'last_empls_economicalst_849M', 'last_empls_employer_name_740M', 'max_addres_role_871L', 'max_relatedpersons_role_762T', 'last_addres_role_871L', 'last_relatedpersons_role_762T', 'max_num_group1_15', 'max_num_group2_15', 'last_num_group1_15', 'last_num_group2_15']
# 167个
cat_cols_829 = ['description_5085714M', 'education_1103M', 'education_88M', 'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L', 'riskassesment_302T', 'bankacctype_710L', 'cardtype_51L', 'credtype_322L', 'disbursementtype_67L', 'equalitydataagreement_891L', 'equalityempfrom_62L', 'inittransactioncode_186L', 'isbidproductrequest_292L', 'isdebitcard_729L', 'lastapprcommoditycat_1041M', 'lastcancelreason_561M', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'twobodfilling_608L', 'typesuite_864L', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'last_cancelreason_3545846M', 'last_education_1138M', 'last_postype_4733339M', 'last_rejectreason_755M', 'last_rejectreasonclient_4145042M', 'max_credacc_status_367L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_status_219L', 'last_credacc_status_367L', 'last_credtype_587L', 'last_familystate_726L', 'last_inittransactioncode_279L', 'last_isbidproduct_390L', 'last_isdebitcard_527L', 'last_status_219L', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'last_classificationofcontr_13M', 'last_classificationofcontr_400M', 'last_contractst_545M', 'last_contractst_964M', 'last_description_351M', 'last_financialinstitution_382M', 'last_financialinstitution_591M', 'last_purposeofcred_426M', 'last_purposeofcred_874M', 'last_subjectrole_182M', 'last_subjectrole_93M', 'max_classificationofcontr_1114M', 'max_contractst_516M', 'max_contracttype_653M', 'max_credor_3940957M', 'max_periodicityofpmts_997M', 'max_pmtmethod_731M', 'max_purposeofcred_722M', 'max_subjectrole_326M', 'max_subjectrole_43M', 'last_classificationofcontr_1114M', 'last_contractst_516M', 'last_contracttype_653M', 'last_credor_3940957M', 'last_periodicityofpmts_997M', 'last_pmtmethod_731M', 'last_purposeofcred_722M', 'last_subjectrole_326M', 'last_subjectrole_43M', 'max_periodicityofpmts_997L', 'last_periodicityofpmts_997L', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'last_education_927M', 'last_empladdr_district_926M', 'last_empladdr_zipcode_114M', 'last_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_gender_992L', 'max_housetype_905L', 'max_housingtype_772L', 'max_incometype_1044T', 'max_isreference_387L', 'max_maritalst_703L', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_role_993L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'last_contaddr_matchlist_1032L', 'last_contaddr_smempladdr_334L', 'last_empl_employedtotal_800L', 'last_empl_industry_691L', 'last_familystate_447L', 'last_gender_992L', 'last_housetype_905L', 'last_incometype_1044T', 'last_isreference_387L', 'last_maritalst_703L', 'last_relationshiptoclient_415T', 'last_relationshiptoclient_642T', 'last_remitter_829L', 'last_role_1084L', 'last_role_993L', 'last_safeguarantyflag_411L', 'last_sex_738L', 'last_type_25L', 'max_collater_typofvalofguarant_298M', 'max_collater_typofvalofguarant_407M', 'max_collaterals_typeofguarante_359M', 'max_collaterals_typeofguarante_669M', 'max_subjectroles_name_541M', 'max_subjectroles_name_838M', 'last_collater_typofvalofguarant_298M', 'last_collater_typofvalofguarant_407M', 'last_collaterals_typeofguarante_359M', 'last_collaterals_typeofguarante_669M', 'last_subjectroles_name_541M', 'last_subjectroles_name_838M', 'max_cacccardblochreas_147M', 'last_cacccardblochreas_147M', 'max_conts_type_509L', 'max_credacc_cards_status_52L', 'last_conts_type_509L', 'last_credacc_cards_status_52L', 'max_conts_role_79M', 'max_empls_economicalst_849M', 'max_empls_employer_name_740M', 'last_conts_role_79M', 'last_empls_economicalst_849M', 'last_empls_employer_name_740M', 'max_addres_role_871L', 'max_relatedpersons_role_762T', 'last_addres_role_871L', 'last_relatedpersons_role_762T']
# 662个
non_cat_cols_829 = [i for i in df_train_829 if i not in cat_cols_829]
print('len(cat_cols_829) : ', len(cat_cols_829))
print('len(non_cat_cols_829): ', len(non_cat_cols_829))

df_train[cat_cols] = df_train[cat_cols].astype(str)

# ======================================== 特征列分类 =====================================

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


print(df_train.head())

fold = 1
for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

    # X_train(≈40000,386), y_train(≈40000)
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]    
    

    # ===============================
    X_train[cat_cols] = X_train[cat_cols].astype("category")
    X_valid[cat_cols] = X_valid[cat_cols].astype("category")

    params1 = {
        "boosting_type": "gbdt",
        "colsample_bynode": 0.8,
        "colsample_bytree": 0.8,
        "device": device,
        "extra_trees": True,
        "learning_rate": 0.05,
        "l1_regularization": 0.1,
        "l2_regularization": 10,
        "max_depth": 20,
        "metric": "auc",
        "n_estimators": 2000,
        "num_leaves": 64,
        "objective": "binary",
        "random_state": 42,
        "verbose": -1,
    }
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
    #     'gpu_use_dp' : True, # 转化float为64精度

    #     # # 平衡类别之间的权重  损失函数不会因为样本不平衡而被“推向”样本量偏少的类别中
    #     # "sample_weight":'balanced',
    # }

    # 一次训练
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_valid, y_valid)],
        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
        # init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
    )
    model.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    model2 = model

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
    

    fitted_models_lgb.append(model2)
    y_pred_valid = model2.predict_proba(X_valid)[:,1]
    auc_score = roc_auc_score(y_valid, y_pred_valid)
    print('auc_score: ', auc_score)
    cv_scores_lgb.append(auc_score)
    print()
    print("分隔符")
    print()
    # ===========================


    # ======================================
    # train_pool = Pool(X_train, y_train,cat_features=cat_cols)
    # val_pool = Pool(X_valid, y_valid,cat_features=cat_cols)
    
# #     train_pool = Pool(X_train, y_train)
# #     val_pool = Pool(X_valid, y_valid)
     

    # clf = CatBoostClassifier( 
    #     best_model_min_trees = 1000,
    #     boosting_type = "Plain",
    #     eval_metric = "AUC",
    #     iterations = 6000,
    #     learning_rate = 0.05,
    #     l2_leaf_reg = 10,
    #     max_leaves = 64,
    #     random_seed = 42,
    #     task_type = "GPU",
    #     use_best_model = True
    # ) 

#     clf = CatBoostClassifier(
#         eval_metric='AUC',
#         task_type='GPU',
#         learning_rate=0.03, # 0.03
#         iterations=6000, # n_est
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
    
    # =================================

    fold = fold+1

# print("CV AUC scores: ", cv_scores_cat)
# print("Mean CV AUC score: ", np.mean(cv_scores_cat))

# print("CV AUC scores: ", cv_scores_lgb)
# print("Mean CV AUC score: ", np.mean(cv_scores_lgb))

# print("CV AUC scores: ", cv_scores_xgb)
# print("Mean CV AUC score: ", np.mean(cv_scores_xgb))

# print("CV AUC scores: ", cv_scores_cat_dw)
# print("Mean CV AUC score: ", np.mean(cv_scores_cat_dw))

# print("CV AUC scores: ", cv_scores_cat_lg)
# print("Mean CV AUC score: ", np.mean(cv_scores_cat_lg))

# print("CV AUC scores: ", cv_scores_lgb_dart)
# print("Mean CV AUC score: ", np.mean(cv_scores_lgb_dart))

# print("CV AUC scores: ", cv_scores_lgb_rf)
# print("Mean CV AUC score: ", np.mean(cv_scores_lgb_rf))

# ======================================== 训练3树模型 =====================================

# ======================================== 推理验证 =====================================
fitted_models_cat1 = []
fitted_models_lgb1 = []

fitted_models_cat2 = []
fitted_models_lgb2 = []

fitted_models_cat3 = []
fitted_models_lgb3 = []

for fold in range(1,6):
    clf = CatBoostClassifier() 
    clf.load_model(f"/home/xyli/kaggle/kaggle_HomeCredit/dataset9/catboost_fold{fold}.cbm")
    fitted_models_cat1.append(clf)
    
    model = lgb.LGBMClassifier()
    model = lgb.Booster(model_file=f"/home/xyli/kaggle/kaggle_HomeCredit/dataset8/lgbm_fold{fold}.txt")
    fitted_models_lgb1.append(model)
    
    clf2 = CatBoostClassifier()
    clf2.load_model(f"/home/xyli/kaggle/kaggle_HomeCredit/dataset5/catboost_fold{fold}.cbm")
    fitted_models_cat2.append(clf2) 
    
    model2 = lgb.LGBMClassifier()
    model2 = lgb.Booster(model_file=f"/home/xyli/kaggle/kaggle_HomeCredit/dataset4/lgbm_fold{fold}.txt")
    fitted_models_lgb2.append(model2)

    clf3 = CatBoostClassifier()
    clf3.load_model(f"/home/xyli/kaggle/kaggle_HomeCredit/dataset18/catboost_fold{fold}.cbm")
    fitted_models_cat3.append(clf3) 
    
    model3 = lgb.LGBMClassifier()
    model3 = lgb.Booster(model_file=f"/home/xyli/kaggle/kaggle_HomeCredit/dataset18/lgbm_fold{fold}.txt")
    fitted_models_lgb3.append(model3)


class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self

    def predict_proba(self, X, fold):
        fold = fold -1
        y_preds = []

        # from IPython import embed
        # embed()

        X[cat_cols] = X[cat_cols].astype("str")
        # y_preds += [estimator.predict_proba(X[df_train_829])[:, 1] for estimator in [self.estimators[0+fold]]]
        # y_preds += [estimator.predict_proba(X[df_train_386])[:, 1] for estimator in [self.estimators[5+fold]]]
        y_preds += [estimator.predict_proba(X[df_train])[:, 1] for estimator in [self.estimators[10+fold]]]
        
        X[cat_cols] = X[cat_cols].astype("category")
        # y_preds += [estimator.predict(X[df_train_829]) for estimator in [self.estimators[15+fold]]]
        # y_preds += [estimator.predict(X[df_train_386]) for estimator in [self.estimators[20+fold]]]
        y_preds += [estimator.predict(X[df_train]) for estimator in [self.estimators[25+fold]]]

        return np.mean(y_preds, axis=0)

model = VotingModel(fitted_models_cat1 + fitted_models_cat2 +fitted_models_cat3+ fitted_models_lgb1 + fitted_models_lgb2+fitted_models_lgb3)


fold = 1
for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

    # X_train(≈40000,386), y_train(≈40000)
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]    
    
    valid_preds = model.predict_proba(X_valid, fold) 
    valid_score = roc_auc_score(y_valid, valid_preds)
    print(f'fold:{fold} valid_score: ', valid_score)

    fold = fold+1

# ======================================== 推理验证 =====================================


