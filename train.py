
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

# 虽然块，但会导致识别类别很不准，需要手动调整类别
# df_train = pl.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train832.csv').to_pandas()

# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train832.csv') 
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train832.csv', nrows=50000)
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train389FE.csv')
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train419.csv')
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train.csv')





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



    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                # 可能默认替换表达式中第1个列名吧
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
                df = df.with_columns(pl.col(col).dt.total_days()) # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    

    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                # if isnull > 0.7:
                # if isnull > 0.95: # ZhiXing Jiang
                if isnull > 0.9: # kontsev
                # if isnull == 1:
#                 if isnull > 0.99:
                    df = df.drop(col)
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                # if (freq == 1) | (freq > 200):
                if (freq == 1) | (freq > 1000):
                # if freq > 200:
#                 if (freq == 1) | (freq > 400):
#                 if (freq == 1):
                    df = df.drop(col)
        
        return df
    
def handle_dates(df: pl.DataFrame) -> pl.DataFrame:
        """
        Handles date columns in the DataFrame.

        Args:
        - df (pl.DataFrame): Input DataFrame.

        Returns:
        - pl.DataFrame: DataFrame with transformed date columns.
        """
        for col in df.columns:
            if col.endswith("D"):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days().cast(pl.Int32))

        df = df.rename(
            {
                "MONTH": "month",
                "WEEK_NUM": "week_num"
            }
        )
                
        df = df.with_columns(
            [
                pl.col("date_decision").dt.year().alias("year").cast(pl.Int16),
                pl.col("date_decision").dt.day().alias("day").cast(pl.UInt8),
            ]
        )

        return df.drop("date_decision")

def filter_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filters columns in the DataFrame based on null percentage and unique values for string columns.

    Args:
    - df (pl.DataFrame): Input DataFrame.

    Returns:
    - pl.DataFrame: DataFrame with filtered columns.
    """
    for col in df.columns:
        if col not in ["case_id", "year", "month", "week_num", "target"]:
            null_pct = df[col].is_null().mean()
            # if isnull > 0.9: # kontsev
            if null_pct > 0.95:
            # if null_pct == 1: 
                df = df.drop(col)  

    for col in df.columns:
        if (col not in ["case_id", "year", "month", "week_num", "target"]) & (
            df[col].dtype == pl.String
        ):
            freq = df[col].n_unique()

            # lgbm会报错:lightgbm.basic.LightGBMError: bin size 258 cannot run on GPU
            # if (freq == 1) | (freq > 1000): # kontsev
            if (freq > 200) | (freq == 1): 
                df = df.drop(col)

    return df

def transform_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transforms columns in the DataFrame according to predefined rules.

    Args:
    - df (pl.DataFrame): Input DataFrame.

    Returns:
    - pl.DataFrame: DataFrame with transformed columns.
    """
    if "riskassesment_302T" in df.columns:
        if df["riskassesment_302T"].dtype == pl.Null:
            df = df.with_columns(
                [
                    pl.Series(
                        "riskassesment_302T_rng", df["riskassesment_302T"], pl.UInt8
                    ),
                    pl.Series(
                        "riskassesment_302T_mean", df["riskassesment_302T"], pl.UInt8
                    ),
                ]
            )
        else:
            pct_low: pl.Series = (
                df["riskassesment_302T"]
                .str.split(" - ")
                .apply(lambda x: x[0].replace("%", ""))
                .cast(pl.UInt8)
            )
            pct_high: pl.Series = (
                df["riskassesment_302T"]
                .str.split(" - ")
                .apply(lambda x: x[1].replace("%", ""))
                .cast(pl.UInt8)
            )

            diff: pl.Series = pct_high - pct_low
            avg: pl.Series = ((pct_low + pct_high) / 2).cast(pl.Float32)

            del pct_high, pct_low
            gc.collect()

            df = df.with_columns(
                [
                    diff.alias("riskassesment_302T_rng"),
                    avg.alias("riskassesment_302T_mean"),
                ]
            )

        df.drop("riskassesment_302T")

    return df


class Aggregator:
    #Please add or subtract features yourself, be aware that too many features will take up too much space.
    def num_expr(df):
        # P是逾期天数，A是借贷数目
        # 感觉1个均值即可
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr = [pl.max(col).alias(f"{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的
        
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的
        
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]

        # my code
        expr_std = [pl.std(col).alias(f"std_{col}") for col in cols]
        expr_sum = [pl.sum(col).alias(f"sum_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        expr_median = [pl.median(col).alias(f"median_{col}") for col in cols]
        # 0.754300 排列顺序
        # return expr_max + expr_min + expr_last + expr_first + expr_mean

        # return expr_max + expr_mean + expr_var # notebookv8 
        # return expr_max +expr_last+expr_mean # 829+386 
        # return expr_max +expr_last+expr_mean+expr_var # 829+386 + notebookv8

        # return expr_max # ZhiXing Jiang
        # return expr +expr_last+expr_mean+expr_var+expr_count + expr_median+expr_max # kontsev +expr_max
        return expr + expr_mean + expr_var # kontsev
    
    def date_expr(df):
        # D是借贷日期
        # 感觉1个均值就行
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr = [pl.max(col).alias(f"{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的

        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的
        
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        # 0.754300 排列顺序
        # return  expr_max + expr_min  +  expr_last + expr_first + expr_mean

        # return expr_max + expr_mean + expr_var # notebookv8
        # return  expr_max +expr_last+expr_mean # 829+386
        # return  expr_max +expr_last+expr_mean+expr_var # 829+386+notebookv8 
        # return expr_max # ZhiXing Jiang

        # return  expr +expr_last+expr_mean+expr_max # kontsev+expr_max
        return  expr + expr_mean + expr_var  # kontsev


    
    def str_expr(df):
        # M是地址编号
        # 1个人最多也就几个地址吧，感觉取2个就可以了
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr = [pl.max(col).alias(f"{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的 
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的
        expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]

        # my code
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_mode = [pl.col(col).drop_nulls().mode().first().alias(f'mode_{col}') for col in cols]

        # 0.754300 排列顺序
        # return  expr_max + expr_min + expr_last + expr_first + expr_count
        

        # return expr_max # notebookv8
        # return expr_max +expr_last # 829+386
        # return  expr_max +expr_last # 829+386+notebookv8
        # return expr_max # ZhiXing Jiang
        return  expr # kontsev

    def other_expr(df):
        # T、L代表各种杂七杂八的信息
        # 这一块可做特征工程提分，但粗略的来说1个均值更合适
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr = [pl.max(col).alias(f"{col}") for col in cols]
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



        # return expr_max # notebookv8
        # return  expr_max +expr_last # 829+386
        # return  expr_max +expr_last # 829+386+notebookv8
        # return expr_max # ZhiXing Jiang
        # return  expr +expr_last +expr_max # kontsev +expr_max
        return  expr # kontsev +expr_max

    
    def count_expr(df):
        
        # 其他一个case_id对应多条信息的，由于不知道具体是啥意思，所以统计特征用mean是比较好的感觉
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols] 
        expr = [pl.max(col).alias(f"{col}") for col in cols]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in cols] # 原本是忽略的
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols] # 原本是忽略的

        # my code 
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        # 0.755666 排列顺序
        # return  expr_max + expr_min + expr_last + expr_first + expr_count


        # return expr_max # notebookv8
        # return  expr_max +expr_last # 829+386
        # return  expr_max +expr_last # 829+386+notebookv8
        # return expr_max # ZhiXing Jiang
        return  expr +expr_last+expr_count +expr_max # kontsev +expr_max
        return  expr # kontsev +expr_max
    
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

    df_base = df_base.pipe(Pipeline.handle_dates)
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


class SchemaGen:
    @staticmethod
    def change_dtypes(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Changes the data types of columns in the DataFrame.

        Args:
        - df (pl.LazyFrame): Input LazyFrame.

        Returns:
        - pl.LazyFrame: LazyFrame with modified data types.
        """
        for col in df.columns:
            if col == "case_id":
                df = df.with_columns(pl.col(col).cast(pl.UInt32).alias(col))
            elif col in ["WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.UInt16).alias(col))
            elif col == "date_decision" or col[-1] == "D":
                df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
            elif col[-1] in ["P", "A"]:
                df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
        return df

    @staticmethod
    def scan_files(glob_path: str, depth: int = None) -> pl.LazyFrame:
        """
        Scans Parquet files matching the glob pattern and combines them into a LazyFrame.

        Args:
        - glob_path (str): Glob pattern to match Parquet files.
        - depth (int, optional): Depth level for data aggregation. Defaults to None.

        Returns:
        - pl.LazyFrame: Combined LazyFrame.
        """
        chunks: list[pl.LazyFrame] = []
        for path in glob(str(glob_path)):
            df: pl.LazyFrame = pl.scan_parquet(
                path, low_memory=True, rechunk=True
            ).pipe(SchemaGen.change_dtypes)


            print(f"File {Path(path).stem} loaded into memory.")

            if depth in (1, 2):
                exprs: list[pl.Series] = Aggregator.get_exprs(df)
                df = df.group_by("case_id").agg(exprs)

                del exprs
                gc.collect()

            chunks.append(df)

        df = pl.concat(chunks, how="vertical_relaxed")

        del chunks
        gc.collect()

        df = df.unique(subset=["case_id"])

        return df

    @staticmethod
    def join_dataframes(
        df_base: pl.LazyFrame,
        depth_0: list[pl.LazyFrame],
        depth_1: list[pl.LazyFrame],
        depth_2: list[pl.LazyFrame],
    ) -> pl.DataFrame:
        """
        Joins multiple LazyFrames with a base LazyFrame.

        Args:
        - df_base (pl.LazyFrame): Base LazyFrame.
        - depth_0 (list[pl.LazyFrame]): List of LazyFrames for depth 0.
        - depth_1 (list[pl.LazyFrame]): List of LazyFrames for depth 1.
        - depth_2 (list[pl.LazyFrame]): List of LazyFrames for depth 2.

        Returns:
        - pl.DataFrame: Joined DataFrame.
        """

        # ===============================================================
        # """ 为了兼容0.592的模型，我自己加上去的 """
        # df_base = (
        #     df_base
        #     .with_columns(
        #         month_decision = pl.col("date_decision").dt.month(),
        #         weekday_decision = pl.col("date_decision").dt.weekday(),
        #     )
        # )
        # ===============================================================

        for i, df in enumerate(depth_0 + depth_1 + depth_2):
            df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

        try: # 如果是lazyframe对象，则collect转化为dataframe
            return df_base.collect().pipe(Utility.reduce_memory_usage, "df_train")
        except: # 如果是dataframe则无需转化
            return df_base.pipe(Utility.reduce_memory_usage, "df_train")


class Utility:
    
    def reduce_memory_usage(df: pl.DataFrame, name) -> pl.DataFrame:
        """
        Reduces memory usage of a DataFrame by converting column types.

        Args:
        - df (pl.DataFrame): DataFrame to optimize.
        - name (str): Name of the DataFrame.

        Returns:
        - pl.DataFrame: Optimized DataFrame.
        """
        # print(
        #     f"Memory usage of dataframe \"{name}\" is {round(df.estimated_size('mb'), 4)} MB."
        # )

        int_types = [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]
        float_types = [pl.Float32, pl.Float64]

        for col in df.columns:
            col_type = df[col].dtype
            if col_type in int_types + float_types:
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min is not None and c_max is not None:
                    if col_type in int_types:
                        if c_min >= 0:
                            if (
                                c_min >= np.iinfo(np.uint8).min
                                and c_max <= np.iinfo(np.uint8).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt8))
                            elif (
                                c_min >= np.iinfo(np.uint16).min
                                and c_max <= np.iinfo(np.uint16).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt16))
                            elif (
                                c_min >= np.iinfo(np.uint32).min
                                and c_max <= np.iinfo(np.uint32).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt32))
                            elif (
                                c_min >= np.iinfo(np.uint64).min
                                and c_max <= np.iinfo(np.uint64).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt64))
                        else:
                            if (
                                c_min >= np.iinfo(np.int8).min
                                and c_max <= np.iinfo(np.int8).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int8))
                            elif (
                                c_min >= np.iinfo(np.int16).min
                                and c_max <= np.iinfo(np.int16).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int16))
                            elif (
                                c_min >= np.iinfo(np.int32).min
                                and c_max <= np.iinfo(np.int32).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int32))
                            elif (
                                c_min >= np.iinfo(np.int64).min
                                and c_max <= np.iinfo(np.int64).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int64))
                    elif col_type in float_types:
                        if (
                            c_min > np.finfo(np.float32).min
                            and c_max < np.finfo(np.float32).max
                        ):
                            df = df.with_columns(df[col].cast(pl.Float32))

        # print(
        #     f"Memory usage of dataframe \"{name}\" became {round(df.estimated_size('mb'), 4)} MB."
        # )

        return df



    def to_pandas(df: pl.DataFrame, cat_cols: list[str] = None) -> (pd.DataFrame, list[str]):  # type: ignore
        """
        Converts a Polars DataFrame to a Pandas DataFrame.

        Args:
        - df (pl.DataFrame): Polars DataFrame to convert.
        - cat_cols (list[str]): List of categorical columns. Default is None.

        Returns:
        - (pd.DataFrame, list[str]): Tuple containing the converted Pandas DataFrame and categorical columns.
        """
        df: pd.DataFrame = df.to_pandas()

        if cat_cols is None:
            cat_cols = list(df.select_dtypes("object").columns)

        df[cat_cols] = df[cat_cols].astype("str")

        return df, cat_cols





ROOT            = Path("/home/xyli/kaggle")

TRAIN_DIR       = ROOT / "parquet_files" / "train"
TEST_DIR        = ROOT / "parquet_files" / "test"


print('开始读取数据!')

# train_credit_bureau_a_1 = SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_a_1_*.parquet', 1)
# train_credit_bureau_a_1 = train_credit_bureau_a_1.with_columns(
#     ((pl.col('max_dateofcredend_289D') - pl.col('max_dateofcredstart_739D')).dt.total_days()).alias('max_credit_duration_daysA')
# ).with_columns(
#     ((pl.col('max_dateofcredend_353D') - pl.col('max_dateofcredstart_181D')).dt.total_days()).alias('max_closed_credit_duration_daysA')
# ).with_columns(
#     ((pl.col('max_dateofrealrepmt_138D') - pl.col('max_overdueamountmax2date_1002D')).dt.total_days()).alias('max_time_from_overdue_to_closed_realrepmtA')
# ).with_columns(
#     ((pl.col('max_dateofrealrepmt_138D') - pl.col('max_overdueamountmax2date_1142D')).dt.total_days()).alias('max_time_from_active_overdue_to_realrepmtA')
# )

# train_credit_bureau_b_1 = SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_b_1.parquet', 1)
# train_credit_bureau_b_1 = train_credit_bureau_b_1.with_columns(
#     ((pl.col('max_contractmaturitydate_151D') - pl.col('max_contractdate_551D')).dt.total_days()).alias('contract_duration_days')
# ).with_columns(
#     ((pl.col('max_lastupdate_260D') - pl.col('max_contractdate_551D')).dt.total_days()).alias('last_update_duration_days')
# )

# train_static = SchemaGen.scan_files(TRAIN_DIR / 'train_static_0_*.parquet', 1)
# condition_all_nan = (
#     pl.col('max_maxdbddpdlast1m_3658939P').is_null() &
#     pl.col('max_maxdbddpdtollast12m_3658940P').is_null() &
#     pl.col('max_maxdbddpdtollast6m_4187119P').is_null()
# )

# condition_exceed_thresholds = (
#     (pl.col('max_maxdbddpdlast1m_3658939P') > 31) |
#     (pl.col('max_maxdbddpdtollast12m_3658940P') > 366) |
#     (pl.col('max_maxdbddpdtollast6m_4187119P') > 184)
# )

# train_static = train_static.with_columns(
#     pl.when(condition_all_nan | condition_exceed_thresholds)
#     .then(0)
#     .otherwise(1)
#     .alias('max_dbddpd_boolean')
# )
# train_static = train_static.with_columns(
#     pl.when(
#         (pl.col('max_maxdbddpdlast1m_3658939P') <= 0) &
#         (pl.col('max_maxdbddpdtollast12m_3658940P') <= 0) &
#         (pl.col('max_maxdbddpdtollast6m_4187119P') <= 0)
#     )
#     .then(1)
#     .otherwise(0)
#     .alias('max_pays_debt_on_timeP')
# )

# data_store = {
#     "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
#     "depth_0": [
#         read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
#         read_files(TRAIN_DIR / "train_static_0_*.parquet"),
#     ],
#     "depth_1": [
#         read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
#         read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
#         read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
#         read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_other_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_person_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
#         read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
#     ],
#     "depth_2": [
#         read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
#         read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),

#         # 829+386
#         read_file(TRAIN_DIR / "train_applprev_2.parquet", 2),
#         read_file(TRAIN_DIR / "train_person_2.parquet", 2)
#     ]
# }

data_store:dict = {
    # 缺点是无法识别类型
    'df_base': SchemaGen.scan_files(TRAIN_DIR / 'train_base.parquet'),
    'depth_0': [
        SchemaGen.scan_files(TRAIN_DIR / 'train_static_cb_0.parquet'),

        # ZhiXing Jiang
        SchemaGen.scan_files(TRAIN_DIR / 'train_static_0_*.parquet'),
        # train_static,
        
    ],
    'depth_1': [
        SchemaGen.scan_files(TRAIN_DIR / 'train_applprev_1_*.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_tax_registry_a_1.parquet', 1),

        # ZhiXing Jiang
        SchemaGen.scan_files(TRAIN_DIR / 'train_tax_registry_b_1.parquet', 1),
        # train_credit_bureau_b_1,

        SchemaGen.scan_files(TRAIN_DIR / 'train_tax_registry_c_1.parquet', 1),

        # ZhiXing Jiang
        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_a_1_*.parquet', 1),
        # train_credit_bureau_a_1,

        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_b_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_other_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_person_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_deposit_1.parquet', 1),
        SchemaGen.scan_files(TRAIN_DIR / 'train_debitcard_1.parquet', 1),
    ],
    'depth_2': [
        # ZhiXing Jiang
        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_a_2_*.parquet', 2),

        SchemaGen.scan_files(TRAIN_DIR / 'train_credit_bureau_b_2.parquet', 2),
   
        # ZhiXing Jiang
        # # 829+386
        SchemaGen.scan_files(TRAIN_DIR / 'train_applprev_2.parquet', 2), 
        SchemaGen.scan_files(TRAIN_DIR / 'train_person_2.parquet', 2), 
    ]
}

print('读取数据完毕！')

# df_train_scan: pl.LazyFrame = (
#     SchemaGen.join_dataframes(**data_store) # 别忘记829+386要多加载2个文件+改新增的统计特征
#     .pipe(filter_cols)
#     .pipe(transform_cols) # 兼容0.592
#     .pipe(handle_dates)
#     .pipe(Utility.reduce_memory_usage, "df_train") 
# )
# df_train_scan, cat_cols = Utility.to_pandas(df_train_scan) # 这个是把字符串转化为str
# print("df_train_scan shape:\t", df_train_scan.shape)
# df_train = df_train_scan

df_train = feature_eng(**data_store).collect() # 别忘记829+386要多加载2个文件


# ===============================================================
""" 为了适应以下特征工程代码，我将pl.df转换为pd.df，然后进行特征工程,这一块要等10min左右 """

df_train_columns = list(df_train.columns)
df_train = pd.DataFrame(df_train)
df_train.columns = df_train_columns 
df_train = df_train.set_index('case_id')

# from IPython import embed
# embed()

# 将DataFrame中的所有NaN替换为numpy中的NaN值
# 这将会将DataFrame中所有的NA值（包括NaN和None）替换为numpy中的NaN值
df_train.replace({pd.NA: np.nan}, inplace=True)

df_train['past_now_annuity'] = np.where(df_train['annuity_780A'] == 0, 0, df_train['annuity_853A'] / df_train['annuity_780A'])
# df_test['past_now_annuity'] = np.where(df_test['annuity_780A'] == 0, 0, df_test['annuity_853A'] / df_test['annuity_780A'])

df_train['days_previous_application'] = df_train['approvaldate_319D'] * -1/365
# df_test['days_previous_application'] = df_test['approvaldate_319D'] * -1/365

df_train['previous_income_to_amtin'] = np.where(df_train['amtinstpaidbefduel24m_4187115A'] == 0, np.nan, df_train['byoccupationinc_3656910L'] / df_train['amtinstpaidbefduel24m_4187115A'])
# df_test['previous_income_to_amtin'] = np.where(df_test['amtinstpaidbefduel24m_4187115A'] == 0, np.nan, df_test['byoccupationinc_3656910L'] / df_test['amtinstpaidbefduel24m_4187115A'])

df_train['prev_income_child_rate'] = np.where(df_train['childnum_21L'] == 0, df_train['byoccupationinc_3656910L'], df_train['byoccupationinc_3656910L'] / df_train['childnum_21L'])
# df_test['prev_income_child_rate'] = np.where(df_test['childnum_21L'] == 0, df_test['byoccupationinc_3656910L'], df_test['byoccupationinc_3656910L'] / df_test['childnum_21L'])

df_train['days_creation'] = df_train['creationdate_885D'] * -1/365
# df_test['days_creation'] = df_test['creationdate_885D'] * -1/365

df_train['days_creation_minus_tax_deduction_date'] = df_train['recorddate_4527225D'] - df_train['days_creation']
# df_test['days_creation_minus_tax_deduction_date'] = df_test['recorddate_4527225D'] - df_test['days_creation']

df_train['BYOCCUPINC_DIV_TAX_DEDUC_AMT'] = np.where(df_train['amount_4527230A'] == 0, np.nan, df_train['byoccupationinc_3656910L'] / df_train['amount_4527230A'])
# df_test['BYOCCUPINC_DIV_TAX_DEDUC_AMT'] = np.where(df_test['amount_4527230A'] == 0, np.nan, df_test['byoccupationinc_3656910L'] / df_test['amount_4527230A'])

df_train['MONTHLY_ANNUITY_MINUS_TAX_DEDUC_AMT'] = df_train['annuity_780A'] - df_train['amount_4527230A']
# df_test['MONTHLY_ANNUITY_MINUS_TAX_DEDUC_AMT'] = df_test['annuity_780A'] - df_test['amount_4527230A']

df_train['AMT_BUREAU_PAYMENTS_MINUS_TAX_DEDUC_AMT'] = df_train['amount_4527230A'] - df_train['pmtamount_36A']
# df_test['AMT_BUREAU_PAYMENTS_MINUS_TAX_DEDUC_AMT'] = df_test['amount_4527230A'] - df_test['pmtamount_36A']

df_train['DAYS_DEDUC_DATE'] = df_train['processingdate_168D'] * -1/365
# df_test['DAYS_DEDUC_DATE'] = df_test['processingdate_168D'] * -1/365

df_train['PMTAMOUNT_TO_BYOCCUPINC'] = np.where(df_train['pmtamount_36A'] == 0, np.nan, df_train['byoccupationinc_3656910L'] / df_train['pmtamount_36A'])
# df_test['PMTAMOUNT_TO_BYOCCUPINC'] = np.where(df_test['pmtamount_36A'] == 0, np.nan, df_test['byoccupationinc_3656910L'] / df_test['pmtamount_36A'])


df_train['PERSON_BIRTHDAY'] = df_train['birth_259D'] * -1/365
# df_test['PERSON_BIRTHDAY'] = df_test['birth_259D'] * -1/365

df_train['BIRTHDAY_VS_AMT_CREDIT'] = np.where(df_train['pmtamount_36A'] == 0, 0, df_train['birth_259D'] / df_train['pmtamount_36A'])
# df_test['BIRTHDAY_VS_AMT_CREDIT'] = np.where(df_test['pmtamount_36A'] == 0, 0, df_test['birth_259D'] / df_test['pmtamount_36A'])

df_train['DAYS_CREDIT_VS_DAYS_BIRTHDAY'] = np.where(df_train['days120_123L'] == 0, 0, df_train['birth_259D'] / df_train['days120_123L'])
# df_test['DAYS_CREDIT_VS_DAYS_BIRTHDAY'] = np.where(df_test['days120_123L'] == 0, 0, df_test['birth_259D'] / df_test['days120_123L'])

df_train['START_EMPLOYMENT'] = df_train['empl_employedfrom_271D'] * -1/365
# df_test['START_EMPLOYMENT'] = df_test['empl_employedfrom_271D'] * -1/365

df_train['NEW_DAYS_EMPLOYED_PERC'] = df_train['START_EMPLOYMENT'] / df_train['PERSON_BIRTHDAY']
# df_test['NEW_DAYS_EMPLOYED_PERC'] = df_test['START_EMPLOYMENT'] / df_test['PERSON_BIRTHDAY']

df_train['DEBIT_COMIN_VS_BYOCCUPINC'] = np.where(df_train['amtdebitincoming_4809443A'] == 0, 0, df_train['byoccupationinc_3656910L'] / df_train['amtdebitincoming_4809443A'])
# df_test['DEBIT_COMIN_VS_BYOCCUPINC'] = np.where(df_test['amtdebitincoming_4809443A'] == 0, 0, df_test['byoccupationinc_3656910L'] / df_test['amtdebitincoming_4809443A'])

df_train['DEBIT_COMIN_VS_PMT_AMOUNT'] = np.where(df_train['pmtamount_36A'] == 0, 0, df_train['amtdebitincoming_4809443A'] / df_train['pmtamount_36A'])
# df_test['DEBIT_COMIN_VS_PMT_AMOUNT'] = np.where(df_test['pmtamount_36A'] == 0, 0, df_test['amtdebitincoming_4809443A'] / df_test['pmtamount_36A'])

df_train['DEBIT_VS_ANNUITY'] = np.where(df_train['annuity_780A'] == 0, 0, df_train['amtdepositbalance_4809441A'] / df_train['annuity_780A'])
# df_test['DEBIT_VS_ANNUITY'] = np.where(df_test['annuity_780A'] == 0, 0, df_test['amtdepositbalance_4809441A'] / df_test['annuity_780A'])

df_train['OUTDEBIT_VS_ANNUITY'] = np.where(df_train['annuity_780A'] == 0, 0, df_train['amtdebitoutgoing_4809440A'] / df_train['annuity_780A'])
# df_test['OUTDEBIT_VS_ANNUITY'] = np.where(df_test['annuity_780A'] == 0, 0, df_test['amtdebitoutgoing_4809440A'] / df_test['annuity_780A'])

df_train['PMTAMOUNT_VS_AMTDEPOSIT_INC'] = np.where(df_train['pmtamount_36A'] == 0, 0, df_train['amtdepositincoming_4809444A'] / df_train['pmtamount_36A'])
# df_test['PMTAMOUNT_VS_AMTDEPOSIT_INC'] = np.where(df_test['pmtamount_36A'] == 0, 0, df_test['amtdepositincoming_4809444A'] / df_test['pmtamount_36A'])

df_train['DEPOSTI_IN_VS_OUT'] = df_train['amtdepositincoming_4809444A'] - df_train['amtdepositoutgoing_4809442A']
# df_test['DEPOSTI_IN_VS_OUT'] = df_test['amtdepositincoming_4809444A'] - df_test['amtdepositoutgoing_4809442A']

df_train['ANNUITY_MINUS_INDEPOSIT'] = df_train['annuity_780A'] - df_train['amtdepositincoming_4809444A']
# df_test['ANNUITY_MINUS_INDEPOSIT'] = df_test['annuity_780A'] - df_test['amtdepositincoming_4809444A']

df_train['INDEPOSIT_VS_OUTDEPOSIT'] = np.where(df_train['amtdepositoutgoing_4809442A'] == 0, 0, df_train['amtdepositincoming_4809444A'] / df_train['amtdepositoutgoing_4809442A'])
# df_test['INDEPOSIT_VS_OUTDEPOSIT'] = np.where(df_test['amtdepositoutgoing_4809442A'] == 0, 0, df_test['amtdepositincoming_4809444A'] / df_test['amtdepositoutgoing_4809442A'])

df_train['INDEPOSIT_VS_CHILDNUM'] = np.where(df_train['childnum_21L'] == 0, df_train['amtdepositincoming_4809444A'], df_train['amtdepositincoming_4809444A'] / df_train['childnum_21L'])
# df_test['INDEPOSIT_VS_CHILDNUM'] = np.where(df_test['childnum_21L'] == 0, df_test['amtdepositincoming_4809444A'], df_test['amtdepositincoming_4809444A'] / df_test['childnum_21L'])

df_train['NEW_OVER_EXPECT_CREDIT'] = (df_train['amount_4527230A'] > df_train['amtdepositbalance_4809441A']).replace({False: 0, True: 1})
# df_test['NEW_OVER_EXPECT_CREDIT'] = (df_test['amount_4527230A'] > df_test['amtdepositbalance_4809441A']).replace({False: 0, True: 1})

df_train['NEW_OVER_EXPECT_CREDIT_TAXAMOUNT'] = (df_train['pmtamount_36A'] > df_train['amtdepositbalance_4809441A']).replace({False: 0, True: 1})
# df_test['NEW_OVER_EXPECT_CREDIT_TAXAMOUNT'] = (df_test['pmtamount_36A'] > df_test['amtdepositbalance_4809441A']).replace({False: 0, True: 1})

df_train['CONTRACTSUM_VS_EFFECTIVERATE'] = np.where(df_train['annualeffectiverate_199L'] == 0, np.nan, df_train['contractssum_5085716L'] / df_train['annualeffectiverate_199L'])
# df_test['CONTRACTSUM_VS_EFFECTIVERATE'] = np.where(df_test['annualeffectiverate_199L'] == 0, np.nan, df_test['contractssum_5085716L'] / df_test['annualeffectiverate_199L'])

df_train['EFFRATE_VS_AMTDEPOSIT_INC'] = np.where(df_train['annualeffectiverate_199L'] == 0, 0, df_train['amtdepositincoming_4809444A'] / df_train['annualeffectiverate_199L'])
# df_test['EFFRATE_VS_AMTDEPOSIT_INC'] = np.where(df_test['annualeffectiverate_199L'] == 0, 0, df_test['amtdepositincoming_4809444A'] / df_test['annualeffectiverate_199L'])

df_train['CONTRACTSUM_VS_EFFECTIVERATE_ACTIVE'] = np.where(df_train['annualeffectiverate_63L'] == 0, np.nan, df_train['contractssum_5085716L'] / df_train['annualeffectiverate_63L'])
# df_test['CONTRACTSUM_VS_EFFECTIVERATE_ACTIVE'] = np.where(df_test['annualeffectiverate_63L'] == 0, np.nan, df_test['contractssum_5085716L'] / df_test['annualeffectiverate_63L'])

df_train['PREVRATE_TO_NEW_RATE'] = np.where(df_train['annualeffectiverate_199L'] == 0, 0, df_train['annualeffectiverate_63L'] / df_train['annualeffectiverate_199L'])
# df_test['PREVRATE_TO_NEW_RATE'] = np.where(df_test['annualeffectiverate_199L'] == 0, 0, df_test['annualeffectiverate_63L'] / df_test['annualeffectiverate_199L'])

df_train['TAXAM_CREDLIM'] = (df_train['pmtamount_36A'] > df_train['credlmt_230A']).replace({False: 0, True: 1})
# df_test['TAXAM_CREDLIM'] = (df_test['pmtamount_36A'] > df_test['credlmt_230A']).replace({False: 0, True: 1})

df_train['PREV_CREDLIM_CURRENT_CREDLIM'] = (df_train['credlmt_935A'] > df_train['credlmt_230A']).replace({False: 0, True: 1})
# df_test['PREV_CREDLIM_CURRENT_CREDLIM'] = (df_test['credlmt_935A'] > df_test['credlmt_230A']).replace({False: 0, True: 1})

df_train['DEBIT_COMIN_VS_CREDLIM'] = np.where(df_train['credlmt_935A'] == 0, 0, df_train['amtdebitincoming_4809443A'] / df_train['credlmt_935A'])
# df_test['DEBIT_COMIN_VS_CREDLIM'] = np.where(df_test['credlmt_935A'] == 0, 0, df_test['amtdebitincoming_4809443A'] / df_test['credlmt_935A'])

df_train['EMPL_VS_CREDENTDATE'] = np.where(df_train['dateofcredend_289D'] == 0, np.nan, df_train['START_EMPLOYMENT'] / df_train['dateofcredend_289D'])
# df_test['EMPL_VS_CREDENTDATE'] = np.where(df_test['dateofcredend_289D'] == 0, np.nan, df_test['START_EMPLOYMENT'] / df_test['dateofcredend_289D'])

df_train['CREDENTDATE_MINUS_BIRTHDAY'] = df_train['dateofcredend_289D'] - df_train['PERSON_BIRTHDAY']
# df_test['CREDENTDATE_MINUS_BIRTHDAY'] = df_test['dateofcredend_289D'] - df_test['PERSON_BIRTHDAY']

df_train['CREDENTDATE_VS_BIRTHDAY'] = np.where(df_train['PERSON_BIRTHDAY'] == 0, np.nan, df_train['dateofcredend_289D'] / df_train['PERSON_BIRTHDAY'])
# df_test['CREDENTDATE_VS_BIRTHDAY'] = np.where(df_test['PERSON_BIRTHDAY'] == 0, np.nan, df_test['dateofcredend_289D'] / df_test['PERSON_BIRTHDAY'])

df_train['PREVCREDENTDATE'] = df_train['dateofcredend_353D'] * -1/365
# df_test['PREVCREDENTDATE'] = df_test['dateofcredend_353D'] * -1/365

df_train['PREVCREDENTDATE_CLOSEDCONTR'] = df_train['dateofcredstart_181D'] * -1/365
# df_test['PREVCREDENTDATE_CLOSEDCONTR'] = df_test['dateofcredstart_181D'] * -1/365

df_train['PREVCREDENTDATE_OPENCONTR'] = df_train['dateofcredstart_739D'] * -1/365
# df_test['PREVCREDENTDATE_OPENCONTR'] = df_test['dateofcredstart_739D'] * -1/365

df_train['PREVCREDENTDATE_CLOSEDCONTR_REAL'] = df_train['dateofrealrepmt_138D'] * -1/365
# df_test['PREVCREDENTDATE_CLOSEDCONTR_REAL'] = df_test['dateofrealrepmt_138D'] * -1/365

df_train['CREDIT_OVERDUE'] = (df_train['PREVCREDENTDATE_CLOSEDCONTR_REAL'] > df_train['PREVCREDENTDATE_CLOSEDCONTR']).replace({False: 0, True: 1})
# df_test['CREDIT_OVERDUE'] = (df_test['PREVCREDENTDATE_CLOSEDCONTR_REAL'] > df_test['PREVCREDENTDATE_CLOSEDCONTR']).replace({False: 0, True: 1})

df_train['OUTSTANDING_DEBIT'] = (df_train['debtoutstand_525A'] > df_train['amtdepositbalance_4809441A']).replace({False: 0, True: 1})
# df_test['OUTSTANDING_DEBIT'] = (df_test['debtoutstand_525A'] > df_test['amtdepositbalance_4809441A']).replace({False: 0, True: 1})

df_train['OUTSTANDING_DEBIT_VS_BYOCCUPINC'] = np.where(df_train['byoccupationinc_3656910L'] == 0, np.nan, df_train['debtoutstand_525A'] / df_train['byoccupationinc_3656910L'])
# df_test['OUTSTANDING_DEBIT_VS_BYOCCUPINC'] = np.where(df_test['byoccupationinc_3656910L'] == 0, np.nan, df_test['debtoutstand_525A'] / df_test['byoccupationinc_3656910L'])

df_train['OUTSTANDING_DEBIT_VS_BYOCCUPINC_CLASSIF'] = (df_train['debtoutstand_525A'] > df_train['byoccupationinc_3656910L']).replace({False: 0, True: 1})
# df_test['OUTSTANDING_DEBIT_VS_BYOCCUPINC_CLASSIF'] = (df_test['debtoutstand_525A'] > df_test['byoccupationinc_3656910L']).replace({False: 0, True: 1})

df_train['DEBIT_TO_NUM_OF_CHILD'] = np.where(df_train['childnum_21L'] == 0, df_train['debtoutstand_525A'], df_train['debtoutstand_525A'] / df_train['childnum_21L'])
# df_test['DEBIT_TO_NUM_OF_CHILD'] = np.where(df_test['childnum_21L'] == 0, df_test['debtoutstand_525A'], df_test['debtoutstand_525A'] / df_test['childnum_21L'])

df_train['DEBIT_OVERDUE'] = (df_train['debtoverdue_47A'] > 0).replace({False: 0, True: 1})
# df_test['DEBIT_OVERDUE'] = (df_test['debtoverdue_47A'] > 0).replace({False: 0, True: 1})

df_train['DEBIT_OVERDUE_VS_BYOCCUPINC'] = np.where(df_train['byoccupationinc_3656910L'] == 0, np.nan, df_train['debtoverdue_47A'] / df_train['byoccupationinc_3656910L'])
# df_test['DEBIT_OVERDUE_VS_BYOCCUPINC'] = np.where(df_test['byoccupationinc_3656910L'] == 0, np.nan, df_test['debtoverdue_47A'] / df_test['byoccupationinc_3656910L'])

df_train['OUTSTANDING_DEBIT_VS_DEBIT_OVERDUE'] = np.where(df_train['debtoverdue_47A'] == 0, np.nan, df_train['debtoutstand_525A'] / df_train['debtoverdue_47A'])
# df_test['OUTSTANDING_DEBIT_VS_DEBIT_OVERDUE'] = np.where(df_test['debtoverdue_47A'] == 0, np.nan, df_test['debtoutstand_525A'] / df_test['debtoverdue_47A'])

df_train['INSTLAM_VS_CONNTRACTSUM'] = np.where(df_train['contractssum_5085716L'] == 0, np.nan, df_train['instlamount_768A'] / df_train['contractssum_5085716L'])
# df_test['INSTLAM_VS_CONNTRACTSUM'] = np.where(df_test['contractssum_5085716L'] == 0, np.nan, df_test['instlamount_768A'] / df_test['contractssum_5085716L'])

df_train['INSTLAM_CLOSED_VS_CONNTRACTSUM'] = np.where(df_train['contractssum_5085716L'] == 0, np.nan, df_train['instlamount_852A'] / df_train['contractssum_5085716L'])
# df_test['INSTLAM_CLOSED_VS_CONNTRACTSUM'] = np.where(df_test['contractssum_5085716L'] == 0, np.nan, df_test['instlamount_852A'] / df_test['contractssum_5085716L'])

df_train['ANNUITY_VS_INSTLAMO'] = np.where(df_train['instlamount_852A'] == 0, np.nan, df_train['annuity_780A'] / df_train['instlamount_852A'])
# df_test['ANNUITY_VS_INSTLAMO'] = np.where(df_test['instlamount_852A'] == 0, np.nan, df_test['annuity_780A'] / df_test['instlamount_852A'])

df_train['ANNUITY_VS_DEBT_OVERDUE'] = np.where(df_train['debtoverdue_47A'] == 0, np.nan, df_train['annuity_780A'] / df_train['debtoverdue_47A'])
# df_test['ANNUITY_VS_DEBT_OVERDUE'] = np.where(df_test['debtoverdue_47A'] == 0, np.nan, df_test['annuity_780A'] / df_test['debtoverdue_47A'])

df_train['ANNUITY_VS_DEBT_OVERSTAND'] = np.where(df_train['debtoutstand_525A'] == 0, np.nan, df_train['annuity_780A'] / df_train['debtoutstand_525A'])
# df_test['ANNUITY_VS_DEBT_OVERSTAND'] = np.where(df_test['debtoutstand_525A'] == 0, np.nan, df_test['annuity_780A'] / df_test['debtoutstand_525A'])

df_train['EMP_BIGGER_CREDITDATE'] = (df_train['START_EMPLOYMENT'] < df_train['PREVCREDENTDATE']).replace({False: 0, True: 1})
# df_test['EMP_BIGGER_CREDITDATE'] = (df_test['START_EMPLOYMENT'] < df_test['PREVCREDENTDATE']).replace({False: 0, True: 1})

df_train['OUTSTANDING_AMT_VS_CONTRSUM'] = np.where(df_train['contractssum_5085716L'] == 0, np.nan, df_train['outstandingamount_362A'] / df_train['contractssum_5085716L'])
# df_test['OUTSTANDING_AMT_VS_CONTRSUM'] = np.where(df_test['contractssum_5085716L'] == 0, np.nan, df_test['outstandingamount_362A'] / df_test['contractssum_5085716L'])

df_train['DEBIT_OUT_VS_OVERDUE_AMT'] = np.where(df_train['overdueamount_659A'] == 0, np.nan, df_train['debtoutstand_525A'] / df_train['overdueamount_659A'])
# df_test['DEBIT_OUT_VS_OVERDUE_AMT'] = np.where(df_test['overdueamount_659A'] == 0, np.nan, df_test['debtoutstand_525A'] / df_test['overdueamount_659A'])

df_train['OVERDUE_AMT_DATE'] = df_train['overdueamountmax2date_1002D'] * -1/365
# df_test['OVERDUE_AMT_DATE'] = df_test['overdueamountmax2date_1002D'] * -1/365

df_train['DEBIT_OUT_VS_OVERDUE_CNT'] = np.where(df_train['overdueamountmax_155A'] == 0, np.nan, df_train['debtoutstand_525A'] / df_train['overdueamountmax_155A'])
# df_test['DEBIT_OUT_VS_OVERDUE_CNT'] = np.where(df_test['overdueamountmax_155A'] == 0, np.nan, df_test['debtoutstand_525A'] / df_test['overdueamountmax_155A'])

df_train['PERIODPMT_VS_DBTOUTSTAND'] = np.where(df_train['periodicityofpmts_1102L'] == 0, 0, df_train['debtoutstand_525A'] / df_train['periodicityofpmts_1102L'])
# df_test['PERIODPMT_VS_DBTOUTSTAND'] = np.where(df_test['periodicityofpmts_1102L'] == 0, 0, df_test['debtoutstand_525A'] / df_test['periodicityofpmts_1102L'])

df_train['PERIODPMT_VS_PMTAM'] = np.where(df_train['periodicityofpmts_1102L'] == 0, 0, df_train['pmtamount_36A'] / df_train['periodicityofpmts_1102L'])
# df_test['PERIODPMT_VS_PMTAM'] = np.where(df_test['periodicityofpmts_1102L'] == 0, 0, df_test['pmtamount_36A'] / df_test['periodicityofpmts_1102L'])

df_train['TOT_DEBT_TO_CHILD_NUM'] = np.where(df_train['childnum_21L'] == 0, df_train['totaldebt_9A'], df_train['totaldebt_9A'] / df_train['childnum_21L'])
# df_test['TOT_DEBT_TO_CHILD_NUM'] = np.where(df_test['childnum_21L'] == 0, df_test['totaldebt_9A'], df_test['totaldebt_9A'] / df_test['childnum_21L'])

df_train['TOT_DEBT_TO_DEPOSIT_INC'] = np.where(df_train['amtdepositincoming_4809444A'] == 0, np.nan, df_train['totaldebt_9A'] / df_train['amtdepositincoming_4809444A'])
# df_test['TOT_DEBT_TO_DEPOSIT_INC'] = np.where(df_test['amtdepositincoming_4809444A'] == 0, np.nan, df_test['totaldebt_9A'] / df_test['amtdepositincoming_4809444A'])

df_train['TOT_DEBT_TO_DEBT_OUT'] = df_train['totaldebt_9A'] - df_train['debtoutstand_525A']
# df_test['TOT_DEBT_TO_DEBT_OUT'] = df_test['totaldebt_9A'] - df_test['debtoutstand_525A']

df_train['TOT_DEBT_TO_ANNUITY'] = df_train['totaldebt_9A'] - df_train['annuity_780A']
# df_test['TOT_DEBT_TO_ANNUITY'] = df_test['totaldebt_9A'] - df_test['annuity_780A']

df_train['TOT_DEBT_TO_DEBTOVERDUE'] = (df_train['debtoverdue_47A'] > df_train['totaldebt_9A']).replace({False: 0, True: 1})
# df_test['TOT_DEBT_TO_DEBTOVERDUE'] = (df_test['debtoverdue_47A'] > df_test['totaldebt_9A']).replace({False: 0, True: 1})

df_train['TOT_DEBT_TO_DEPOSIT_INC'] = np.where(df_train['pmtamount_36A'] == 0, np.nan, df_train['totaldebt_9A'] / df_train['pmtamount_36A'])
# df_test['TOT_DEBT_TO_DEPOSIT_INC'] = np.where(df_test['pmtamount_36A'] == 0, np.nan, df_test['totaldebt_9A'] / df_test['pmtamount_36A'])

df_train['TOT_DEBT_TO_BYOCCUPINC'] = np.where(df_train['totaldebt_9A'] == 0, np.nan, df_train['byoccupationinc_3656910L'] / df_train['totaldebt_9A'])
# df_test['TOT_DEBT_TO_BYOCCUPINC'] = np.where(df_test['totaldebt_9A'] == 0, np.nan, df_test['byoccupationinc_3656910L'] / df_test['totaldebt_9A'])

df_train['ACTIVE_CONT_TOTAL_DEBT'] = np.where(df_train['totaldebt_9A'] == 0, np.nan, df_train['amount_1115A'] / df_train['totaldebt_9A'])
# df_test['ACTIVE_CONT_TOTAL_DEBT'] = np.where(df_test['totaldebt_9A'] == 0, np.nan, df_test['amount_1115A'] / df_test['totaldebt_9A'])

df_train['DEBIT_OVERDUE_VS_ACTIVE_CONT'] = np.where(df_train['amount_1115A'] == 0, np.nan, df_train['debtoverdue_47A'] / df_train['amount_1115A'])
# df_test['DEBIT_OVERDUE_VS_ACTIVE_CONT'] = np.where(df_test['amount_1115A'] == 0, np.nan, df_test['debtoverdue_47A'] / df_test['amount_1115A'])

df_train['OUTSTANDING_DEBIT_VS_ACTIVE_CONT'] = np.where(df_train['amount_1115A'] == 0, np.nan, df_train['debtoutstand_525A'] / df_train['amount_1115A'])
# df_test['OUTSTANDING_DEBIT_VS_ACTIVE_CONT'] = np.where(df_test['amount_1115A'] == 0, np.nan, df_test['debtoutstand_525A'] / df_test['amount_1115A'])

df_train['ACTIVE_CONT_VS_AMTDEPOSIT_INC'] = np.where(df_train['amount_1115A'] == 0, 0, df_train['amtdepositincoming_4809444A'] / df_train['amount_1115A'])
# df_test['ACTIVE_CONT_VS_AMTDEPOSIT_INC'] = np.where(df_test['amount_1115A'] == 0, 0, df_test['amtdepositincoming_4809444A'] / df_test['amount_1115A'])

df_train['ACTIVE_CONT_DATE'] = df_train['contractdate_551D'] * -1/365
# df_test['ACTIVE_CONT_DATE'] = df_test['contractdate_551D'] * -1/365

df_train['ACTIVE_CONT_MATURITY'] = df_train['contractmaturitydate_151D'] * -1/365
# df_test['ACTIVE_CONT_MATURITY'] = df_test['contractmaturitydate_151D'] * -1/365

df_train['EMP_VS_ACRIVE_CONT_DATE'] = (df_train['START_EMPLOYMENT'] < df_train['ACTIVE_CONT_MATURITY']).replace({False: 0, True: 1})
# df_test['EMP_VS_ACRIVE_CONT_DATE'] = (df_test['START_EMPLOYMENT'] < df_test['ACTIVE_CONT_MATURITY']).replace({False: 0, True: 1})

df_train['TAXAM_CREDLIM_ACTIVE'] = (df_train['pmtamount_36A'] > df_train['credlmt_1052A']).replace({False: 0, True: 1})
# df_test['TAXAM_CREDLIM_ACTIVE'] = (df_test['pmtamount_36A'] > df_test['credlmt_1052A']).replace({False: 0, True: 1})

df_train['PREV_CREDLIM_CURRENT_CREDLIM_ACTIVE'] = (df_train['credlmt_1052A'] > df_train['credlmt_1052A']).replace({False: 0, True: 1})
# df_test['PREV_CREDLIM_CURRENT_CREDLIM_ACTIVE'] = (df_test['credlmt_1052A'] > df_test['credlmt_1052A']).replace({False: 0, True: 1})

df_train['DEBIT_COMIN_VS_CREDLIM_ACTIVE'] = np.where(df_train['credlmt_1052A'] == 0, 0, df_train['amtdebitincoming_4809443A'] / df_train['credlmt_1052A'])
# df_test['DEBIT_COMIN_VS_CREDLIM_ACTIVE'] = np.where(df_test['credlmt_1052A'] == 0, 0, df_test['amtdebitincoming_4809443A'] / df_test['credlmt_1052A'])

df_train['CREDIT_LOAN_ACT_VS_CHILD_NUM'] = np.where(df_train['childnum_21L'] == 0, df_train['credlmt_3940954A'], df_train['credlmt_3940954A'] / df_train['childnum_21L'])
# df_test['CREDIT_LOAN_ACT_VS_CHILD_NUM'] = np.where(df_test['childnum_21L'] == 0, df_test['credlmt_3940954A'], df_test['credlmt_3940954A'] / df_test['childnum_21L'])

df_train['CREDIT_LOAN_ACT_VS_CHILD_NUM'] = np.where(df_train['childnum_21L'] == 0, 0, df_train['credlmt_3940954A'] / df_train['childnum_21L'])
# df_test['CREDIT_LOAN_ACT_VS_CHILD_NUM'] = np.where(df_test['childnum_21L'] == 0, 0, df_test['credlmt_3940954A'] / df_test['childnum_21L'])

df_train['CREDIT_LOAN_ACT_VS_TOTAL_DEBT'] = np.where(df_train['totaldebt_9A'] == 0, 0, df_train['credlmt_3940954A'] / df_train['totaldebt_9A'])
# df_test['CREDIT_LOAN_ACT_VS_TOTAL_DEBT'] = np.where(df_test['totaldebt_9A'] == 0, 0, df_test['credlmt_3940954A'] / df_test['totaldebt_9A'])

df_train['DEBT_PAST_DUE_VS_TOTAL_DEBT'] = np.where(df_train['totaldebt_9A'] == 0, 0, df_train['debtpastduevalue_732A'] / df_train['totaldebt_9A'])
# df_test['DEBT_PAST_DUE_VS_TOTAL_DEBT'] = np.where(df_test['totaldebt_9A'] == 0, 0, df_test['debtpastduevalue_732A'] / df_test['totaldebt_9A'])

df_train['DEBT_PAST_DUE_VS_ANNUITY'] = np.where(df_train['annuity_780A'] == 0, 0, df_train['debtpastduevalue_732A'] / df_train['annuity_780A'])
# df_test['DEBT_PAST_DUE_VS_ANNUITY'] = np.where(df_test['annuity_780A'] == 0, 0, df_test['debtpastduevalue_732A'] / df_test['annuity_780A'])

df_train['DEPOSIT_DEBT_PAST_DUE'] = np.where(df_train['debtpastduevalue_732A'] == 0, 0, df_train['amtdepositbalance_4809441A'] / df_train['debtpastduevalue_732A'])
# df_test['DEPOSIT_DEBT_PAST_DUE'] = np.where(df_test['debtpastduevalue_732A'] == 0, 0, df_test['amtdepositbalance_4809441A'] / df_test['debtpastduevalue_732A'])

df_train['LAST_UPDATE'] = df_train['lastupdate_260D'] * -1/365
# df_test['LAST_UPDATE'] = df_test['lastupdate_260D'] * -1/365

# 1. Индекс платежной дисциплины
df_train['payment_discipline_index'] = (df_train['amtinstpaidbefduel24m_4187115A'] / 24) * 100
# df_test['payment_discipline_index'] = (df_test['amtinstpaidbefduel24m_4187115A'] / 24) * 100

# 2
df_train['annuity_amtin'] = np.where(df_train['amtinstpaidbefduel24m_4187115A'] == 0, 0, df_train['annuity_780A'] / df_train['amtinstpaidbefduel24m_4187115A'])
# df_test['annuity_amtin'] = np.where(df_test['amtinstpaidbefduel24m_4187115A'] == 0, 0, df_test['annuity_780A'] / df_test['amtinstpaidbefduel24m_4187115A'])

# 3
df_train['annuity_next_month_ratio'] = np.where(df_train['annuity_780A'] == 0, 0, df_train['annuitynextmonth_57A'] / df_train['annuity_780A'])
# df_test['annuity_next_month_ratio'] = np.where(df_test['annuity_780A'] == 0, 0, df_test['annuitynextmonth_57A'] / df_test['annuity_780A'])

# 4
df_train['annuity_year'] = df_train['annuity_780A'] * 12
# df_test['annuity_year'] = df_test['annuity_780A'] * 12

# 5
df_train['annuity_week'] = df_train['annuity_780A'] / 4
# df_test['annuity_week'] = df_test['annuity_780A'] / 4

# 6
df_train['application_week'] = df_train['applications30d_658L'] / 4
# df_test['application_week'] = df_test['applications30d_658L'] / 4

# 7
df_train['DAYS_BIRTH'] = df_train['birthdate_574D'] * -1 / 365
# df_test['DAYS_BIRTH'] = df_test['birthdate_574D'] * -1 / 365

# 8
df_train['CLIENTS_DAYS_BIRTH'] = df_train['dateofbirth_337D'] * -1/365
# df_test['CLIENTS_DAYS_BIRTH'] = df_test['dateofbirth_337D'] * -1/365

# 9
df_train['AVG_BUREAU_CONTRACTS_DAY'] = df_train['days120_123L'] / 120
# df_test['AVG_BUREAU_CONTRACTS_DAY'] = df_test['days120_123L'] / 120

list(df_train.select_dtypes(include=["int", 'float']).columns)
df_train['feature1'] = np.where(df_train['currdebt_22A'] == 0, 0, df_train['credamount_770A']/df_train['currdebt_22A'])
df_train['feature2'] = np.where(df_train['currdebt_22A'] == 0, 0, df_train['downpmt_116A']/df_train['currdebt_22A'])
df_train['feature3'] = np.where(df_train['totaldebt_9A'] == 0, 0, df_train['price_1097A']/df_train['totaldebt_9A'])
df_train['feature4'] = np.where(df_train['totalsettled_863A'] == 0, 0, df_train['annuitynextmonth_57A']/df_train['totalsettled_863A'])
df_train['feature5'] = np.where(df_train['currdebt_22A'] == 0, 0, df_train['annuitynextmonth_57A']/df_train['currdebt_22A'])
df_train['feature6'] = np.where(df_train['maxdebt4_972A'] == 0, 0, df_train['maininc_215A']/df_train['maxdebt4_972A'])
df_train['feature7'] = np.where(df_train['credamount_770A'] == 0, 0, df_train['disbursedcredamount_1113A']/df_train['credamount_770A'])
df_train['feature8'] = np.where(df_train['amtinstpaidbefduel24m_4187115A'] == 0, 0, df_train['annuitynextmonth_57A']/df_train['amtinstpaidbefduel24m_4187115A'])
df_train['feature9'] = np.where(df_train['totaldebt_9A'] == 0, 0, df_train['avgoutstandbalancel6m_4187114A']/df_train['totaldebt_9A'])
df_train['feature10'] = (df_train['credamount_770A'] > df_train['annuitynextmonth_57A']).replace({False: 0, True: 1})

# print((df_train == np.inf).any().any(), (df_train == -np.inf).any().any())
# df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# print((df_train == np.inf).any().any(), (df_train == -np.inf).any().any())

cols2drop = ['cacccardblochreas_147M', 'cancelreason_3545846M', 'contaddr_smempladdr_334L', 'contaddr_matchlist_1032L', 'credor_3940957M', ]
df_train.drop(columns=cols2drop, inplace=True)
# df_test.drop(columns=cols2drop, inplace=True)


# columns_to_drop = [column for column in df_train.columns if column.startswith('num_group')]
# for col in columns_to_drop:
#     if col in list(df_train.columns):
#         df_train.drop(columns=col, inplace=True)

df_train = pl.DataFrame(df_train)
# ===============================================================


df_train = df_train.pipe(Pipeline.filter_cols)
df_train, cat_cols = to_pandas(df_train)    
df_train = Utility.reduce_memory_usage(df_train, "df_train")

print("df_train shape:\t", df_train.shape)

del data_store
gc.collect()


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
# print("train data shape:\t", df_train.shape)


# print('cat_cols: ', cat_cols)
# print('df_train.columns: ', list(df_train.columns))
# df_train.to_csv('/home/xyli/kaggle/kaggle_HomeCredit/train389FE.csv', index=False)
# import sys
# sys.exit() 

# df_train = df_train[:50000]

# ======================================== 读入df_train =====================================


# ======================================== 导入配置 =====================================

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
# y_scan = df_train_scan["target"]

# weeks = df_train["WEEK_NUM"]
try:
    weeks = df_train["week_num"]
except:
    weeks = df_train["WEEK_NUM"] 

# try:
#     weeks_scan = df_train_scan["week_num"]
# except:
#     weeks_scan = df_train_scan["WEEK_NUM"] 
 
try:
    # df_train= df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
    df_train= df_train.drop(columns=["target"]) 
except:
    print("这个代码已经执行过1次了！")

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

# 470
df_train_470_lgbm = ['month', 'week_num', 'assignmentdate_238D', 'assignmentdate_4527235D', 'birthdate_574D', 'contractssum_5085716L', 'dateofbirth_337D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'description_5085714M', 'education_1103M', 'education_88M', 'firstquarter_103L', 'fourthquarter_440L', 'maritalst_385M', 'maritalst_893M', 'numberofqueries_373L', 'pmtaverage_3A', 'pmtaverage_4527227A', 'pmtcount_4527229L', 'pmtcount_693L', 'pmtscount_423L', 'pmtssum_45A', 'requesttype_4525192L', 'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D', 'secondquarter_766L', 'thirdquarter_1082L', 'actualdpdtolerance_344P', 'amtinstpaidbefduel24m_4187115A', 'annuity_780A', 'annuitynextmonth_57A', 'applicationcnt_361L', 'applications30d_658L', 'applicationscnt_1086L', 'applicationscnt_464L', 'applicationscnt_629L', 'applicationscnt_867L', 'avgdbddpdlast24m_3658932P', 'avgdbddpdlast3m_4187120P', 'avgdbdtollast24m_4525197P', 'avgdpdtolclosure24_3658938P', 'avginstallast24m_3658937A', 'avglnamtstart24m_4525187A', 'avgmaxdpdlast9m_3716943P', 'avgoutstandbalancel6m_4187114A', 'avgpmtlast12m_4525200A', 'bankacctype_710L', 'cardtype_51L', 'clientscnt12m_3712952L', 'clientscnt3m_3712950L', 'clientscnt6m_3712949L', 'clientscnt_100L', 'clientscnt_1022L', 'clientscnt_1071L', 'clientscnt_1130L', 'clientscnt_157L', 'clientscnt_257L', 'clientscnt_304L', 'clientscnt_360L', 'clientscnt_493L', 'clientscnt_533L', 'clientscnt_887L', 'clientscnt_946L', 'cntincpaycont9m_3716944L', 'cntpmts24_3658933L', 'commnoinclast6m_3546845L', 'credamount_770A', 'credtype_322L', 'currdebt_22A', 'currdebtcredtyperange_828A', 'datefirstoffer_1144D', 'datelastinstal40dpd_247D', 'datelastunpaid_3546854D', 'daysoverduetolerancedd_3976961L', 'deferredmnthsnum_166L', 'disbursedcredamount_1113A', 'disbursementtype_67L', 'downpmt_116A', 'dtlastpmtallstes_4499206D', 'eir_270L', 'equalitydataagreement_891L', 'firstclxcampaign_1125D', 'firstdatedue_489D', 'homephncnt_628L', 'inittransactionamount_650A', 'inittransactioncode_186L', 'interestrate_311L', 'isbidproduct_1095L', 'isdebitcard_729L', 'lastactivateddate_801D', 'lastapplicationdate_877D', 'lastapprcommoditycat_1041M', 'lastapprcredamount_781A', 'lastapprdate_640D', 'lastcancelreason_561M', 'lastdelinqdate_224D', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectcredamount_222A', 'lastrejectdate_50D', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'maininc_215A', 'mastercontrelectronic_519L', 'mastercontrexist_109L', 'maxannuity_159A', 'maxdbddpdlast1m_3658939P', 'maxdbddpdtollast12m_3658940P', 'maxdbddpdtollast6m_4187119P', 'maxdebt4_972A', 'maxdpdfrom6mto36m_3546853P', 'maxdpdinstldate_3546855D', 'maxdpdinstlnum_3546846P', 'maxdpdlast12m_727P', 'maxdpdlast24m_143P', 'maxdpdlast3m_392P', 'maxdpdlast6m_474P', 'maxdpdlast9m_1059P', 'maxdpdtolerance_374P', 'maxinstallast24m_3658928A', 'maxlnamtstart6m_4525199A', 'maxoutstandbalancel12m_4187113A', 'maxpmtlast3m_4525190A', 'mindbddpdlast24m_3658935P', 'mindbdtollast24m_4525191P', 'mobilephncnt_593L', 'monthsannuity_845L', 'numactivecreds_622L', 'numactivecredschannel_414L', 'numactiverelcontr_750L', 'numcontrs3months_479L', 'numincomingpmts_3546848L', 'numinstlallpaidearly3d_817L', 'numinstls_657L', 'numinstlsallpaid_934L', 'numinstlswithdpd10_728L', 'numinstlswithdpd5_4187116L', 'numinstlswithoutdpd_562L', 'numinstmatpaidtearly2d_4499204L', 'numinstpaid_4499208L', 'numinstpaidearly3d_3546850L', 'numinstpaidearly3dest_4493216L', 'numinstpaidearly5d_1087L', 'numinstpaidearly5dest_4493211L', 'numinstpaidearly5dobd_4499205L', 'numinstpaidearly_338L', 'numinstpaidearlyest_4493214L', 'numinstpaidlastcontr_4325080L', 'numinstpaidlate1d_3546852L', 'numinstregularpaid_973L', 'numinstregularpaidest_4493210L', 'numinsttopaygr_769L', 'numinsttopaygrest_4493213L', 'numinstunpaidmax_3546851L', 'numinstunpaidmaxest_4493212L', 'numnotactivated_1143L', 'numpmtchanneldd_318L', 'numrejects9m_859L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'pctinstlsallpaidearl3d_427L', 'pctinstlsallpaidlat10d_839L', 'pctinstlsallpaidlate1d_3546856L', 'pctinstlsallpaidlate4d_3546849L', 'pctinstlsallpaidlate6d_3546844L', 'pmtnum_254L', 'posfpd10lastmonth_333P', 'posfpd30lastmonth_3976960P', 'posfstqpd30lastmonth_3976962P', 'price_1097A', 'sellerplacecnt_915L', 'sellerplacescnt_216L', 'sumoutstandtotal_3546847A', 'sumoutstandtotalest_4493215A', 'totaldebt_9A', 'totalsettled_863A', 'totinstallast1m_4525188A', 'twobodfilling_608L', 'typesuite_864L', 'validfrom_1069D', 'max_actualdpd_943P', 'max_annuity_853A', 'max_credacc_actualbalance_314A', 'max_credacc_credlmt_575A', 'max_credacc_maxhisbal_375A', 'max_credacc_minhisbal_90A', 'max_credamount_590A', 'max_currdebt_94A', 'max_downpmt_134A', 'max_mainoccupationinc_437A', 'max_maxdpdtolerance_577P', 'max_outstandingdebt_522A', 'max_revolvingaccount_394A', 'mean_actualdpd_943P', 'mean_annuity_853A', 'mean_credacc_actualbalance_314A', 'mean_credacc_credlmt_575A', 'mean_credacc_maxhisbal_375A', 'mean_credacc_minhisbal_90A', 'mean_credamount_590A', 'mean_currdebt_94A', 'mean_downpmt_134A', 'mean_mainoccupationinc_437A', 'mean_maxdpdtolerance_577P', 'mean_outstandingdebt_522A', 'mean_revolvingaccount_394A', 'var_actualdpd_943P', 'var_annuity_853A', 'var_credacc_credlmt_575A', 'var_credamount_590A', 'var_currdebt_94A', 'var_downpmt_134A', 'var_mainoccupationinc_437A', 'var_maxdpdtolerance_577P', 'var_outstandingdebt_522A', 'max_approvaldate_319D', 'max_creationdate_885D', 'max_dateactivated_425D', 'max_dtlastpmt_581D', 'max_dtlastpmtallstes_3545839D', 'max_employedfrom_700D', 'max_firstnonzeroinstldate_307D', 'mean_approvaldate_319D', 'mean_creationdate_885D', 'mean_dateactivated_425D', 'mean_dtlastpmt_581D', 'mean_dtlastpmtallstes_3545839D', 'mean_employedfrom_700D', 'mean_firstnonzeroinstldate_307D', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'max_byoccupationinc_3656910L', 'max_childnum_21L', 'max_credacc_status_367L', 'max_credacc_transactions_402L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_pmtnum_8L', 'max_status_219L', 'max_tenor_203L', 'max_num_group1', 'max_amount_4527230A', 'mean_amount_4527230A', 'var_amount_4527230A', 'max_recorddate_4527225D', 'mean_recorddate_4527225D', 'max_num_group1_3', 'max_amount_4917619A', 'mean_amount_4917619A', 'var_amount_4917619A', 'max_deductiondate_4917603D', 'mean_deductiondate_4917603D', 'max_num_group1_4', 'max_pmtamount_36A', 'mean_pmtamount_36A', 'var_pmtamount_36A', 'max_processingdate_168D', 'mean_processingdate_168D', 'max_num_group1_5', 'max_credlmt_230A', 'max_credlmt_935A', 'max_debtoutstand_525A', 'max_debtoverdue_47A', 'max_dpdmax_139P', 'max_dpdmax_757P', 'max_instlamount_768A', 'max_instlamount_852A', 'max_monthlyinstlamount_332A', 'max_monthlyinstlamount_674A', 'max_outstandingamount_354A', 'max_outstandingamount_362A', 'max_overdueamount_31A', 'max_overdueamount_659A', 'max_overdueamountmax2_14A', 'max_overdueamountmax2_398A', 'max_overdueamountmax_155A', 'max_overdueamountmax_35A', 'max_residualamount_488A', 'max_residualamount_856A', 'max_totalamount_6A', 'max_totalamount_996A', 'max_totaldebtoverduevalue_178A', 'max_totaldebtoverduevalue_718A', 'max_totaloutstanddebtvalue_39A', 'max_totaloutstanddebtvalue_668A', 'mean_credlmt_230A', 'mean_credlmt_935A', 'mean_debtoutstand_525A', 'mean_debtoverdue_47A', 'mean_dpdmax_139P', 'mean_dpdmax_757P', 'mean_instlamount_768A', 'mean_instlamount_852A', 'mean_monthlyinstlamount_332A', 'mean_monthlyinstlamount_674A', 'mean_outstandingamount_354A', 'mean_outstandingamount_362A', 'mean_overdueamount_31A', 'mean_overdueamount_659A', 'mean_overdueamountmax2_14A', 'mean_overdueamountmax2_398A', 'mean_overdueamountmax_155A', 'mean_overdueamountmax_35A', 'mean_residualamount_488A', 'mean_residualamount_856A', 'mean_totalamount_6A', 'mean_totalamount_996A', 'mean_totaldebtoverduevalue_178A', 'mean_totaldebtoverduevalue_718A', 'mean_totaloutstanddebtvalue_39A', 'mean_totaloutstanddebtvalue_668A', 'var_credlmt_230A', 'var_credlmt_935A', 'var_dpdmax_139P', 'var_dpdmax_757P', 'var_instlamount_768A', 'var_instlamount_852A', 'var_monthlyinstlamount_332A', 'var_monthlyinstlamount_674A', 'var_outstandingamount_354A', 'var_outstandingamount_362A', 'var_overdueamount_31A', 'var_overdueamount_659A', 'var_overdueamountmax2_14A', 'var_overdueamountmax2_398A', 'var_overdueamountmax_155A', 'var_overdueamountmax_35A', 'var_residualamount_488A', 'var_residualamount_856A', 'var_totalamount_6A', 'var_totalamount_996A', 'max_dateofcredend_289D', 'max_dateofcredend_353D', 'max_dateofcredstart_181D', 'max_dateofcredstart_739D', 'max_dateofrealrepmt_138D', 'max_lastupdate_1112D', 'max_lastupdate_388D', 'max_numberofoverdueinstlmaxdat_148D', 'max_numberofoverdueinstlmaxdat_641D', 'max_overdueamountmax2date_1002D', 'max_overdueamountmax2date_1142D', 'max_refreshdate_3813885D', 'mean_dateofcredend_289D', 'mean_dateofcredend_353D', 'mean_dateofcredstart_181D', 'mean_dateofcredstart_739D', 'mean_dateofrealrepmt_138D', 'mean_lastupdate_1112D', 'mean_lastupdate_388D', 'mean_numberofoverdueinstlmaxdat_148D', 'mean_numberofoverdueinstlmaxdat_641D', 'mean_overdueamountmax2date_1002D', 'mean_overdueamountmax2date_1142D', 'mean_refreshdate_3813885D', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'max_annualeffectiverate_199L', 'max_annualeffectiverate_63L', 'max_contractsum_5085717L', 'max_dpdmaxdatemonth_442T', 'max_dpdmaxdatemonth_89T', 'max_dpdmaxdateyear_596T', 'max_dpdmaxdateyear_896T', 'max_nominalrate_281L', 'max_nominalrate_498L', 'max_numberofcontrsvalue_258L', 'max_numberofcontrsvalue_358L', 'max_numberofinstls_229L', 'max_numberofinstls_320L', 'max_numberofoutstandinstls_520L', 'max_numberofoutstandinstls_59L', 'max_numberofoverdueinstlmax_1039L', 'max_numberofoverdueinstlmax_1151L', 'max_numberofoverdueinstls_725L', 'max_numberofoverdueinstls_834L', 'max_overdueamountmaxdatemonth_284T', 'max_overdueamountmaxdatemonth_365T', 'max_overdueamountmaxdateyear_2T', 'max_overdueamountmaxdateyear_994T', 'max_periodicityofpmts_1102L', 'max_periodicityofpmts_837L', 'max_prolongationcount_1120L', 'max_num_group1_6', 'max_mainoccupationinc_384A', 'mean_mainoccupationinc_384A', 'max_birth_259D', 'max_empl_employedfrom_271D', 'mean_birth_259D', 'mean_empl_employedfrom_271D', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_housetype_905L', 'max_incometype_1044T', 'max_personindex_1023L', 'max_persontype_1072L', 'max_persontype_792L', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'max_num_group1_9', 'max_amount_416A', 'mean_amount_416A', 'max_openingdate_313D', 'mean_openingdate_313D', 'max_num_group1_10', 'max_openingdate_857D', 'mean_openingdate_857D', 'max_num_group1_11', 'max_pmts_dpd_1073P', 'max_pmts_dpd_303P', 'max_pmts_overdue_1140A', 'max_pmts_overdue_1152A', 'mean_pmts_dpd_1073P', 'mean_pmts_dpd_303P', 'mean_pmts_overdue_1140A', 'mean_pmts_overdue_1152A', 'var_pmts_dpd_1073P', 'var_pmts_dpd_303P', 'var_pmts_overdue_1140A', 'var_pmts_overdue_1152A', 'max_collater_typofvalofguarant_298M', 'max_collater_typofvalofguarant_407M', 'max_collaterals_typeofguarante_359M', 'max_collaterals_typeofguarante_669M', 'max_subjectroles_name_541M', 'max_subjectroles_name_838M', 'max_collater_valueofguarantee_1124L', 'max_collater_valueofguarantee_876L', 'max_pmts_month_158T', 'max_pmts_month_706T', 'max_pmts_year_1139T', 'max_pmts_year_507T', 'max_num_group1_12', 'max_num_group2', 'year', 'day']
# 72
cat_cols_470_lgbm = ['description_5085714M', 'education_1103M', 'education_88M', 'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L', 'bankacctype_710L', 'cardtype_51L', 'credtype_322L', 'disbursementtype_67L', 'equalitydataagreement_891L', 'inittransactioncode_186L', 'isdebitcard_729L', 'lastapprcommoditycat_1041M', 'lastcancelreason_561M', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'twobodfilling_608L', 'typesuite_864L', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'max_credacc_status_367L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_status_219L', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_housetype_905L', 'max_incometype_1044T', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'max_collater_typofvalofguarant_298M', 'max_collater_typofvalofguarant_407M', 'max_collaterals_typeofguarante_359M', 'max_collaterals_typeofguarante_669M', 'max_subjectroles_name_541M', 'max_subjectroles_name_838M']
#
non_cat_cols_cat_470_type = ['uint32', 'uint8', 'float64', 'float64', 'float64', 'float32', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float32', 'float64', 'float32', 'float64', 'float64', 'float32', 'float32', 'float32', 'bool', 'float64', 'float64', 'float32', 'float64', 'float64', 'float32', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float32', 'float32', 'float32', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float32', 'float32', 'int16', 'float64', 'int16', 'float64', 'float32', 'float32', 'float32', 'uint8', 'float32', 'float32', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float64', 'float64', 'uint16', 'uint8']
non_cat_cols_470_lgbm = ['month', 'week_num', 'assignmentdate_238D', 'assignmentdate_4527235D', 'birthdate_574D', 'contractssum_5085716L', 'dateofbirth_337D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'firstquarter_103L', 'fourthquarter_440L', 'numberofqueries_373L', 'pmtaverage_3A', 'pmtaverage_4527227A', 'pmtcount_4527229L', 'pmtcount_693L', 'pmtscount_423L', 'pmtssum_45A', 'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D', 'secondquarter_766L', 'thirdquarter_1082L', 'actualdpdtolerance_344P', 'amtinstpaidbefduel24m_4187115A', 'annuity_780A', 'annuitynextmonth_57A', 'applicationcnt_361L', 'applications30d_658L', 'applicationscnt_1086L', 'applicationscnt_464L', 'applicationscnt_629L', 'applicationscnt_867L', 'avgdbddpdlast24m_3658932P', 'avgdbddpdlast3m_4187120P', 'avgdbdtollast24m_4525197P', 'avgdpdtolclosure24_3658938P', 'avginstallast24m_3658937A', 'avglnamtstart24m_4525187A', 'avgmaxdpdlast9m_3716943P', 'avgoutstandbalancel6m_4187114A', 'avgpmtlast12m_4525200A', 'clientscnt12m_3712952L', 'clientscnt3m_3712950L', 'clientscnt6m_3712949L', 'clientscnt_100L', 'clientscnt_1022L', 'clientscnt_1071L', 'clientscnt_1130L', 'clientscnt_157L', 'clientscnt_257L', 'clientscnt_304L', 'clientscnt_360L', 'clientscnt_493L', 'clientscnt_533L', 'clientscnt_887L', 'clientscnt_946L', 'cntincpaycont9m_3716944L', 'cntpmts24_3658933L', 'commnoinclast6m_3546845L', 'credamount_770A', 'currdebt_22A', 'currdebtcredtyperange_828A', 'datefirstoffer_1144D', 'datelastinstal40dpd_247D', 'datelastunpaid_3546854D', 'daysoverduetolerancedd_3976961L', 'deferredmnthsnum_166L', 'disbursedcredamount_1113A', 'downpmt_116A', 'dtlastpmtallstes_4499206D', 'eir_270L', 'firstclxcampaign_1125D', 'firstdatedue_489D', 'homephncnt_628L', 'inittransactionamount_650A', 'interestrate_311L', 'isbidproduct_1095L', 'lastactivateddate_801D', 'lastapplicationdate_877D', 'lastapprcredamount_781A', 'lastapprdate_640D', 'lastdelinqdate_224D', 'lastrejectcredamount_222A', 'lastrejectdate_50D', 'maininc_215A', 'mastercontrelectronic_519L', 'mastercontrexist_109L', 'maxannuity_159A', 'maxdbddpdlast1m_3658939P', 'maxdbddpdtollast12m_3658940P', 'maxdbddpdtollast6m_4187119P', 'maxdebt4_972A', 'maxdpdfrom6mto36m_3546853P', 'maxdpdinstldate_3546855D', 'maxdpdinstlnum_3546846P', 'maxdpdlast12m_727P', 'maxdpdlast24m_143P', 'maxdpdlast3m_392P', 'maxdpdlast6m_474P', 'maxdpdlast9m_1059P', 'maxdpdtolerance_374P', 'maxinstallast24m_3658928A', 'maxlnamtstart6m_4525199A', 'maxoutstandbalancel12m_4187113A', 'maxpmtlast3m_4525190A', 'mindbddpdlast24m_3658935P', 'mindbdtollast24m_4525191P', 'mobilephncnt_593L', 'monthsannuity_845L', 'numactivecreds_622L', 'numactivecredschannel_414L', 'numactiverelcontr_750L', 'numcontrs3months_479L', 'numincomingpmts_3546848L', 'numinstlallpaidearly3d_817L', 'numinstls_657L', 'numinstlsallpaid_934L', 'numinstlswithdpd10_728L', 'numinstlswithdpd5_4187116L', 'numinstlswithoutdpd_562L', 'numinstmatpaidtearly2d_4499204L', 'numinstpaid_4499208L', 'numinstpaidearly3d_3546850L', 'numinstpaidearly3dest_4493216L', 'numinstpaidearly5d_1087L', 'numinstpaidearly5dest_4493211L', 'numinstpaidearly5dobd_4499205L', 'numinstpaidearly_338L', 'numinstpaidearlyest_4493214L', 'numinstpaidlastcontr_4325080L', 'numinstpaidlate1d_3546852L', 'numinstregularpaid_973L', 'numinstregularpaidest_4493210L', 'numinsttopaygr_769L', 'numinsttopaygrest_4493213L', 'numinstunpaidmax_3546851L', 'numinstunpaidmaxest_4493212L', 'numnotactivated_1143L', 'numpmtchanneldd_318L', 'numrejects9m_859L', 'pctinstlsallpaidearl3d_427L', 'pctinstlsallpaidlat10d_839L', 'pctinstlsallpaidlate1d_3546856L', 'pctinstlsallpaidlate4d_3546849L', 'pctinstlsallpaidlate6d_3546844L', 'pmtnum_254L', 'posfpd10lastmonth_333P', 'posfpd30lastmonth_3976960P', 'posfstqpd30lastmonth_3976962P', 'price_1097A', 'sellerplacecnt_915L', 'sellerplacescnt_216L', 'sumoutstandtotal_3546847A', 'sumoutstandtotalest_4493215A', 'totaldebt_9A', 'totalsettled_863A', 'totinstallast1m_4525188A', 'validfrom_1069D', 'max_actualdpd_943P', 'max_annuity_853A', 'max_credacc_actualbalance_314A', 'max_credacc_credlmt_575A', 'max_credacc_maxhisbal_375A', 'max_credacc_minhisbal_90A', 'max_credamount_590A', 'max_currdebt_94A', 'max_downpmt_134A', 'max_mainoccupationinc_437A', 'max_maxdpdtolerance_577P', 'max_outstandingdebt_522A', 'max_revolvingaccount_394A', 'mean_actualdpd_943P', 'mean_annuity_853A', 'mean_credacc_actualbalance_314A', 'mean_credacc_credlmt_575A', 'mean_credacc_maxhisbal_375A', 'mean_credacc_minhisbal_90A', 'mean_credamount_590A', 'mean_currdebt_94A', 'mean_downpmt_134A', 'mean_mainoccupationinc_437A', 'mean_maxdpdtolerance_577P', 'mean_outstandingdebt_522A', 'mean_revolvingaccount_394A', 'var_actualdpd_943P', 'var_annuity_853A', 'var_credacc_credlmt_575A', 'var_credamount_590A', 'var_currdebt_94A', 'var_downpmt_134A', 'var_mainoccupationinc_437A', 'var_maxdpdtolerance_577P', 'var_outstandingdebt_522A', 'max_approvaldate_319D', 'max_creationdate_885D', 'max_dateactivated_425D', 'max_dtlastpmt_581D', 'max_dtlastpmtallstes_3545839D', 'max_employedfrom_700D', 'max_firstnonzeroinstldate_307D', 'mean_approvaldate_319D', 'mean_creationdate_885D', 'mean_dateactivated_425D', 'mean_dtlastpmt_581D', 'mean_dtlastpmtallstes_3545839D', 'mean_employedfrom_700D', 'mean_firstnonzeroinstldate_307D', 'max_byoccupationinc_3656910L', 'max_childnum_21L', 'max_credacc_transactions_402L', 'max_pmtnum_8L', 'max_tenor_203L', 'max_num_group1', 'max_amount_4527230A', 'mean_amount_4527230A', 'var_amount_4527230A', 'max_recorddate_4527225D', 'mean_recorddate_4527225D', 'max_num_group1_3', 'max_amount_4917619A', 'mean_amount_4917619A', 'var_amount_4917619A', 'max_deductiondate_4917603D', 'mean_deductiondate_4917603D', 'max_num_group1_4', 'max_pmtamount_36A', 'mean_pmtamount_36A', 'var_pmtamount_36A', 'max_processingdate_168D', 'mean_processingdate_168D', 'max_num_group1_5', 'max_credlmt_230A', 'max_credlmt_935A', 'max_debtoutstand_525A', 'max_debtoverdue_47A', 'max_dpdmax_139P', 'max_dpdmax_757P', 'max_instlamount_768A', 'max_instlamount_852A', 'max_monthlyinstlamount_332A', 'max_monthlyinstlamount_674A', 'max_outstandingamount_354A', 'max_outstandingamount_362A', 'max_overdueamount_31A', 'max_overdueamount_659A', 'max_overdueamountmax2_14A', 'max_overdueamountmax2_398A', 'max_overdueamountmax_155A', 'max_overdueamountmax_35A', 'max_residualamount_488A', 'max_residualamount_856A', 'max_totalamount_6A', 'max_totalamount_996A', 'max_totaldebtoverduevalue_178A', 'max_totaldebtoverduevalue_718A', 'max_totaloutstanddebtvalue_39A', 'max_totaloutstanddebtvalue_668A', 'mean_credlmt_230A', 'mean_credlmt_935A', 'mean_debtoutstand_525A', 'mean_debtoverdue_47A', 'mean_dpdmax_139P', 'mean_dpdmax_757P', 'mean_instlamount_768A', 'mean_instlamount_852A', 'mean_monthlyinstlamount_332A', 'mean_monthlyinstlamount_674A', 'mean_outstandingamount_354A', 'mean_outstandingamount_362A', 'mean_overdueamount_31A', 'mean_overdueamount_659A', 'mean_overdueamountmax2_14A', 'mean_overdueamountmax2_398A', 'mean_overdueamountmax_155A', 'mean_overdueamountmax_35A', 'mean_residualamount_488A', 'mean_residualamount_856A', 'mean_totalamount_6A', 'mean_totalamount_996A', 'mean_totaldebtoverduevalue_178A', 'mean_totaldebtoverduevalue_718A', 'mean_totaloutstanddebtvalue_39A', 'mean_totaloutstanddebtvalue_668A', 'var_credlmt_230A', 'var_credlmt_935A', 'var_dpdmax_139P', 'var_dpdmax_757P', 'var_instlamount_768A', 'var_instlamount_852A', 'var_monthlyinstlamount_332A', 'var_monthlyinstlamount_674A', 'var_outstandingamount_354A', 'var_outstandingamount_362A', 'var_overdueamount_31A', 'var_overdueamount_659A', 'var_overdueamountmax2_14A', 'var_overdueamountmax2_398A', 'var_overdueamountmax_155A', 'var_overdueamountmax_35A', 'var_residualamount_488A', 'var_residualamount_856A', 'var_totalamount_6A', 'var_totalamount_996A', 'max_dateofcredend_289D', 'max_dateofcredend_353D', 'max_dateofcredstart_181D', 'max_dateofcredstart_739D', 'max_dateofrealrepmt_138D', 'max_lastupdate_1112D', 'max_lastupdate_388D', 'max_numberofoverdueinstlmaxdat_148D', 'max_numberofoverdueinstlmaxdat_641D', 'max_overdueamountmax2date_1002D', 'max_overdueamountmax2date_1142D', 'max_refreshdate_3813885D', 'mean_dateofcredend_289D', 'mean_dateofcredend_353D', 'mean_dateofcredstart_181D', 'mean_dateofcredstart_739D', 'mean_dateofrealrepmt_138D', 'mean_lastupdate_1112D', 'mean_lastupdate_388D', 'mean_numberofoverdueinstlmaxdat_148D', 'mean_numberofoverdueinstlmaxdat_641D', 'mean_overdueamountmax2date_1002D', 'mean_overdueamountmax2date_1142D', 'mean_refreshdate_3813885D', 'max_annualeffectiverate_199L', 'max_annualeffectiverate_63L', 'max_contractsum_5085717L', 'max_dpdmaxdatemonth_442T', 'max_dpdmaxdatemonth_89T', 'max_dpdmaxdateyear_596T', 'max_dpdmaxdateyear_896T', 'max_nominalrate_281L', 'max_nominalrate_498L', 'max_numberofcontrsvalue_258L', 'max_numberofcontrsvalue_358L', 'max_numberofinstls_229L', 'max_numberofinstls_320L', 'max_numberofoutstandinstls_520L', 'max_numberofoutstandinstls_59L', 'max_numberofoverdueinstlmax_1039L', 'max_numberofoverdueinstlmax_1151L', 'max_numberofoverdueinstls_725L', 'max_numberofoverdueinstls_834L', 'max_overdueamountmaxdatemonth_284T', 'max_overdueamountmaxdatemonth_365T', 'max_overdueamountmaxdateyear_2T', 'max_overdueamountmaxdateyear_994T', 'max_periodicityofpmts_1102L', 'max_periodicityofpmts_837L', 'max_prolongationcount_1120L', 'max_num_group1_6', 'max_mainoccupationinc_384A', 'mean_mainoccupationinc_384A', 'max_birth_259D', 'max_empl_employedfrom_271D', 'mean_birth_259D', 'mean_empl_employedfrom_271D', 'max_personindex_1023L', 'max_persontype_1072L', 'max_persontype_792L', 'max_num_group1_9', 'max_amount_416A', 'mean_amount_416A', 'max_openingdate_313D', 'mean_openingdate_313D', 'max_num_group1_10', 'max_openingdate_857D', 'mean_openingdate_857D', 'max_num_group1_11', 'max_pmts_dpd_1073P', 'max_pmts_dpd_303P', 'max_pmts_overdue_1140A', 'max_pmts_overdue_1152A', 'mean_pmts_dpd_1073P', 'mean_pmts_dpd_303P', 'mean_pmts_overdue_1140A', 'mean_pmts_overdue_1152A', 'var_pmts_dpd_1073P', 'var_pmts_dpd_303P', 'var_pmts_overdue_1140A', 'var_pmts_overdue_1152A', 'max_collater_valueofguarantee_1124L', 'max_collater_valueofguarantee_876L', 'max_pmts_month_158T', 'max_pmts_month_706T', 'max_pmts_year_1139T', 'max_pmts_year_507T', 'max_num_group1_12', 'max_num_group2', 'year', 'day']


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

# ZhiXing Jiang
# 66个 
cat_cols_338 = ['description_5085714M', 'education_1103M', 'education_88M', 'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L', 'max_lastapprcommoditycat_1041M', 'max_lastcancelreason_561M', 'max_lastrejectcommoditycat_161M', 'max_lastrejectcommodtypec_5251769M', 'max_lastrejectreason_759M', 'max_lastrejectreasonclient_4145040M', 'max_bankacctype_710L', 'max_cardtype_51L', 'max_credtype_322L', 'max_disbursementtype_67L', 'max_equalitydataagreement_891L', 'max_inittransactioncode_186L', 'max_isdebitcard_729L', 'max_lastst_736L', 'max_opencred_647L', 'max_paytype1st_925L', 'max_paytype_783L', 'max_twobodfilling_608L', 'max_typesuite_864L', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'max_credacc_status_367L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_status_219L', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_housetype_905L', 'max_incometype_1044T', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L']
# 338个
df_train_338 = ['WEEK_NUM', 'month_decision', 'weekday_decision', 'assignmentdate_238D', 'assignmentdate_4527235D', 'birthdate_574D', 'contractssum_5085716L', 'dateofbirth_337D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'description_5085714M', 'education_1103M', 'education_88M', 'firstquarter_103L', 'fourthquarter_440L', 'maritalst_385M', 'maritalst_893M', 'numberofqueries_373L', 'pmtaverage_3A', 'pmtaverage_4527227A', 'pmtcount_4527229L', 'pmtcount_693L', 'pmtscount_423L', 'pmtssum_45A', 'requesttype_4525192L', 'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D', 'secondquarter_766L', 'thirdquarter_1082L', 'max_actualdpdtolerance_344P', 'max_amtinstpaidbefduel24m_4187115A', 'max_annuity_780A', 'max_annuitynextmonth_57A', 'max_avgdbddpdlast24m_3658932P', 'max_avgdbddpdlast3m_4187120P', 'max_avgdbdtollast24m_4525197P', 'max_avgdpdtolclosure24_3658938P', 'max_avginstallast24m_3658937A', 'max_avglnamtstart24m_4525187A', 'max_avgmaxdpdlast9m_3716943P', 'max_avgoutstandbalancel6m_4187114A', 'max_avgpmtlast12m_4525200A', 'max_credamount_770A', 'max_currdebt_22A', 'max_currdebtcredtyperange_828A', 'max_disbursedcredamount_1113A', 'max_downpmt_116A', 'max_inittransactionamount_650A', 'max_lastapprcredamount_781A', 'max_lastrejectcredamount_222A', 'max_maininc_215A', 'max_maxannuity_159A', 'max_maxdbddpdlast1m_3658939P', 'max_maxdbddpdtollast12m_3658940P', 'max_maxdbddpdtollast6m_4187119P', 'max_maxdebt4_972A', 'max_maxdpdfrom6mto36m_3546853P', 'max_maxdpdinstlnum_3546846P', 'max_maxdpdlast12m_727P', 'max_maxdpdlast24m_143P', 'max_maxdpdlast3m_392P', 'max_maxdpdlast6m_474P', 'max_maxdpdlast9m_1059P', 'max_maxdpdtolerance_374P', 'max_maxinstallast24m_3658928A', 'max_maxlnamtstart6m_4525199A', 'max_maxoutstandbalancel12m_4187113A', 'max_maxpmtlast3m_4525190A', 'max_mindbddpdlast24m_3658935P', 'max_mindbdtollast24m_4525191P', 'max_posfpd10lastmonth_333P', 'max_posfpd30lastmonth_3976960P', 'max_posfstqpd30lastmonth_3976962P', 'max_price_1097A', 'max_sumoutstandtotal_3546847A', 'max_sumoutstandtotalest_4493215A', 'max_totaldebt_9A', 'max_totalsettled_863A', 'max_totinstallast1m_4525188A', 'max_datefirstoffer_1144D', 'max_datelastinstal40dpd_247D', 'max_datelastunpaid_3546854D', 'max_dtlastpmtallstes_4499206D', 'max_firstclxcampaign_1125D', 'max_firstdatedue_489D', 'max_lastactivateddate_801D', 'max_lastapplicationdate_877D', 'max_lastapprdate_640D', 'max_lastdelinqdate_224D', 'max_lastrejectdate_50D', 'max_maxdpdinstldate_3546855D', 'max_validfrom_1069D', 'max_lastapprcommoditycat_1041M', 'max_lastcancelreason_561M', 'max_lastrejectcommoditycat_161M', 'max_lastrejectcommodtypec_5251769M', 'max_lastrejectreason_759M', 'max_lastrejectreasonclient_4145040M', 'max_applicationcnt_361L', 'max_applications30d_658L', 'max_applicationscnt_1086L', 'max_applicationscnt_464L', 'max_applicationscnt_629L', 'max_applicationscnt_867L', 'max_bankacctype_710L', 'max_cardtype_51L', 'max_clientscnt12m_3712952L', 'max_clientscnt3m_3712950L', 'max_clientscnt6m_3712949L', 'max_clientscnt_100L', 'max_clientscnt_1022L', 'max_clientscnt_1071L', 'max_clientscnt_1130L', 'max_clientscnt_157L', 'max_clientscnt_257L', 'max_clientscnt_304L', 'max_clientscnt_360L', 'max_clientscnt_493L', 'max_clientscnt_533L', 'max_clientscnt_887L', 'max_clientscnt_946L', 'max_cntincpaycont9m_3716944L', 'max_cntpmts24_3658933L', 'max_commnoinclast6m_3546845L', 'max_credtype_322L', 'max_daysoverduetolerancedd_3976961L', 'max_deferredmnthsnum_166L', 'max_disbursementtype_67L', 'max_eir_270L', 'max_equalitydataagreement_891L', 'max_homephncnt_628L', 'max_inittransactioncode_186L', 'max_interestrate_311L', 'max_isbidproduct_1095L', 'max_isdebitcard_729L', 'max_lastst_736L', 'max_mastercontrelectronic_519L', 'max_mastercontrexist_109L', 'max_mobilephncnt_593L', 'max_monthsannuity_845L', 'max_numactivecreds_622L', 'max_numactivecredschannel_414L', 'max_numactiverelcontr_750L', 'max_numcontrs3months_479L', 'max_numincomingpmts_3546848L', 'max_numinstlallpaidearly3d_817L', 'max_numinstls_657L', 'max_numinstlsallpaid_934L', 'max_numinstlswithdpd10_728L', 'max_numinstlswithdpd5_4187116L', 'max_numinstlswithoutdpd_562L', 'max_numinstmatpaidtearly2d_4499204L', 'max_numinstpaid_4499208L', 'max_numinstpaidearly3d_3546850L', 'max_numinstpaidearly3dest_4493216L', 'max_numinstpaidearly5d_1087L', 'max_numinstpaidearly5dest_4493211L', 'max_numinstpaidearly5dobd_4499205L', 'max_numinstpaidearly_338L', 'max_numinstpaidearlyest_4493214L', 'max_numinstpaidlastcontr_4325080L', 'max_numinstpaidlate1d_3546852L', 'max_numinstregularpaid_973L', 'max_numinstregularpaidest_4493210L', 'max_numinsttopaygr_769L', 'max_numinsttopaygrest_4493213L', 'max_numinstunpaidmax_3546851L', 'max_numinstunpaidmaxest_4493212L', 'max_numnotactivated_1143L', 'max_numpmtchanneldd_318L', 'max_numrejects9m_859L', 'max_opencred_647L', 'max_paytype1st_925L', 'max_paytype_783L', 'max_pctinstlsallpaidearl3d_427L', 'max_pctinstlsallpaidlat10d_839L', 'max_pctinstlsallpaidlate1d_3546856L', 'max_pctinstlsallpaidlate4d_3546849L', 'max_pctinstlsallpaidlate6d_3546844L', 'max_pmtnum_254L', 'max_sellerplacecnt_915L', 'max_sellerplacescnt_216L', 'max_twobodfilling_608L', 'max_typesuite_864L', 'max_actualdpd_943P', 'max_annuity_853A', 'max_credacc_actualbalance_314A', 'max_credacc_credlmt_575A', 'max_credacc_maxhisbal_375A', 'max_credacc_minhisbal_90A', 'max_credamount_590A', 'max_currdebt_94A', 'max_downpmt_134A', 'max_mainoccupationinc_437A', 'max_maxdpdtolerance_577P', 'max_outstandingdebt_522A', 'max_revolvingaccount_394A', 'max_approvaldate_319D', 'max_creationdate_885D', 'max_dateactivated_425D', 'max_dtlastpmt_581D', 'max_dtlastpmtallstes_3545839D', 'max_employedfrom_700D', 'max_firstnonzeroinstldate_307D', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'max_byoccupationinc_3656910L', 'max_childnum_21L', 'max_credacc_status_367L', 'max_credacc_transactions_402L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_pmtnum_8L', 'max_status_219L', 'max_tenor_203L', 'max_num_group1', 'max_amount_4527230A', 'max_recorddate_4527225D', 'max_num_group1_3', 'max_pmtamount_36A', 'max_processingdate_168D', 'max_num_group1_5', 'max_credlmt_230A', 'max_credlmt_935A', 'max_debtoutstand_525A', 'max_debtoverdue_47A', 'max_dpdmax_139P', 'max_dpdmax_757P', 'max_instlamount_768A', 'max_instlamount_852A', 'max_monthlyinstlamount_332A', 'max_monthlyinstlamount_674A', 'max_outstandingamount_354A', 'max_outstandingamount_362A', 'max_overdueamount_31A', 'max_overdueamount_659A', 'max_overdueamountmax2_14A', 'max_overdueamountmax2_398A', 'max_overdueamountmax_155A', 'max_overdueamountmax_35A', 'max_residualamount_488A', 'max_residualamount_856A', 'max_totalamount_6A', 'max_totalamount_996A', 'max_totaldebtoverduevalue_178A', 'max_totaldebtoverduevalue_718A', 'max_totaloutstanddebtvalue_39A', 'max_totaloutstanddebtvalue_668A', 'max_dateofcredend_289D', 'max_dateofcredend_353D', 'max_dateofcredstart_181D', 'max_dateofcredstart_739D', 'max_dateofrealrepmt_138D', 'max_lastupdate_1112D', 'max_lastupdate_388D', 'max_numberofoverdueinstlmaxdat_148D', 'max_numberofoverdueinstlmaxdat_641D', 'max_overdueamountmax2date_1002D', 'max_overdueamountmax2date_1142D', 'max_refreshdate_3813885D', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'max_annualeffectiverate_199L', 'max_annualeffectiverate_63L', 'max_contractsum_5085717L', 'max_dpdmaxdatemonth_442T', 'max_dpdmaxdatemonth_89T', 'max_dpdmaxdateyear_596T', 'max_dpdmaxdateyear_896T', 'max_nominalrate_281L', 'max_nominalrate_498L', 'max_numberofcontrsvalue_258L', 'max_numberofcontrsvalue_358L', 'max_numberofinstls_229L', 'max_numberofinstls_320L', 'max_numberofoutstandinstls_520L', 'max_numberofoutstandinstls_59L', 'max_numberofoverdueinstlmax_1039L', 'max_numberofoverdueinstlmax_1151L', 'max_numberofoverdueinstls_725L', 'max_numberofoverdueinstls_834L', 'max_overdueamountmaxdatemonth_284T', 'max_overdueamountmaxdatemonth_365T', 'max_overdueamountmaxdateyear_2T', 'max_overdueamountmaxdateyear_994T', 'max_periodicityofpmts_1102L', 'max_periodicityofpmts_837L', 'max_prolongationcount_1120L', 'max_num_group1_6', 'max_mainoccupationinc_384A', 'max_birth_259D', 'max_empl_employedfrom_271D', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_housetype_905L', 'max_incometype_1044T', 'max_personindex_1023L', 'max_persontype_1072L', 'max_persontype_792L', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'max_num_group1_9', 'max_amount_416A', 'max_openingdate_313D', 'max_num_group1_10', 'max_openingdate_857D', 'max_num_group1_11']

# ZhiXing Jiangv2
# 66个 
cat_cols_344 = ['description_5085714M', 'education_1103M', 'education_88M', 'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L', 'max_lastapprcommoditycat_1041M', 'max_lastcancelreason_561M', 'max_lastrejectcommoditycat_161M', 'max_lastrejectcommodtypec_5251769M', 'max_lastrejectreason_759M', 'max_lastrejectreasonclient_4145040M', 'max_bankacctype_710L', 'max_cardtype_51L', 'max_credtype_322L', 'max_disbursementtype_67L', 'max_equalitydataagreement_891L', 'max_inittransactioncode_186L', 'max_isdebitcard_729L', 'max_lastst_736L', 'max_opencred_647L', 'max_paytype1st_925L', 'max_paytype_783L', 'max_twobodfilling_608L', 'max_typesuite_864L', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'max_credacc_status_367L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_status_219L', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_housetype_905L', 'max_incometype_1044T', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L']
# 344个
df_train_344 = ['WEEK_NUM', 'month_decision', 'weekday_decision', 'assignmentdate_238D', 'assignmentdate_4527235D', 'birthdate_574D', 'contractssum_5085716L', 'dateofbirth_337D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'description_5085714M', 'education_1103M', 'education_88M', 'firstquarter_103L', 'fourthquarter_440L', 'maritalst_385M', 'maritalst_893M', 'numberofqueries_373L', 'pmtaverage_3A', 'pmtaverage_4527227A', 'pmtcount_4527229L', 'pmtcount_693L', 'pmtscount_423L', 'pmtssum_45A', 'requesttype_4525192L', 'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D', 'secondquarter_766L', 'thirdquarter_1082L', 'max_actualdpdtolerance_344P', 'max_amtinstpaidbefduel24m_4187115A', 'max_annuity_780A', 'max_annuitynextmonth_57A', 'max_avgdbddpdlast24m_3658932P', 'max_avgdbddpdlast3m_4187120P', 'max_avgdbdtollast24m_4525197P', 'max_avgdpdtolclosure24_3658938P', 'max_avginstallast24m_3658937A', 'max_avglnamtstart24m_4525187A', 'max_avgmaxdpdlast9m_3716943P', 'max_avgoutstandbalancel6m_4187114A', 'max_avgpmtlast12m_4525200A', 'max_credamount_770A', 'max_currdebt_22A', 'max_currdebtcredtyperange_828A', 'max_disbursedcredamount_1113A', 'max_downpmt_116A', 'max_inittransactionamount_650A', 'max_lastapprcredamount_781A', 'max_lastrejectcredamount_222A', 'max_maininc_215A', 'max_maxannuity_159A', 'max_maxdbddpdlast1m_3658939P', 'max_maxdbddpdtollast12m_3658940P', 'max_maxdbddpdtollast6m_4187119P', 'max_maxdebt4_972A', 'max_maxdpdfrom6mto36m_3546853P', 'max_maxdpdinstlnum_3546846P', 'max_maxdpdlast12m_727P', 'max_maxdpdlast24m_143P', 'max_maxdpdlast3m_392P', 'max_maxdpdlast6m_474P', 'max_maxdpdlast9m_1059P', 'max_maxdpdtolerance_374P', 'max_maxinstallast24m_3658928A', 'max_maxlnamtstart6m_4525199A', 'max_maxoutstandbalancel12m_4187113A', 'max_maxpmtlast3m_4525190A', 'max_mindbddpdlast24m_3658935P', 'max_mindbdtollast24m_4525191P', 'max_posfpd10lastmonth_333P', 'max_posfpd30lastmonth_3976960P', 'max_posfstqpd30lastmonth_3976962P', 'max_price_1097A', 'max_sumoutstandtotal_3546847A', 'max_sumoutstandtotalest_4493215A', 'max_totaldebt_9A', 'max_totalsettled_863A', 'max_totinstallast1m_4525188A', 'max_datefirstoffer_1144D', 'max_datelastinstal40dpd_247D', 'max_datelastunpaid_3546854D', 'max_dtlastpmtallstes_4499206D', 'max_firstclxcampaign_1125D', 'max_firstdatedue_489D', 'max_lastactivateddate_801D', 'max_lastapplicationdate_877D', 'max_lastapprdate_640D', 'max_lastdelinqdate_224D', 'max_lastrejectdate_50D', 'max_maxdpdinstldate_3546855D', 'max_validfrom_1069D', 'max_lastapprcommoditycat_1041M', 'max_lastcancelreason_561M', 'max_lastrejectcommoditycat_161M', 'max_lastrejectcommodtypec_5251769M', 'max_lastrejectreason_759M', 'max_lastrejectreasonclient_4145040M', 'max_applicationcnt_361L', 'max_applications30d_658L', 'max_applicationscnt_1086L', 'max_applicationscnt_464L', 'max_applicationscnt_629L', 'max_applicationscnt_867L', 'max_bankacctype_710L', 'max_cardtype_51L', 'max_clientscnt12m_3712952L', 'max_clientscnt3m_3712950L', 'max_clientscnt6m_3712949L', 'max_clientscnt_100L', 'max_clientscnt_1022L', 'max_clientscnt_1071L', 'max_clientscnt_1130L', 'max_clientscnt_157L', 'max_clientscnt_257L', 'max_clientscnt_304L', 'max_clientscnt_360L', 'max_clientscnt_493L', 'max_clientscnt_533L', 'max_clientscnt_887L', 'max_clientscnt_946L', 'max_cntincpaycont9m_3716944L', 'max_cntpmts24_3658933L', 'max_commnoinclast6m_3546845L', 'max_credtype_322L', 'max_daysoverduetolerancedd_3976961L', 'max_deferredmnthsnum_166L', 'max_disbursementtype_67L', 'max_eir_270L', 'max_equalitydataagreement_891L', 'max_homephncnt_628L', 'max_inittransactioncode_186L', 'max_interestrate_311L', 'max_isbidproduct_1095L', 'max_isdebitcard_729L', 'max_lastst_736L', 'max_mastercontrelectronic_519L', 'max_mastercontrexist_109L', 'max_mobilephncnt_593L', 'max_monthsannuity_845L', 'max_numactivecreds_622L', 'max_numactivecredschannel_414L', 'max_numactiverelcontr_750L', 'max_numcontrs3months_479L', 'max_numincomingpmts_3546848L', 'max_numinstlallpaidearly3d_817L', 'max_numinstls_657L', 'max_numinstlsallpaid_934L', 'max_numinstlswithdpd10_728L', 'max_numinstlswithdpd5_4187116L', 'max_numinstlswithoutdpd_562L', 'max_numinstmatpaidtearly2d_4499204L', 'max_numinstpaid_4499208L', 'max_numinstpaidearly3d_3546850L', 'max_numinstpaidearly3dest_4493216L', 'max_numinstpaidearly5d_1087L', 'max_numinstpaidearly5dest_4493211L', 'max_numinstpaidearly5dobd_4499205L', 'max_numinstpaidearly_338L', 'max_numinstpaidearlyest_4493214L', 'max_numinstpaidlastcontr_4325080L', 'max_numinstpaidlate1d_3546852L', 'max_numinstregularpaid_973L', 'max_numinstregularpaidest_4493210L', 'max_numinsttopaygr_769L', 'max_numinsttopaygrest_4493213L', 'max_numinstunpaidmax_3546851L', 'max_numinstunpaidmaxest_4493212L', 'max_numnotactivated_1143L', 'max_numpmtchanneldd_318L', 'max_numrejects9m_859L', 'max_opencred_647L', 'max_paytype1st_925L', 'max_paytype_783L', 'max_pctinstlsallpaidearl3d_427L', 'max_pctinstlsallpaidlat10d_839L', 'max_pctinstlsallpaidlate1d_3546856L', 'max_pctinstlsallpaidlate4d_3546849L', 'max_pctinstlsallpaidlate6d_3546844L', 'max_pmtnum_254L', 'max_sellerplacecnt_915L', 'max_sellerplacescnt_216L', 'max_twobodfilling_608L', 'max_typesuite_864L', 'max_dbddpd_boolean', 'max_pays_debt_on_timeP', 'max_actualdpd_943P', 'max_annuity_853A', 'max_credacc_actualbalance_314A', 'max_credacc_credlmt_575A', 'max_credacc_maxhisbal_375A', 'max_credacc_minhisbal_90A', 'max_credamount_590A', 'max_currdebt_94A', 'max_downpmt_134A', 'max_mainoccupationinc_437A', 'max_maxdpdtolerance_577P', 'max_outstandingdebt_522A', 'max_revolvingaccount_394A', 'max_approvaldate_319D', 'max_creationdate_885D', 'max_dateactivated_425D', 'max_dtlastpmt_581D', 'max_dtlastpmtallstes_3545839D', 'max_employedfrom_700D', 'max_firstnonzeroinstldate_307D', 'max_cancelreason_3545846M', 'max_education_1138M', 'max_postype_4733339M', 'max_rejectreason_755M', 'max_rejectreasonclient_4145042M', 'max_byoccupationinc_3656910L', 'max_childnum_21L', 'max_credacc_status_367L', 'max_credacc_transactions_402L', 'max_credtype_587L', 'max_familystate_726L', 'max_inittransactioncode_279L', 'max_isbidproduct_390L', 'max_isdebitcard_527L', 'max_pmtnum_8L', 'max_status_219L', 'max_tenor_203L', 'max_num_group1', 'max_amount_4527230A', 'max_recorddate_4527225D', 'max_num_group1_3', 'max_pmtamount_36A', 'max_processingdate_168D', 'max_num_group1_5', 'max_credlmt_230A', 'max_credlmt_935A', 'max_debtoutstand_525A', 'max_debtoverdue_47A', 'max_dpdmax_139P', 'max_dpdmax_757P', 'max_instlamount_768A', 'max_instlamount_852A', 'max_monthlyinstlamount_332A', 'max_monthlyinstlamount_674A', 'max_outstandingamount_354A', 'max_outstandingamount_362A', 'max_overdueamount_31A', 'max_overdueamount_659A', 'max_overdueamountmax2_14A', 'max_overdueamountmax2_398A', 'max_overdueamountmax_155A', 'max_overdueamountmax_35A', 'max_residualamount_488A', 'max_residualamount_856A', 'max_totalamount_6A', 'max_totalamount_996A', 'max_totaldebtoverduevalue_178A', 'max_totaldebtoverduevalue_718A', 'max_totaloutstanddebtvalue_39A', 'max_totaloutstanddebtvalue_668A', 'max_dateofcredend_289D', 'max_dateofcredend_353D', 'max_dateofcredstart_181D', 'max_dateofcredstart_739D', 'max_dateofrealrepmt_138D', 'max_lastupdate_1112D', 'max_lastupdate_388D', 'max_numberofoverdueinstlmaxdat_148D', 'max_numberofoverdueinstlmaxdat_641D', 'max_overdueamountmax2date_1002D', 'max_overdueamountmax2date_1142D', 'max_refreshdate_3813885D', 'max_classificationofcontr_13M', 'max_classificationofcontr_400M', 'max_contractst_545M', 'max_contractst_964M', 'max_description_351M', 'max_financialinstitution_382M', 'max_financialinstitution_591M', 'max_purposeofcred_426M', 'max_purposeofcred_874M', 'max_subjectrole_182M', 'max_subjectrole_93M', 'max_annualeffectiverate_199L', 'max_annualeffectiverate_63L', 'max_contractsum_5085717L', 'max_dpdmaxdatemonth_442T', 'max_dpdmaxdatemonth_89T', 'max_dpdmaxdateyear_596T', 'max_dpdmaxdateyear_896T', 'max_nominalrate_281L', 'max_nominalrate_498L', 'max_numberofcontrsvalue_258L', 'max_numberofcontrsvalue_358L', 'max_numberofinstls_229L', 'max_numberofinstls_320L', 'max_numberofoutstandinstls_520L', 'max_numberofoutstandinstls_59L', 'max_numberofoverdueinstlmax_1039L', 'max_numberofoverdueinstlmax_1151L', 'max_numberofoverdueinstls_725L', 'max_numberofoverdueinstls_834L', 'max_overdueamountmaxdatemonth_284T', 'max_overdueamountmaxdatemonth_365T', 'max_overdueamountmaxdateyear_2T', 'max_overdueamountmaxdateyear_994T', 'max_periodicityofpmts_1102L', 'max_periodicityofpmts_837L', 'max_prolongationcount_1120L', 'max_num_group1_6', 'max_credit_duration_daysA', 'max_closed_credit_duration_daysA', 'max_time_from_overdue_to_closed_realrepmtA', 'max_time_from_active_overdue_to_realrepmtA', 'max_mainoccupationinc_384A', 'max_birth_259D', 'max_empl_employedfrom_271D', 'max_education_927M', 'max_empladdr_district_926M', 'max_empladdr_zipcode_114M', 'max_language1_981M', 'max_contaddr_matchlist_1032L', 'max_contaddr_smempladdr_334L', 'max_empl_employedtotal_800L', 'max_empl_industry_691L', 'max_familystate_447L', 'max_housetype_905L', 'max_incometype_1044T', 'max_personindex_1023L', 'max_persontype_1072L', 'max_persontype_792L', 'max_relationshiptoclient_415T', 'max_relationshiptoclient_642T', 'max_remitter_829L', 'max_role_1084L', 'max_safeguarantyflag_411L', 'max_sex_738L', 'max_type_25L', 'max_num_group1_9', 'max_amount_416A', 'max_openingdate_313D', 'max_num_group1_10', 'max_openingdate_857D', 'max_num_group1_11']


# 129个 
cat_cols_714 = ['description_5085714M', 'education_1103M', 'education_88M', 'maritalst_385M', 'maritalst_893M', 'requesttype_4525192L', 'bankacctype_710L', 'cardtype_51L', 'credtype_322L', 'disbursementtype_67L', 'inittransactioncode_186L', 'isdebitcard_729L', 'lastapprcommoditycat_1041M', 'lastapprcommoditytypec_5251766M', 'lastcancelreason_561M', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'previouscontdistrict_112M', 'twobodfilling_608L', 'typesuite_864L', 'district_544M', 'education_1138M', 'postype_4733339M', 'profession_152M', 'rejectreason_755M', 'rejectreasonclient_4145042M', 'last_cancelreason_3545846M', 'last_district_544M', 'last_education_1138M', 'last_postype_4733339M', 'last_rejectreason_755M', 'last_rejectreasonclient_4145042M', 'credacc_status_367L', 'credtype_587L', 'familystate_726L', 'inittransactioncode_279L', 'isbidproduct_390L', 'isdebitcard_527L', 'status_219L', 'last_credtype_587L', 'last_familystate_726L', 'last_inittransactioncode_279L', 'last_isbidproduct_390L', 'last_status_219L', 'classificationofcontr_13M', 'classificationofcontr_400M', 'contractst_545M', 'contractst_964M', 'description_351M', 'financialinstitution_382M', 'financialinstitution_591M', 'purposeofcred_426M', 'purposeofcred_874M', 'subjectrole_182M', 'subjectrole_93M', 'last_classificationofcontr_13M', 'last_classificationofcontr_400M', 'last_contractst_545M', 'last_contractst_964M', 'last_description_351M', 'last_financialinstitution_382M', 'last_financialinstitution_591M', 'last_purposeofcred_426M', 'last_purposeofcred_874M', 'last_subjectrole_182M', 'last_subjectrole_93M', 'contaddr_district_15M', 'education_927M', 'empladdr_district_926M', 'empladdr_zipcode_114M', 'language1_981M', 'registaddr_district_1083M', 'last_contaddr_district_15M', 'last_education_927M', 'last_empladdr_district_926M', 'last_empladdr_zipcode_114M', 'last_language1_981M', 'last_registaddr_district_1083M', 'empl_employedtotal_800L', 'empl_industry_691L', 'familystate_447L', 'incometype_1044T', 'relationshiptoclient_415T', 'relationshiptoclient_642T', 'remitter_829L', 'role_1084L', 'safeguarantyflag_411L', 'sex_738L', 'type_25L', 'last_contaddr_matchlist_1032L', 'last_contaddr_smempladdr_334L', 'last_incometype_1044T', 'last_relationshiptoclient_415T', 'last_relationshiptoclient_642T', 'last_remitter_829L', 'last_role_1084L', 'last_safeguarantyflag_411L', 'last_sex_738L', 'last_type_25L', 'collater_typofvalofguarant_298M', 'collater_typofvalofguarant_407M', 'collaterals_typeofguarante_359M', 'collaterals_typeofguarante_669M', 'subjectroles_name_541M', 'subjectroles_name_838M', 'last_collater_typofvalofguarant_298M', 'last_collater_typofvalofguarant_407M', 'last_collaterals_typeofguarante_359M', 'last_collaterals_typeofguarante_669M', 'last_subjectroles_name_541M', 'last_subjectroles_name_838M', 'last_cacccardblochreas_147M', 'conts_type_509L', 'credacc_cards_status_52L', 'last_conts_type_509L', 'addres_district_368M', 'conts_role_79M', 'empls_economicalst_849M', 'empls_employer_name_740M', 'last_addres_district_368M', 'last_conts_role_79M', 'last_empls_economicalst_849M', 'last_empls_employer_name_740M']
# 714个
df_train_714 = ['WEEK_NUM', 'month_decision', 'weekday_decision', 'birthdate_574D', 'contractssum_5085716L', 'dateofbirth_337D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'description_5085714M', 'education_1103M', 'education_88M', 'firstquarter_103L', 'fourthquarter_440L', 'maritalst_385M', 'maritalst_893M', 'numberofqueries_373L', 'pmtscount_423L', 'pmtssum_45A', 'requesttype_4525192L', 'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D', 'secondquarter_766L', 'thirdquarter_1082L', 'actualdpdtolerance_344P', 'amtinstpaidbefduel24m_4187115A', 'annuity_780A', 'annuitynextmonth_57A', 'applicationcnt_361L', 'applications30d_658L', 'applicationscnt_1086L', 'applicationscnt_464L', 'applicationscnt_629L', 'applicationscnt_867L', 'avgdbddpdlast24m_3658932P', 'avgdbddpdlast3m_4187120P', 'avgdbdtollast24m_4525197P', 'avgdpdtolclosure24_3658938P', 'avginstallast24m_3658937A', 'avglnamtstart24m_4525187A', 'avgmaxdpdlast9m_3716943P', 'avgoutstandbalancel6m_4187114A', 'avgpmtlast12m_4525200A', 'bankacctype_710L', 'cardtype_51L', 'clientscnt12m_3712952L', 'clientscnt3m_3712950L', 'clientscnt6m_3712949L', 'clientscnt_100L', 'clientscnt_1022L', 'clientscnt_1071L', 'clientscnt_1130L', 'clientscnt_157L', 'clientscnt_257L', 'clientscnt_304L', 'clientscnt_360L', 'clientscnt_493L', 'clientscnt_533L', 'clientscnt_887L', 'clientscnt_946L', 'cntincpaycont9m_3716944L', 'cntpmts24_3658933L', 'commnoinclast6m_3546845L', 'credamount_770A', 'credtype_322L', 'currdebt_22A', 'currdebtcredtyperange_828A', 'datefirstoffer_1144D', 'datelastunpaid_3546854D', 'daysoverduetolerancedd_3976961L', 'deferredmnthsnum_166L', 'disbursedcredamount_1113A', 'disbursementtype_67L', 'downpmt_116A', 'dtlastpmtallstes_4499206D', 'eir_270L', 'firstclxcampaign_1125D', 'firstdatedue_489D', 'homephncnt_628L', 'inittransactionamount_650A', 'inittransactioncode_186L', 'interestrate_311L', 'isbidproduct_1095L', 'isdebitcard_729L', 'lastactivateddate_801D', 'lastapplicationdate_877D', 'lastapprcommoditycat_1041M', 'lastapprcommoditytypec_5251766M', 'lastapprcredamount_781A', 'lastapprdate_640D', 'lastcancelreason_561M', 'lastdelinqdate_224D', 'lastrejectcommoditycat_161M', 'lastrejectcommodtypec_5251769M', 'lastrejectcredamount_222A', 'lastrejectdate_50D', 'lastrejectreason_759M', 'lastrejectreasonclient_4145040M', 'lastst_736L', 'maininc_215A', 'mastercontrelectronic_519L', 'mastercontrexist_109L', 'maxannuity_159A', 'maxdbddpdlast1m_3658939P', 'maxdbddpdtollast12m_3658940P', 'maxdbddpdtollast6m_4187119P', 'maxdebt4_972A', 'maxdpdfrom6mto36m_3546853P', 'maxdpdinstldate_3546855D', 'maxdpdinstlnum_3546846P', 'maxdpdlast12m_727P', 'maxdpdlast24m_143P', 'maxdpdlast3m_392P', 'maxdpdlast6m_474P', 'maxdpdlast9m_1059P', 'maxdpdtolerance_374P', 'maxinstallast24m_3658928A', 'maxlnamtstart6m_4525199A', 'maxoutstandbalancel12m_4187113A', 'maxpmtlast3m_4525190A', 'mindbddpdlast24m_3658935P', 'mindbdtollast24m_4525191P', 'mobilephncnt_593L', 'monthsannuity_845L', 'numactivecreds_622L', 'numactivecredschannel_414L', 'numactiverelcontr_750L', 'numcontrs3months_479L', 'numincomingpmts_3546848L', 'numinstlallpaidearly3d_817L', 'numinstls_657L', 'numinstlsallpaid_934L', 'numinstlswithdpd10_728L', 'numinstlswithdpd5_4187116L', 'numinstlswithoutdpd_562L', 'numinstmatpaidtearly2d_4499204L', 'numinstpaid_4499208L', 'numinstpaidearly3d_3546850L', 'numinstpaidearly3dest_4493216L', 'numinstpaidearly5d_1087L', 'numinstpaidearly5dest_4493211L', 'numinstpaidearly5dobd_4499205L', 'numinstpaidearly_338L', 'numinstpaidearlyest_4493214L', 'numinstpaidlastcontr_4325080L', 'numinstpaidlate1d_3546852L', 'numinstregularpaid_973L', 'numinstregularpaidest_4493210L', 'numinsttopaygr_769L', 'numinsttopaygrest_4493213L', 'numinstunpaidmax_3546851L', 'numinstunpaidmaxest_4493212L', 'numnotactivated_1143L', 'numpmtchanneldd_318L', 'numrejects9m_859L', 'opencred_647L', 'paytype1st_925L', 'paytype_783L', 'pctinstlsallpaidearl3d_427L', 'pctinstlsallpaidlat10d_839L', 'pctinstlsallpaidlate1d_3546856L', 'pctinstlsallpaidlate4d_3546849L', 'pctinstlsallpaidlate6d_3546844L', 'pmtnum_254L', 'posfpd10lastmonth_333P', 'posfpd30lastmonth_3976960P', 'posfstqpd30lastmonth_3976962P', 'previouscontdistrict_112M', 'price_1097A', 'sellerplacecnt_915L', 'sellerplacescnt_216L', 'sumoutstandtotal_3546847A', 'sumoutstandtotalest_4493215A', 'totaldebt_9A', 'totalsettled_863A', 'totinstallast1m_4525188A', 'twobodfilling_608L', 'typesuite_864L', 'actualdpd_943P', 'annuity_853A', 'credacc_actualbalance_314A', 'credacc_credlmt_575A', 'credacc_maxhisbal_375A', 'credacc_minhisbal_90A', 'credamount_590A', 'currdebt_94A', 'downpmt_134A', 'mainoccupationinc_437A', 'maxdpdtolerance_577P', 'outstandingdebt_522A', 'revolvingaccount_394A', 'last_actualdpd_943P', 'last_annuity_853A', 'last_credacc_credlmt_575A', 'last_credamount_590A', 'last_currdebt_94A', 'last_downpmt_134A', 'last_mainoccupationinc_437A', 'last_maxdpdtolerance_577P', 'last_outstandingdebt_522A', 'mean_actualdpd_943P', 'mean_annuity_853A', 'mean_credacc_actualbalance_314A', 'mean_credacc_credlmt_575A', 'mean_credacc_maxhisbal_375A', 'mean_credacc_minhisbal_90A', 'mean_credamount_590A', 'mean_currdebt_94A', 'mean_downpmt_134A', 'mean_mainoccupationinc_437A', 'mean_maxdpdtolerance_577P', 'mean_outstandingdebt_522A', 'mean_revolvingaccount_394A', 'var_actualdpd_943P', 'var_annuity_853A', 'var_credacc_credlmt_575A', 'var_credamount_590A', 'var_currdebt_94A', 'var_downpmt_134A', 'var_mainoccupationinc_437A', 'var_maxdpdtolerance_577P', 'var_outstandingdebt_522A', 'count_actualdpd_943P', 'count_annuity_853A', 'count_credacc_actualbalance_314A', 'count_credacc_credlmt_575A', 'count_credacc_maxhisbal_375A', 'count_credacc_minhisbal_90A', 'count_credamount_590A', 'count_currdebt_94A', 'count_downpmt_134A', 'count_mainoccupationinc_437A', 'count_maxdpdtolerance_577P', 'count_outstandingdebt_522A', 'count_revolvingaccount_394A', 'median_actualdpd_943P', 'median_annuity_853A', 'median_credacc_actualbalance_314A', 'median_credacc_credlmt_575A', 'median_credacc_maxhisbal_375A', 'median_credacc_minhisbal_90A', 'median_credamount_590A', 'median_currdebt_94A', 'median_downpmt_134A', 'median_mainoccupationinc_437A', 'median_maxdpdtolerance_577P', 'median_outstandingdebt_522A', 'median_revolvingaccount_394A', 'approvaldate_319D', 'creationdate_885D', 'dateactivated_425D', 'dtlastpmt_581D', 'dtlastpmtallstes_3545839D', 'employedfrom_700D', 'firstnonzeroinstldate_307D', 'last_approvaldate_319D', 'last_creationdate_885D', 'last_dateactivated_425D', 'last_dtlastpmt_581D', 'last_dtlastpmtallstes_3545839D', 'last_employedfrom_700D', 'last_firstnonzeroinstldate_307D', 'mean_approvaldate_319D', 'mean_creationdate_885D', 'mean_dateactivated_425D', 'mean_dtlastpmt_581D', 'mean_dtlastpmtallstes_3545839D', 'mean_employedfrom_700D', 'mean_firstnonzeroinstldate_307D', 'district_544M', 'education_1138M', 'postype_4733339M', 'profession_152M', 'rejectreason_755M', 'rejectreasonclient_4145042M', 'last_cancelreason_3545846M', 'last_district_544M', 'last_education_1138M', 'last_postype_4733339M', 'last_rejectreason_755M', 'last_rejectreasonclient_4145042M', 'byoccupationinc_3656910L', 'childnum_21L', 'credacc_status_367L', 'credacc_transactions_402L', 'credtype_587L', 'familystate_726L', 'inittransactioncode_279L', 'isbidproduct_390L', 'isdebitcard_527L', 'pmtnum_8L', 'status_219L', 'tenor_203L', 'last_byoccupationinc_3656910L', 'last_childnum_21L', 'last_credtype_587L', 'last_familystate_726L', 'last_inittransactioncode_279L', 'last_isbidproduct_390L', 'last_pmtnum_8L', 'last_status_219L', 'last_tenor_203L', 'last_num_group1', 'count_num_group1', 'amount_4527230A', 'last_amount_4527230A', 'mean_amount_4527230A', 'var_amount_4527230A', 'count_amount_4527230A', 'median_amount_4527230A', 'recorddate_4527225D', 'last_recorddate_4527225D', 'mean_recorddate_4527225D', 'last_num_group1_3', 'count_num_group1_3', 'pmtamount_36A', 'last_pmtamount_36A', 'mean_pmtamount_36A', 'var_pmtamount_36A', 'count_pmtamount_36A', 'median_pmtamount_36A', 'processingdate_168D', 'last_processingdate_168D', 'mean_processingdate_168D', 'last_num_group1_5', 'count_num_group1_5', 'credlmt_230A', 'credlmt_935A', 'debtoutstand_525A', 'debtoverdue_47A', 'dpdmax_139P', 'dpdmax_757P', 'instlamount_768A', 'instlamount_852A', 'monthlyinstlamount_332A', 'monthlyinstlamount_674A', 'outstandingamount_354A', 'outstandingamount_362A', 'overdueamount_31A', 'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2_398A', 'overdueamountmax_155A', 'overdueamountmax_35A', 'residualamount_488A', 'residualamount_856A', 'totalamount_6A', 'totalamount_996A', 'totaldebtoverduevalue_178A', 'totaldebtoverduevalue_718A', 'totaloutstanddebtvalue_39A', 'totaloutstanddebtvalue_668A', 'mean_credlmt_230A', 'mean_credlmt_935A', 'mean_debtoutstand_525A', 'mean_debtoverdue_47A', 'mean_dpdmax_139P', 'mean_dpdmax_757P', 'mean_instlamount_768A', 'mean_instlamount_852A', 'mean_monthlyinstlamount_332A', 'mean_monthlyinstlamount_674A', 'mean_outstandingamount_354A', 'mean_outstandingamount_362A', 'mean_overdueamount_31A', 'mean_overdueamount_659A', 'mean_overdueamountmax2_14A', 'mean_overdueamountmax2_398A', 'mean_overdueamountmax_155A', 'mean_overdueamountmax_35A', 'mean_residualamount_488A', 'mean_residualamount_856A', 'mean_totalamount_6A', 'mean_totalamount_996A', 'mean_totaldebtoverduevalue_178A', 'mean_totaldebtoverduevalue_718A', 'mean_totaloutstanddebtvalue_39A', 'mean_totaloutstanddebtvalue_668A', 'var_credlmt_230A', 'var_credlmt_935A', 'var_dpdmax_139P', 'var_dpdmax_757P', 'var_instlamount_768A', 'var_monthlyinstlamount_332A', 'var_monthlyinstlamount_674A', 'var_outstandingamount_354A', 'var_outstandingamount_362A', 'var_overdueamount_31A', 'var_overdueamount_659A', 'var_overdueamountmax2_14A', 'var_overdueamountmax2_398A', 'var_overdueamountmax_155A', 'var_overdueamountmax_35A', 'var_residualamount_488A', 'var_residualamount_856A', 'var_totalamount_6A', 'var_totalamount_996A', 'count_credlmt_230A', 'count_credlmt_935A', 'count_debtoutstand_525A', 'count_debtoverdue_47A', 'count_dpdmax_139P', 'count_dpdmax_757P', 'count_instlamount_768A', 'count_instlamount_852A', 'count_monthlyinstlamount_332A', 'count_monthlyinstlamount_674A', 'count_outstandingamount_354A', 'count_outstandingamount_362A', 'count_overdueamount_31A', 'count_overdueamount_659A', 'count_overdueamountmax2_14A', 'count_overdueamountmax2_398A', 'count_overdueamountmax_155A', 'count_overdueamountmax_35A', 'count_residualamount_488A', 'count_residualamount_856A', 'count_totalamount_6A', 'count_totalamount_996A', 'count_totaldebtoverduevalue_178A', 'count_totaldebtoverduevalue_718A', 'count_totaloutstanddebtvalue_39A', 'count_totaloutstanddebtvalue_668A', 'median_credlmt_230A', 'median_credlmt_935A', 'median_debtoutstand_525A', 'median_debtoverdue_47A', 'median_dpdmax_139P', 'median_dpdmax_757P', 'median_instlamount_768A', 'median_instlamount_852A', 'median_monthlyinstlamount_332A', 'median_monthlyinstlamount_674A', 'median_outstandingamount_354A', 'median_outstandingamount_362A', 'median_overdueamount_31A', 'median_overdueamount_659A', 'median_overdueamountmax2_14A', 'median_overdueamountmax2_398A', 'median_overdueamountmax_155A', 'median_overdueamountmax_35A', 'median_residualamount_488A', 'median_residualamount_856A', 'median_totalamount_6A', 'median_totalamount_996A', 'median_totaldebtoverduevalue_178A', 'median_totaldebtoverduevalue_718A', 'median_totaloutstanddebtvalue_39A', 'median_totaloutstanddebtvalue_668A', 'dateofcredend_289D', 'dateofcredend_353D', 'dateofcredstart_181D', 'dateofcredstart_739D', 'dateofrealrepmt_138D', 'lastupdate_1112D', 'lastupdate_388D', 'numberofoverdueinstlmaxdat_148D', 'numberofoverdueinstlmaxdat_641D', 'overdueamountmax2date_1002D', 'overdueamountmax2date_1142D', 'refreshdate_3813885D', 'last_refreshdate_3813885D', 'mean_dateofcredend_289D', 'mean_dateofcredend_353D', 'mean_dateofcredstart_181D', 'mean_dateofcredstart_739D', 'mean_dateofrealrepmt_138D', 'mean_lastupdate_1112D', 'mean_lastupdate_388D', 'mean_numberofoverdueinstlmaxdat_148D', 'mean_numberofoverdueinstlmaxdat_641D', 'mean_overdueamountmax2date_1002D', 'mean_overdueamountmax2date_1142D', 'mean_refreshdate_3813885D', 'classificationofcontr_13M', 'classificationofcontr_400M', 'contractst_545M', 'contractst_964M', 'description_351M', 'financialinstitution_382M', 'financialinstitution_591M', 'purposeofcred_426M', 'purposeofcred_874M', 'subjectrole_182M', 'subjectrole_93M', 'last_classificationofcontr_13M', 'last_classificationofcontr_400M', 'last_contractst_545M', 'last_contractst_964M', 'last_description_351M', 'last_financialinstitution_382M', 'last_financialinstitution_591M', 'last_purposeofcred_426M', 'last_purposeofcred_874M', 'last_subjectrole_182M', 'last_subjectrole_93M', 'annualeffectiverate_199L', 'annualeffectiverate_63L', 'contractsum_5085717L', 'dpdmaxdatemonth_442T', 'dpdmaxdatemonth_89T', 'dpdmaxdateyear_596T', 'dpdmaxdateyear_896T', 'nominalrate_281L', 'nominalrate_498L', 'numberofcontrsvalue_258L', 'numberofcontrsvalue_358L', 'numberofinstls_229L', 'numberofinstls_320L', 'numberofoutstandinstls_520L', 'numberofoutstandinstls_59L', 'numberofoverdueinstlmax_1039L', 'numberofoverdueinstlmax_1151L', 'numberofoverdueinstls_725L', 'numberofoverdueinstls_834L', 'overdueamountmaxdatemonth_284T', 'overdueamountmaxdatemonth_365T', 'overdueamountmaxdateyear_2T', 'overdueamountmaxdateyear_994T', 'periodicityofpmts_1102L', 'periodicityofpmts_837L', 'last_num_group1_6', 'count_num_group1_6', 'mainoccupationinc_384A', 'last_mainoccupationinc_384A', 'mean_mainoccupationinc_384A', 'count_mainoccupationinc_384A', 'median_mainoccupationinc_384A', 'birth_259D', 'empl_employedfrom_271D', 'last_birth_259D', 'mean_birth_259D', 'mean_empl_employedfrom_271D', 'contaddr_district_15M', 'education_927M', 'empladdr_district_926M', 'empladdr_zipcode_114M', 'language1_981M', 'registaddr_district_1083M', 'last_contaddr_district_15M', 'last_education_927M', 'last_empladdr_district_926M', 'last_empladdr_zipcode_114M', 'last_language1_981M', 'last_registaddr_district_1083M', 'empl_employedtotal_800L', 'empl_industry_691L', 'familystate_447L', 'incometype_1044T', 'personindex_1023L', 'persontype_1072L', 'persontype_792L', 'relationshiptoclient_415T', 'relationshiptoclient_642T', 'remitter_829L', 'role_1084L', 'safeguarantyflag_411L', 'sex_738L', 'type_25L', 'last_contaddr_matchlist_1032L', 'last_contaddr_smempladdr_334L', 'last_incometype_1044T', 'last_personindex_1023L', 'last_persontype_1072L', 'last_persontype_792L', 'last_relationshiptoclient_415T', 'last_relationshiptoclient_642T', 'last_remitter_829L', 'last_role_1084L', 'last_safeguarantyflag_411L', 'last_sex_738L', 'last_type_25L', 'last_num_group1_9', 'count_num_group1_9', 'pmts_dpd_1073P', 'pmts_dpd_303P', 'pmts_overdue_1140A', 'pmts_overdue_1152A', 'mean_pmts_dpd_1073P', 'mean_pmts_dpd_303P', 'mean_pmts_overdue_1140A', 'mean_pmts_overdue_1152A', 'var_pmts_dpd_1073P', 'var_pmts_dpd_303P', 'var_pmts_overdue_1140A', 'var_pmts_overdue_1152A', 'count_pmts_dpd_1073P', 'count_pmts_dpd_303P', 'count_pmts_overdue_1140A', 'count_pmts_overdue_1152A', 'median_pmts_dpd_1073P', 'median_pmts_dpd_303P', 'median_pmts_overdue_1140A', 'median_pmts_overdue_1152A', 'collater_typofvalofguarant_298M', 'collater_typofvalofguarant_407M', 'collaterals_typeofguarante_359M', 'collaterals_typeofguarante_669M', 'subjectroles_name_541M', 'subjectroles_name_838M', 'last_collater_typofvalofguarant_298M', 'last_collater_typofvalofguarant_407M', 'last_collaterals_typeofguarante_359M', 'last_collaterals_typeofguarante_669M', 'last_subjectroles_name_541M', 'last_subjectroles_name_838M', 'collater_valueofguarantee_1124L', 'collater_valueofguarantee_876L', 'pmts_month_158T', 'pmts_month_706T', 'pmts_year_1139T', 'pmts_year_507T', 'last_pmts_month_158T', 'last_pmts_month_706T', 'last_pmts_year_1139T', 'last_pmts_year_507T', 'last_num_group1_12', 'last_num_group2', 'count_num_group1_12', 'count_num_group2', 'last_cacccardblochreas_147M', 'conts_type_509L', 'credacc_cards_status_52L', 'last_conts_type_509L', 'last_num_group1_14', 'last_num_group2_14', 'count_num_group1_14', 'count_num_group2_14', 'addres_district_368M', 'conts_role_79M', 'empls_economicalst_849M', 'empls_employer_name_740M', 'last_addres_district_368M', 'last_conts_role_79M', 'last_empls_economicalst_849M', 'last_empls_employer_name_740M', 'last_num_group1_15', 'last_num_group2_15', 'count_num_group1_15', 'count_num_group2_15', 'past_now_annuity', 'days_previous_application', 'previous_income_to_amtin', 'prev_income_child_rate', 'days_creation', 'days_creation_minus_tax_deduction_date', 'MONTHLY_ANNUITY_MINUS_TAX_DEDUC_AMT', 'DAYS_DEDUC_DATE', 'PMTAMOUNT_TO_BYOCCUPINC', 'PERSON_BIRTHDAY', 'BIRTHDAY_VS_AMT_CREDIT', 'DAYS_CREDIT_VS_DAYS_BIRTHDAY', 'START_EMPLOYMENT', 'NEW_DAYS_EMPLOYED_PERC', 'NEW_OVER_EXPECT_CREDIT', 'NEW_OVER_EXPECT_CREDIT_TAXAMOUNT', 'TAXAM_CREDLIM', 'PREV_CREDLIM_CURRENT_CREDLIM', 'DEBIT_COMIN_VS_CREDLIM', 'EMPL_VS_CREDENTDATE', 'CREDENTDATE_MINUS_BIRTHDAY', 'CREDENTDATE_VS_BIRTHDAY', 'PREVCREDENTDATE', 'PREVCREDENTDATE_CLOSEDCONTR', 'PREVCREDENTDATE_OPENCONTR', 'PREVCREDENTDATE_CLOSEDCONTR_REAL', 'CREDIT_OVERDUE', 'OUTSTANDING_DEBIT', 'OUTSTANDING_DEBIT_VS_BYOCCUPINC', 'OUTSTANDING_DEBIT_VS_BYOCCUPINC_CLASSIF', 'DEBIT_TO_NUM_OF_CHILD', 'DEBIT_OVERDUE', 'DEBIT_OVERDUE_VS_BYOCCUPINC', 'ANNUITY_VS_INSTLAMO', 'ANNUITY_VS_DEBT_OVERSTAND', 'EMP_BIGGER_CREDITDATE', 'OVERDUE_AMT_DATE', 'DEBIT_OUT_VS_OVERDUE_CNT', 'PERIODPMT_VS_DBTOUTSTAND', 'PERIODPMT_VS_PMTAM', 'TOT_DEBT_TO_CHILD_NUM', 'TOT_DEBT_TO_DEPOSIT_INC', 'TOT_DEBT_TO_DEBT_OUT', 'TOT_DEBT_TO_ANNUITY', 'TOT_DEBT_TO_DEBTOVERDUE', 'TOT_DEBT_TO_BYOCCUPINC', 'EMP_VS_ACRIVE_CONT_DATE', 'TAXAM_CREDLIM_ACTIVE', 'PREV_CREDLIM_CURRENT_CREDLIM_ACTIVE', 'CREDIT_LOAN_ACT_VS_CHILD_NUM', 'CREDIT_LOAN_ACT_VS_TOTAL_DEBT', 'DEBT_PAST_DUE_VS_TOTAL_DEBT', 'payment_discipline_index', 'annuity_amtin', 'annuity_next_month_ratio', 'annuity_year', 'annuity_week', 'application_week', 'DAYS_BIRTH', 'CLIENTS_DAYS_BIRTH', 'AVG_BUREAU_CONTRACTS_DAY', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10']
non_cat_cols_714 = ['WEEK_NUM', 'month_decision', 'weekday_decision', 'birthdate_574D', 'contractssum_5085716L', 'dateofbirth_337D', 'days120_123L', 'days180_256L', 'days30_165L', 'days360_512L', 'days90_310L', 'firstquarter_103L', 'fourthquarter_440L', 'numberofqueries_373L', 'pmtscount_423L', 'pmtssum_45A', 'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D', 'secondquarter_766L', 'thirdquarter_1082L', 'actualdpdtolerance_344P', 'amtinstpaidbefduel24m_4187115A', 'annuity_780A', 'annuitynextmonth_57A', 'applicationcnt_361L', 'applications30d_658L', 'applicationscnt_1086L', 'applicationscnt_464L', 'applicationscnt_629L', 'applicationscnt_867L', 'avgdbddpdlast24m_3658932P', 'avgdbddpdlast3m_4187120P', 'avgdbdtollast24m_4525197P', 'avgdpdtolclosure24_3658938P', 'avginstallast24m_3658937A', 'avglnamtstart24m_4525187A', 'avgmaxdpdlast9m_3716943P', 'avgoutstandbalancel6m_4187114A', 'avgpmtlast12m_4525200A', 'clientscnt12m_3712952L', 'clientscnt3m_3712950L', 'clientscnt6m_3712949L', 'clientscnt_100L', 'clientscnt_1022L', 'clientscnt_1071L', 'clientscnt_1130L', 'clientscnt_157L', 'clientscnt_257L', 'clientscnt_304L', 'clientscnt_360L', 'clientscnt_493L', 'clientscnt_533L', 'clientscnt_887L', 'clientscnt_946L', 'cntincpaycont9m_3716944L', 'cntpmts24_3658933L', 'commnoinclast6m_3546845L', 'credamount_770A', 'currdebt_22A', 'currdebtcredtyperange_828A', 'datefirstoffer_1144D', 'datelastunpaid_3546854D', 'daysoverduetolerancedd_3976961L', 'deferredmnthsnum_166L', 'disbursedcredamount_1113A', 'downpmt_116A', 'dtlastpmtallstes_4499206D', 'eir_270L', 'firstclxcampaign_1125D', 'firstdatedue_489D', 'homephncnt_628L', 'inittransactionamount_650A', 'interestrate_311L', 'isbidproduct_1095L', 'lastactivateddate_801D', 'lastapplicationdate_877D', 'lastapprcredamount_781A', 'lastapprdate_640D', 'lastdelinqdate_224D', 'lastrejectcredamount_222A', 'lastrejectdate_50D', 'maininc_215A', 'mastercontrelectronic_519L', 'mastercontrexist_109L', 'maxannuity_159A', 'maxdbddpdlast1m_3658939P', 'maxdbddpdtollast12m_3658940P', 'maxdbddpdtollast6m_4187119P', 'maxdebt4_972A', 'maxdpdfrom6mto36m_3546853P', 'maxdpdinstldate_3546855D', 'maxdpdinstlnum_3546846P', 'maxdpdlast12m_727P', 'maxdpdlast24m_143P', 'maxdpdlast3m_392P', 'maxdpdlast6m_474P', 'maxdpdlast9m_1059P', 'maxdpdtolerance_374P', 'maxinstallast24m_3658928A', 'maxlnamtstart6m_4525199A', 'maxoutstandbalancel12m_4187113A', 'maxpmtlast3m_4525190A', 'mindbddpdlast24m_3658935P', 'mindbdtollast24m_4525191P', 'mobilephncnt_593L', 'monthsannuity_845L', 'numactivecreds_622L', 'numactivecredschannel_414L', 'numactiverelcontr_750L', 'numcontrs3months_479L', 'numincomingpmts_3546848L', 'numinstlallpaidearly3d_817L', 'numinstls_657L', 'numinstlsallpaid_934L', 'numinstlswithdpd10_728L', 'numinstlswithdpd5_4187116L', 'numinstlswithoutdpd_562L', 'numinstmatpaidtearly2d_4499204L', 'numinstpaid_4499208L', 'numinstpaidearly3d_3546850L', 'numinstpaidearly3dest_4493216L', 'numinstpaidearly5d_1087L', 'numinstpaidearly5dest_4493211L', 'numinstpaidearly5dobd_4499205L', 'numinstpaidearly_338L', 'numinstpaidearlyest_4493214L', 'numinstpaidlastcontr_4325080L', 'numinstpaidlate1d_3546852L', 'numinstregularpaid_973L', 'numinstregularpaidest_4493210L', 'numinsttopaygr_769L', 'numinsttopaygrest_4493213L', 'numinstunpaidmax_3546851L', 'numinstunpaidmaxest_4493212L', 'numnotactivated_1143L', 'numpmtchanneldd_318L', 'numrejects9m_859L', 'pctinstlsallpaidearl3d_427L', 'pctinstlsallpaidlat10d_839L', 'pctinstlsallpaidlate1d_3546856L', 'pctinstlsallpaidlate4d_3546849L', 'pctinstlsallpaidlate6d_3546844L', 'pmtnum_254L', 'posfpd10lastmonth_333P', 'posfpd30lastmonth_3976960P', 'posfstqpd30lastmonth_3976962P', 'price_1097A', 'sellerplacecnt_915L', 'sellerplacescnt_216L', 'sumoutstandtotal_3546847A', 'sumoutstandtotalest_4493215A', 'totaldebt_9A', 'totalsettled_863A', 'totinstallast1m_4525188A', 'actualdpd_943P', 'annuity_853A', 'credacc_actualbalance_314A', 'credacc_credlmt_575A', 'credacc_maxhisbal_375A', 'credacc_minhisbal_90A', 'credamount_590A', 'currdebt_94A', 'downpmt_134A', 'mainoccupationinc_437A', 'maxdpdtolerance_577P', 'outstandingdebt_522A', 'revolvingaccount_394A', 'last_actualdpd_943P', 'last_annuity_853A', 'last_credacc_credlmt_575A', 'last_credamount_590A', 'last_currdebt_94A', 'last_downpmt_134A', 'last_mainoccupationinc_437A', 'last_maxdpdtolerance_577P', 'last_outstandingdebt_522A', 'mean_actualdpd_943P', 'mean_annuity_853A', 'mean_credacc_actualbalance_314A', 'mean_credacc_credlmt_575A', 'mean_credacc_maxhisbal_375A', 'mean_credacc_minhisbal_90A', 'mean_credamount_590A', 'mean_currdebt_94A', 'mean_downpmt_134A', 'mean_mainoccupationinc_437A', 'mean_maxdpdtolerance_577P', 'mean_outstandingdebt_522A', 'mean_revolvingaccount_394A', 'var_actualdpd_943P', 'var_annuity_853A', 'var_credacc_credlmt_575A', 'var_credamount_590A', 'var_currdebt_94A', 'var_downpmt_134A', 'var_mainoccupationinc_437A', 'var_maxdpdtolerance_577P', 'var_outstandingdebt_522A', 'count_actualdpd_943P', 'count_annuity_853A', 'count_credacc_actualbalance_314A', 'count_credacc_credlmt_575A', 'count_credacc_maxhisbal_375A', 'count_credacc_minhisbal_90A', 'count_credamount_590A', 'count_currdebt_94A', 'count_downpmt_134A', 'count_mainoccupationinc_437A', 'count_maxdpdtolerance_577P', 'count_outstandingdebt_522A', 'count_revolvingaccount_394A', 'median_actualdpd_943P', 'median_annuity_853A', 'median_credacc_actualbalance_314A', 'median_credacc_credlmt_575A', 'median_credacc_maxhisbal_375A', 'median_credacc_minhisbal_90A', 'median_credamount_590A', 'median_currdebt_94A', 'median_downpmt_134A', 'median_mainoccupationinc_437A', 'median_maxdpdtolerance_577P', 'median_outstandingdebt_522A', 'median_revolvingaccount_394A', 'approvaldate_319D', 'creationdate_885D', 'dateactivated_425D', 'dtlastpmt_581D', 'dtlastpmtallstes_3545839D', 'employedfrom_700D', 'firstnonzeroinstldate_307D', 'last_approvaldate_319D', 'last_creationdate_885D', 'last_dateactivated_425D', 'last_dtlastpmt_581D', 'last_dtlastpmtallstes_3545839D', 'last_employedfrom_700D', 'last_firstnonzeroinstldate_307D', 'mean_approvaldate_319D', 'mean_creationdate_885D', 'mean_dateactivated_425D', 'mean_dtlastpmt_581D', 'mean_dtlastpmtallstes_3545839D', 'mean_employedfrom_700D', 'mean_firstnonzeroinstldate_307D', 'byoccupationinc_3656910L', 'childnum_21L', 'credacc_transactions_402L', 'pmtnum_8L', 'tenor_203L', 'last_byoccupationinc_3656910L', 'last_childnum_21L', 'last_pmtnum_8L', 'last_tenor_203L', 'last_num_group1', 'count_num_group1', 'amount_4527230A', 'last_amount_4527230A', 'mean_amount_4527230A', 'var_amount_4527230A', 'count_amount_4527230A', 'median_amount_4527230A', 'recorddate_4527225D', 'last_recorddate_4527225D', 'mean_recorddate_4527225D', 'last_num_group1_3', 'count_num_group1_3', 'pmtamount_36A', 'last_pmtamount_36A', 'mean_pmtamount_36A', 'var_pmtamount_36A', 'count_pmtamount_36A', 'median_pmtamount_36A', 'processingdate_168D', 'last_processingdate_168D', 'mean_processingdate_168D', 'last_num_group1_5', 'count_num_group1_5', 'credlmt_230A', 'credlmt_935A', 'debtoutstand_525A', 'debtoverdue_47A', 'dpdmax_139P', 'dpdmax_757P', 'instlamount_768A', 'instlamount_852A', 'monthlyinstlamount_332A', 'monthlyinstlamount_674A', 'outstandingamount_354A', 'outstandingamount_362A', 'overdueamount_31A', 'overdueamount_659A', 'overdueamountmax2_14A', 'overdueamountmax2_398A', 'overdueamountmax_155A', 'overdueamountmax_35A', 'residualamount_488A', 'residualamount_856A', 'totalamount_6A', 'totalamount_996A', 'totaldebtoverduevalue_178A', 'totaldebtoverduevalue_718A', 'totaloutstanddebtvalue_39A', 'totaloutstanddebtvalue_668A', 'mean_credlmt_230A', 'mean_credlmt_935A', 'mean_debtoutstand_525A', 'mean_debtoverdue_47A', 'mean_dpdmax_139P', 'mean_dpdmax_757P', 'mean_instlamount_768A', 'mean_instlamount_852A', 'mean_monthlyinstlamount_332A', 'mean_monthlyinstlamount_674A', 'mean_outstandingamount_354A', 'mean_outstandingamount_362A', 'mean_overdueamount_31A', 'mean_overdueamount_659A', 'mean_overdueamountmax2_14A', 'mean_overdueamountmax2_398A', 'mean_overdueamountmax_155A', 'mean_overdueamountmax_35A', 'mean_residualamount_488A', 'mean_residualamount_856A', 'mean_totalamount_6A', 'mean_totalamount_996A', 'mean_totaldebtoverduevalue_178A', 'mean_totaldebtoverduevalue_718A', 'mean_totaloutstanddebtvalue_39A', 'mean_totaloutstanddebtvalue_668A', 'var_credlmt_230A', 'var_credlmt_935A', 'var_dpdmax_139P', 'var_dpdmax_757P', 'var_instlamount_768A', 'var_monthlyinstlamount_332A', 'var_monthlyinstlamount_674A', 'var_outstandingamount_354A', 'var_outstandingamount_362A', 'var_overdueamount_31A', 'var_overdueamount_659A', 'var_overdueamountmax2_14A', 'var_overdueamountmax2_398A', 'var_overdueamountmax_155A', 'var_overdueamountmax_35A', 'var_residualamount_488A', 'var_residualamount_856A', 'var_totalamount_6A', 'var_totalamount_996A', 'count_credlmt_230A', 'count_credlmt_935A', 'count_debtoutstand_525A', 'count_debtoverdue_47A', 'count_dpdmax_139P', 'count_dpdmax_757P', 'count_instlamount_768A', 'count_instlamount_852A', 'count_monthlyinstlamount_332A', 'count_monthlyinstlamount_674A', 'count_outstandingamount_354A', 'count_outstandingamount_362A', 'count_overdueamount_31A', 'count_overdueamount_659A', 'count_overdueamountmax2_14A', 'count_overdueamountmax2_398A', 'count_overdueamountmax_155A', 'count_overdueamountmax_35A', 'count_residualamount_488A', 'count_residualamount_856A', 'count_totalamount_6A', 'count_totalamount_996A', 'count_totaldebtoverduevalue_178A', 'count_totaldebtoverduevalue_718A', 'count_totaloutstanddebtvalue_39A', 'count_totaloutstanddebtvalue_668A', 'median_credlmt_230A', 'median_credlmt_935A', 'median_debtoutstand_525A', 'median_debtoverdue_47A', 'median_dpdmax_139P', 'median_dpdmax_757P', 'median_instlamount_768A', 'median_instlamount_852A', 'median_monthlyinstlamount_332A', 'median_monthlyinstlamount_674A', 'median_outstandingamount_354A', 'median_outstandingamount_362A', 'median_overdueamount_31A', 'median_overdueamount_659A', 'median_overdueamountmax2_14A', 'median_overdueamountmax2_398A', 'median_overdueamountmax_155A', 'median_overdueamountmax_35A', 'median_residualamount_488A', 'median_residualamount_856A', 'median_totalamount_6A', 'median_totalamount_996A', 'median_totaldebtoverduevalue_178A', 'median_totaldebtoverduevalue_718A', 'median_totaloutstanddebtvalue_39A', 'median_totaloutstanddebtvalue_668A', 'dateofcredend_289D', 'dateofcredend_353D', 'dateofcredstart_181D', 'dateofcredstart_739D', 'dateofrealrepmt_138D', 'lastupdate_1112D', 'lastupdate_388D', 'numberofoverdueinstlmaxdat_148D', 'numberofoverdueinstlmaxdat_641D', 'overdueamountmax2date_1002D', 'overdueamountmax2date_1142D', 'refreshdate_3813885D', 'last_refreshdate_3813885D', 'mean_dateofcredend_289D', 'mean_dateofcredend_353D', 'mean_dateofcredstart_181D', 'mean_dateofcredstart_739D', 'mean_dateofrealrepmt_138D', 'mean_lastupdate_1112D', 'mean_lastupdate_388D', 'mean_numberofoverdueinstlmaxdat_148D', 'mean_numberofoverdueinstlmaxdat_641D', 'mean_overdueamountmax2date_1002D', 'mean_overdueamountmax2date_1142D', 'mean_refreshdate_3813885D', 'annualeffectiverate_199L', 'annualeffectiverate_63L', 'contractsum_5085717L', 'dpdmaxdatemonth_442T', 'dpdmaxdatemonth_89T', 'dpdmaxdateyear_596T', 'dpdmaxdateyear_896T', 'nominalrate_281L', 'nominalrate_498L', 'numberofcontrsvalue_258L', 'numberofcontrsvalue_358L', 'numberofinstls_229L', 'numberofinstls_320L', 'numberofoutstandinstls_520L', 'numberofoutstandinstls_59L', 'numberofoverdueinstlmax_1039L', 'numberofoverdueinstlmax_1151L', 'numberofoverdueinstls_725L', 'numberofoverdueinstls_834L', 'overdueamountmaxdatemonth_284T', 'overdueamountmaxdatemonth_365T', 'overdueamountmaxdateyear_2T', 'overdueamountmaxdateyear_994T', 'periodicityofpmts_1102L', 'periodicityofpmts_837L', 'last_num_group1_6', 'count_num_group1_6', 'mainoccupationinc_384A', 'last_mainoccupationinc_384A', 'mean_mainoccupationinc_384A', 'count_mainoccupationinc_384A', 'median_mainoccupationinc_384A', 'birth_259D', 'empl_employedfrom_271D', 'last_birth_259D', 'mean_birth_259D', 'mean_empl_employedfrom_271D', 'personindex_1023L', 'persontype_1072L', 'persontype_792L', 'last_personindex_1023L', 'last_persontype_1072L', 'last_persontype_792L', 'last_num_group1_9', 'count_num_group1_9', 'pmts_dpd_1073P', 'pmts_dpd_303P', 'pmts_overdue_1140A', 'pmts_overdue_1152A', 'mean_pmts_dpd_1073P', 'mean_pmts_dpd_303P', 'mean_pmts_overdue_1140A', 'mean_pmts_overdue_1152A', 'var_pmts_dpd_1073P', 'var_pmts_dpd_303P', 'var_pmts_overdue_1140A', 'var_pmts_overdue_1152A', 'count_pmts_dpd_1073P', 'count_pmts_dpd_303P', 'count_pmts_overdue_1140A', 'count_pmts_overdue_1152A', 'median_pmts_dpd_1073P', 'median_pmts_dpd_303P', 'median_pmts_overdue_1140A', 'median_pmts_overdue_1152A', 'collater_valueofguarantee_1124L', 'collater_valueofguarantee_876L', 'pmts_month_158T', 'pmts_month_706T', 'pmts_year_1139T', 'pmts_year_507T', 'last_pmts_month_158T', 'last_pmts_month_706T', 'last_pmts_year_1139T', 'last_pmts_year_507T', 'last_num_group1_12', 'last_num_group2', 'count_num_group1_12', 'count_num_group2', 'last_num_group1_14', 'last_num_group2_14', 'count_num_group1_14', 'count_num_group2_14', 'last_num_group1_15', 'last_num_group2_15', 'count_num_group1_15', 'count_num_group2_15', 'past_now_annuity', 'days_previous_application', 'previous_income_to_amtin', 'prev_income_child_rate', 'days_creation', 'days_creation_minus_tax_deduction_date', 'MONTHLY_ANNUITY_MINUS_TAX_DEDUC_AMT', 'DAYS_DEDUC_DATE', 'PMTAMOUNT_TO_BYOCCUPINC', 'PERSON_BIRTHDAY', 'BIRTHDAY_VS_AMT_CREDIT', 'DAYS_CREDIT_VS_DAYS_BIRTHDAY', 'START_EMPLOYMENT', 'NEW_DAYS_EMPLOYED_PERC', 'NEW_OVER_EXPECT_CREDIT', 'NEW_OVER_EXPECT_CREDIT_TAXAMOUNT', 'TAXAM_CREDLIM', 'PREV_CREDLIM_CURRENT_CREDLIM', 'DEBIT_COMIN_VS_CREDLIM', 'EMPL_VS_CREDENTDATE', 'CREDENTDATE_MINUS_BIRTHDAY', 'CREDENTDATE_VS_BIRTHDAY', 'PREVCREDENTDATE', 'PREVCREDENTDATE_CLOSEDCONTR', 'PREVCREDENTDATE_OPENCONTR', 'PREVCREDENTDATE_CLOSEDCONTR_REAL', 'CREDIT_OVERDUE', 'OUTSTANDING_DEBIT', 'OUTSTANDING_DEBIT_VS_BYOCCUPINC', 'OUTSTANDING_DEBIT_VS_BYOCCUPINC_CLASSIF', 'DEBIT_TO_NUM_OF_CHILD', 'DEBIT_OVERDUE', 'DEBIT_OVERDUE_VS_BYOCCUPINC', 'ANNUITY_VS_INSTLAMO', 'ANNUITY_VS_DEBT_OVERSTAND', 'EMP_BIGGER_CREDITDATE', 'OVERDUE_AMT_DATE', 'DEBIT_OUT_VS_OVERDUE_CNT', 'PERIODPMT_VS_DBTOUTSTAND', 'PERIODPMT_VS_PMTAM', 'TOT_DEBT_TO_CHILD_NUM', 'TOT_DEBT_TO_DEPOSIT_INC', 'TOT_DEBT_TO_DEBT_OUT', 'TOT_DEBT_TO_ANNUITY', 'TOT_DEBT_TO_DEBTOVERDUE', 'TOT_DEBT_TO_BYOCCUPINC', 'EMP_VS_ACRIVE_CONT_DATE', 'TAXAM_CREDLIM_ACTIVE', 'PREV_CREDLIM_CURRENT_CREDLIM_ACTIVE', 'CREDIT_LOAN_ACT_VS_CHILD_NUM', 'CREDIT_LOAN_ACT_VS_TOTAL_DEBT', 'DEBT_PAST_DUE_VS_TOTAL_DEBT', 'payment_discipline_index', 'annuity_amtin', 'annuity_next_month_ratio', 'annuity_year', 'annuity_week', 'application_week', 'DAYS_BIRTH', 'CLIENTS_DAYS_BIRTH', 'AVG_BUREAU_CONTRACTS_DAY', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10']
non_cat_cols_714_type = ['int64', 'int64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'bool', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64', 'float64', 'int64', 'float64', 'float64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64', 'int64', 'int64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64', 'int64', 'float64', 'int64', 'float64', 'int64', 'float64', 'float64', 'float64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64', 'float64', 'int64', 'int64', 'int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64']

# ==================================================
# """ 统计所有数据列类型并保存脚本 """
# from IPython import embed
# embed()

# non_cat_cols_714 = [col for col in df_train_714 if col not in cat_cols_714]
# print(non_cat_cols_714)
# df_train[non_cat_cols_714].dtypes
# non_cat_cols_714_type = [str(dtype) for dtype in df_train[non_cat_cols_714].dtypes.values]
# print(non_cat_cols_714_type)
# ==================================================
# print('len(df_train): ', len(df_train))

# df_train = df_train[df_train_386]
# cat_cols = cat_cols_386

# # from IPython import embed
# # embed()
# """
# 0.1：99551，0.2：22531，0.15：45841，0.17：34396，0.12：72329
# 0.9：18918，0.99：945，0.95：9612，0.92：15492，0.97：5142
# """

# index = pd.read_csv('clean0.9.csv', header=None)[0]
# # index = (y==1) 
# df_train = df_train[~index]
# y = y[~index]
# weeks = weeks[~index]

# print('len(df_train): ', len(df_train))


# ======================================== 特征列分类 =====================================


# ======================================== 二分类模型训练 =====================================
# # df_train = df_train[:20000]
# # weeks = weeks[:20000]

# df_train.loc[0:len(df_train)//2, 'is_test'] = 0
# df_train.loc[len(df_train)//2:len(df_train), 'is_test'] = 1

# y = df_train["is_test"]

# fitted_models_lgb = []
# cv_scores_lgb = []
# fold = 1
# cv = StratifiedGroupKFold(n_splits=10, shuffle=False)
# for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

#     # X_train(≈40000,386), y_train(≈40000)
#     X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
#     X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]    
    

#     # ===============================
#     X_train[cat_cols_386] = X_train[cat_cols_386].astype("category")
#     X_valid[cat_cols_386] = X_valid[cat_cols_386].astype("category")

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
#         'gpu_use_dp' : True, # 转化float为64精度
#     }

#     # 一次训练
#     model = lgb.LGBMClassifier(**params)
#     model.fit(
#         X_train[df_train_386], y_train,
#         eval_set = [(X_valid[df_train_386], y_valid)],
#         callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)],
#         # init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset/lgbm_fold{fold}.txt",
#     )
#     model.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm2_fold{fold}.txt')
#     model2 = model


#     fitted_models_lgb.append(model2)
#     y_pred_valid = model2.predict_proba(X_valid[df_train_386])[:,1]
#     auc_score = roc_auc_score(y_valid, y_pred_valid)
#     print('auc_score: ', auc_score)
#     cv_scores_lgb.append(auc_score)
#     print()
#     print("分隔符")
#     print()
#     # ===========================

#     break
#     fold = fold+1

# print("CV AUC scores: ", cv_scores_lgb)
# print("Mean CV AUC score: ", np.mean(cv_scores_lgb))
# ======================================== 二分类模型训练 =====================================


# ======================================== 训练3树模型 =====================================

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
    
    # if fold<4:
    #     fold += 1
    #     continue

    # X_train(≈40000,386), y_train(≈40000)
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]    
    

    # ===============================
    # X_train[cat_cols] = X_train[cat_cols].astype("category")
    # X_valid[cat_cols] = X_valid[cat_cols].astype("category")

    # # if fold%2 ==1:
    # #     params = {
    # #         "boosting_type": "gbdt",
    # #         "colsample_bynode": 0.8,
    # #         "colsample_bytree": 0.8,
    # #         "device": device,
    # #         "extra_trees": True,
    # #         "learning_rate": 0.05,
    # #         "l1_regularization": 0.1,
    # #         "l2_regularization": 10,
    # #         "max_depth": 20,
    # #         "metric": "auc",
    # #         "n_estimators": 2000,
    # #         "num_leaves": 64,
    # #         "objective": "binary",
    # #         "random_state": 42,
    # #         "verbose": -1,
    # #     }
    # # else:
    # #     params = {
    # #         "boosting_type": "gbdt",
    # #         "colsample_bynode": 0.8,
    # #         "colsample_bytree": 0.8,
    # #         "device": device,
    # #         "extra_trees": True,
    # #         "learning_rate": 0.03,
    # #         "l1_regularization": 0.1,
    # #         "l2_regularization": 10,
    # #         "max_depth": 16,
    # #         "metric": "auc",
    # #         "n_estimators": 2000,
    # #         "num_leaves": 72,
    # #         "objective": "binary",
    # #         "random_state": 42,
    # #         "verbose": -1,
    # #     }



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
    #     # "device": 'gpu', # gpu
    #     "device": 'cpu', # gpu
    #     'gpu_use_dp' : True, # 转化float为64精度


    #     # 'max_bin':275,  

    #     # # 平衡类别之间的权重  损失函数不会因为样本不平衡而被“推向”样本量偏少的类别中
    #     # "sample_weight":'balanced',
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

    # # # 二次优化
    # # params['learning_rate'] = 0.01
    # # model2 = lgb.LGBMClassifier(**params)
    # # model2.fit(
    # #     X_train, y_train,
    # #     eval_set = [(X_valid, y_valid)],
    # #     callbacks = [lgb.log_evaluation(200), lgb.early_stopping(200)],
    # #     init_model = f"/home/xyli/kaggle/kaggle_HomeCredit/dataset8/lgbm_fold{fold}.txt",
    # # )
    # # model2.booster_.save_model(f'/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt')
    

    # fitted_models_lgb.append(model2)
    # y_pred_valid = model2.predict_proba(X_valid)[:,1]
    # auc_score = roc_auc_score(y_valid, y_pred_valid)
    # print('auc_score: ', auc_score)
    # cv_scores_lgb.append(auc_score)
    # print()
    # print("分隔符")
    # print()
    # ===========================


    # ======================================
#     X_train[cat_cols] = X_train[cat_cols].astype("str")
#     X_valid[cat_cols] = X_valid[cat_cols].astype("str")
#     train_pool = Pool(X_train, y_train,cat_features=cat_cols)
#     val_pool = Pool(X_valid, y_valid,cat_features=cat_cols)

#     # clf = CatBoostClassifier( 
#     #     best_model_min_trees = 1200, # 1000
#     #     boosting_type = "Plain",
#     #     eval_metric = "AUC",
#     #     iterations = 6000,
#     #     learning_rate = 0.05,
#     #     l2_leaf_reg = 10,
#     #     max_leaves = 64,
#     #     random_seed = 42,
#     #     task_type = "GPU",
#     #     use_best_model = True
#     # ) 

#     clf = CatBoostClassifier(
#         eval_metric='AUC',
#         task_type='GPU',
#         learning_rate=0.03, # 0.03
#         iterations=6000, # n_est
# #         early_stopping_rounds = 500,
#     )

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
    
#     # =================================

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

with open("log.txt", "a") as f:
    print("CV AUC scores: ", cv_scores_cat, file=f)
    print("Mean CV AUC score: ", np.mean(cv_scores_cat), file=f)

    print("CV AUC scores: ", cv_scores_lgb, file=f)
    print("Mean CV AUC score: ", np.mean(cv_scores_lgb), file=f)

    print("CV AUC scores: ", cv_scores_xgb, file=f)
    print("Mean CV AUC score: ", np.mean(cv_scores_xgb), file=f)

    print("CV AUC scores: ", cv_scores_cat_dw, file=f)
    print("Mean CV AUC score: ", np.mean(cv_scores_cat_dw), file=f)

    print("CV AUC scores: ", cv_scores_cat_lg, file=f)
    print("Mean CV AUC score: ", np.mean(cv_scores_cat_lg), file=f)

    print("CV AUC scores: ", cv_scores_lgb_dart, file=f)
    print("Mean CV AUC score: ", np.mean(cv_scores_lgb_dart), file=f)

    print("CV AUC scores: ", cv_scores_lgb_rf, file=f)
    print("Mean CV AUC score: ", np.mean(cv_scores_lgb_rf), file=f)

# ======================================== 训练3树模型 =====================================







# ======================================== 推理验证 =====================================
from IPython import embed
embed()


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
    clf3.load_model(f"/home/xyli/kaggle/kaggle_HomeCredit/catboost_fold{fold}.cbm")
    fitted_models_cat3.append(clf3) 
    
    model3 = lgb.LGBMClassifier()
    model3 = lgb.Booster(model_file=f"/home/xyli/kaggle/kaggle_HomeCredit/lgbm_fold{fold}.txt")
    fitted_models_lgb3.append(model3)


class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        y_preds = []

        # from IPython import embed
        # embed()

        # X[cat_cols] = X[cat_cols].astype("str")
        # y_preds += [estimator.predict_proba(X[df_train_829])[:, 1] for estimator in self.estimators[0:5]]
        # y_preds += [estimator.predict_proba(X[df_train_386])[:, 1] for estimator in self.estimators[5:10]]
        # y_preds += [estimator.predict_proba(X)[:, 1] for estimator in self.estimators[10:15]]
        
        X[cat_cols] = X[cat_cols].astype("category")
        # y_preds += [estimator.predict(X[df_train_829]) for estimator in self.estimators[15:20]]
        # y_preds += [estimator.predict(X[df_train_386]) for estimator in self.estimators[20:25]]
        y_preds += [estimator.predict(X[df_train_714]) for estimator in self.estimators[25:30]]
         
        return np.mean(y_preds, axis=0)
    
    def predict_proba_scan(self, X):
        y_preds = []
        # from IPython import embed
        # embed()

        X[cat_cols] = X[cat_cols].astype("str")
        y_preds += [estimator.predict_proba(X)[:, 1] for estimator in self.estimators[10:15]]
        
        X[cat_cols] = X[cat_cols].astype("category")
        y_preds += [estimator.predict(X) for estimator in self.estimators[25:30]] 
       

        # X[cat_cols_470] = X[cat_cols_470].astype("category")
        # y_preds += [estimator.predict(X[df_train_470]) for estimator in self.estimators[25:30]]

        return np.mean(y_preds, axis=0)


model = VotingModel(
    fitted_models_cat1 + 
    fitted_models_cat2 +
    fitted_models_cat3 + 
    fitted_models_lgb1 + 
    fitted_models_lgb2 +
    fitted_models_lgb3
)



# 5min
print('开始计算cv')
valid_score = []
# valid_preds = model.predict_proba_scan(df_train) # df_train消掉了额外的2个特征列
# valid_score += [roc_auc_score(y, valid_preds)]
# print(valid_score)
valid_preds = model.predict_proba(df_train)
valid_score += [roc_auc_score(y, valid_preds)]
print(valid_score)


# valid_score += [(valid_score[0]+valid_score[1])/2.0]
# print(valid_score)

# ================= cleanning =======================
# df_train['predict'] = valid_preds
# df_train["target"] = y


# from IPython import embed
# embed()

# df_train['absolute_difference'] = abs(df_train['predict'] - df_train['target'])

# index = df_train['absolute_difference'] >= 0.9
# count_greater_than_05 = df_train[index].shape[0]
# print(count_greater_than_05)
# df_train[index]

# # 0.9~0.99之间, [0.9, 0.99, 0.95, 0.92, 0.97]
# # 0.9：18918，0.99：945，0.95：9612，0.92：15492，0.97：5142
# threshold = 0.97
# print(len(df_train[(df_train['absolute_difference'] >= threshold) & (df_train['target']==1)]))
# index = (df_train['absolute_difference'] >= threshold) & (df_train['target']==1)
# index.to_csv(f'clean{threshold}.csv', index=False, header=False)

# # index = pd.read_csv(f'clean{threshold}.csv', header=None)[0]
# # print(len(df_train[index]))

# # 0.1~0.2之间, [0.1, 0.2, 0.15, 0.17, 0.12]
# # 0.1：99551，0.2：22531，0.15：45841，0.17：34396，0.12：72329
# threshold = 0.12
# print(len(df_train[(df_train['absolute_difference'] >= threshold) & (df_train['target']==0)]))
# index = (df_train['absolute_difference'] >= threshold) & (df_train['target']==0)
# index.to_csv(f'clean{threshold}.csv', index=False, header=False)

# ================= cleanning =======================



# ======================================== 推理验证 =====================================


