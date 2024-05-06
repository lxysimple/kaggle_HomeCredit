
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
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train832.csv')
df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train832.csv', nrows=50000)
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train419.csv')
# df_train = pd.read_csv('/home/xyli/kaggle/kaggle_HomeCredit/train389.csv')

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



# 找到除cat_cols列外的所有列
non_cat_cols = df_train.columns.difference(cat_cols) 

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
# ======================================== 特征列分类 =====================================


# ======================================== 清理数据 =====================================
"""
对cat_cols列外的所有列进行数据清理，即把nan和inf换成该列的均值
"""


cat_cols = cat_cols_386
non_cat_cols = non_cat_cols_386

df_train[cat_cols] = df_train[cat_cols].astype(str)

# print('df_train.shape: ', df_train.shape)
# print('df_train[cat_cols].shape: ', df_train[cat_cols].shape)
# print('df_train[non_cat_cols].shape: ', df_train[non_cat_cols].shape)
# # 求1列均值时，遇到nan/inf会自动忽略
# mean_values = df_train[non_cat_cols].mean()# 找到所有列的均值
# # 如果该列都是nan/inf，均值为inf，则令均值为0
# mean_values = mean_values.replace([np.inf, -np.inf, np.nan], 0)

# for column in non_cat_cols:   
#     # # 将nan换成该列的均值，或者0
#     # df_train[column] = df_train[column].fillna(mean_values[column])
#     # df_train[column].replace([np.inf,-np.inf], mean_values[column], inplace=True)

#     # 将nan换成0
#     df_train[column] = df_train[column].fillna(0)
#     # 将+-无穷值替换为0
#     df_train[column].replace([np.inf,-np.inf], 0, inplace=True)
    
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



# 因为最后一行全是nan，只是为了让编码器学习nan，所以现在就可以去掉了
y = y[:-1]
df_train = df_train[:-1]
weeks = weeks[:-1]
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

all_feat_cols = [i for i in range(662)] # 386 386-113(cat_cols)=273 829 662
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

# from torch.utils.data import DataLoader
# import torch
# import time
# import torch.nn.functional as F



# fold = 1
# for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

#     # if fold <=3:
#     #     fold = fold + 1
#     #     continue

#     # from IPython import embed
#     # embed()

#     # X_train(≈40000,386), y_train(≈40000)
#     X_train, y_train = df_train[non_cat_cols].iloc[idx_train].values, y.iloc[idx_train].values 
#     X_valid, y_valid = df_train[non_cat_cols].iloc[idx_valid].values, y.iloc[idx_valid].values

#     # X_train, y_train = df_train.iloc[idx_train].values, y.iloc[idx_train].values 
#     # X_valid, y_valid = df_train.iloc[idx_valid].values, y.iloc[idx_valid].values

    
#     # 定义dataset与dataloader
#     train_set = MarketDataset(X_train, y_train)
#     # batch_size=15000
#     train_loader = DataLoader(train_set, batch_size=15000, shuffle=True, num_workers=7)
#     valid_set = MarketDataset(X_valid, y_valid)
#     valid_loader = DataLoader(valid_set, batch_size=15000, shuffle=False, num_workers=7)

#     # print(valid_set[0])

    
#     print(f'Fold{fold}:') 
#     torch.cuda.empty_cache()
#     device = torch.device("cuda")

#     model = Model2()
    
#     try:
#         model.load_state_dict(torch.load(f'/home/xyli/kaggle/kaggle_HomeCredit/best_nn_fold{fold}.pt'))
#         print('发现可用baseline, 开始加载')   
#     except:
#         print('未发现可用模型, 从0训练')
#     model = model.cuda()
#     model = DataParallel(model)

#     # lr = 1e-3 weight_decay=1e-5
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
#     # adam的优化版本
#     # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     scheduler = None

#     # scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     #     optimizer, 
#     #     milestones=[20,40], 
#     #     gamma=0.1,
#     #     last_epoch=-1
#     # )

# #     loss_fn = nn.BCEWithLogitsLoss()
#     loss_fn = SmoothBCEwLogits(smoothing=0.005) # 0.005

#     best_train_loss = 999.0
#     best_valid_auc = -1
#     for epoch in range(20):
#         start_time = time.time()
#         train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)
#         valid_pred = inference_fn(model, valid_loader, device)
#         valid_auc = roc_auc_score(y_valid, valid_pred)
#         print(
#             f"FOLD{fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
#             f"roc_auc_score={valid_auc:.5f} "
#             f"time: {(time.time() - start_time) / 60:.2f}min "
#             f"lr: {optimizer.param_groups[0]['lr']}"
#         )
#         with open("log.txt", "a") as f:
#             print(
#                 f"FOLD{fold} EPOCH:{epoch:3} train_loss={train_loss:.5f} "
#                 f"roc_auc_score={valid_auc:.5f} "
#                 f"time: {(time.time() - start_time) / 60:.2f}min "
#                 f"lr: {optimizer.param_groups[0]['lr']}", file=f
#             )

#         if train_loss < best_train_loss and valid_auc > best_valid_auc:
#             best_train_loss = train_loss
#             best_valid_auc = valid_auc
#             torch.save(model.module.state_dict(), f"./best_nn_fold{fold}.pt") 
#             print(
#                 f"best_nn_fold{fold}.pt "
#                 f"best_train_loss: {best_train_loss} "
#                 f"best_valid_auc: {best_valid_auc} "
#             )
#             with open("log.txt", "a") as f:
#                 print(
#                     f"best_nn_fold{fold}.pt "
#                     f"best_train_loss: {best_train_loss} "
#                     f"best_valid_auc: {best_valid_auc} ", file=f
#                 )
            
#     fold = fold+1

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
    #     'gpu_use_dp' : True, # 转化float为64精度

    #     # 平衡类别之间的权重  损失函数不会因为样本不平衡而被“推向”样本量偏少的类别中
    #     "sample_weight":'balanced',
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



# ======================================== 训练线性模型 =====================================
""" 加载训练的模型 """

from torch.utils.data import DataLoader

fitted_models_cat1 = []
fitted_models_lgb1 = []
fitted_models_xgb1 = []
fitted_models_nn = []

fitted_models_cat2 = []
fitted_models_lgb2 = []


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
    
class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        
        y_preds = []

        X[cat_cols_829] = X[cat_cols_829].astype("str")
        y_preds += [estimator.predict_proba(X)[:, 1] for estimator in self.estimators[0:5]]
        y_preds += [estimator.predict_proba(X[df_train_386])[:, 1] for estimator in self.estimators[5:10]]
        
        X[cat_cols_829] = X[cat_cols_829].astype("category")
        y_preds += [estimator.predict(X) for estimator in self.estimators[10:15]]
        y_preds += [estimator.predict(X[df_train_386]) for estimator in self.estimators[15:20]]
        
        
        return y_preds

model = VotingModel(fitted_models_cat1 + fitted_models_cat2 + fitted_models_lgb1 + fitted_models_lgb2)


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

class Model_ensemble(nn.Module):
    def __init__(self):
        super(Model_ensemble, self).__init__()
        self.dense1 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = self.dense1(x)
        
        return x


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        features = data['features'].to(device)
        label = data['label'].to(device)
        
#         print(features.shape)
#         print(label.shape)
        outputs = model(features)
        
        loss = loss_fn(outputs, label)
        loss.requires_grad_(True)   #加入此句就行了
        loss.backward()
        optimizer.step()

        final_loss += loss.item()

    if scheduler:
        scheduler.step()
    final_loss /= len(dataloader)

    return final_loss

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)
        with torch.no_grad():
            outputs = model(features)
        
        preds.append(outputs.detach().cpu().numpy())
        # preds.append(outputs.sigmoid().detach().cpu().numpy())

    # preds = np.concatenate(preds).reshape(-1, 1)
    return preds

def mse_fun(y_valid, valid_pred):
    # 将列表转换为PyTorch张量
    y_valid = torch.tensor(y_valid)
    valid_pred = torch.tensor(valid_pred)
    
    # 计算差值
    diff = y_valid - valid_pred

    # 计算差值的平方
    squared_diff = diff ** 2

    # 计算均方误差
    mse = torch.mean(squared_diff)
    
    return mse





for idx_train, idx_valid in cv.split(df_train, y, groups=weeks): # 5折，循环5次

    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train] 
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid] 


    # from IPython import embed
    # embed()

    train_preds = model.predict_proba(X_train)
    train_preds = torch.tensor(train_preds)
    y_train = torch.tensor(y_train.values) # 保持索引都从0开始
    # 这样每一行是20个模型各预测的结果概率 行数=batchsize
    train_preds = torch.tensor(train_preds).T
    
    print('train_preds.shape: ', train_preds.shape)


    valid_preds = model.predict_proba(X_valid)
    naked_preds = np.mean(valid_preds, axis=0)
    naked_score = roc_auc_score(y_valid, naked_preds)
    print('naked_score: ', naked_score)

    valid_preds = torch.tensor(valid_preds)
    y_valid = torch.tensor(y_valid.values) # 保持索引都从0开始
    valid_preds = torch.tensor(valid_preds).T


    train_set = MarketDataset(train_preds, y_train)
    train_loader = DataLoader(train_set, batch_size=15000, shuffle=True, num_workers=1)
    valid_set = MarketDataset(valid_preds, y_valid)
    valid_loader = DataLoader(valid_set, batch_size=15000, shuffle=False, num_workers=1)





    model = Model_ensemble()
    model.load_state_dict(torch.load(f'/home/xyli/kaggle/kaggle_HomeCredit/best_Model_ensemble.pt'))
    model = model.cuda()
    valid_pred = inference_fn(model, valid_loader, device = torch.device("cuda"))
    # 将多个batch(包含多个向量的列表)合并为1个向量
    valid_pred = [item[0] for sublist in valid_pred for item in sublist] 
    valid_auc = roc_auc_score(y_valid.tolist(), valid_pred)
    print("valid_auc: ", valid_auc)
    continue


    model = Model_ensemble()
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-3)
    loss_fn = nn.MSELoss() # 创建MSE损失函数对象

    best_train_loss = 999.0
    best_valid_auc = -1
    for epoch in range(40):
        train_loss = train_fn(model, optimizer, None, loss_fn, train_loader, device = torch.device("cuda"))
        valid_pred = inference_fn(model, valid_loader, device = torch.device("cuda"))

        # from IPython import embed
        # embed()
        # print('valid_pred: ', valid_pred)
        # print('y_valid.tolist(): ', y_valid.tolist())

        # 将多个batch(包含多个向量的列表)合并为1个向量
        valid_pred = [item[0] for sublist in valid_pred for item in sublist] 
        valid_auc = roc_auc_score(y_valid.tolist(), valid_pred)
        print(
            f"EPOCH:{epoch:3} train_loss={train_loss:.5f} "
            f"roc_auc_score={valid_auc:.5f} "
            f"lr: {optimizer.param_groups[0]['lr']}"
        )

        if train_loss < best_train_loss and valid_auc > best_valid_auc:
        # if valid_auc > best_valid_auc:
            best_train_loss = train_loss
            best_valid_auc = valid_auc
            torch.save(model.state_dict(), f"./best_Model_ensemble.pt") 
            print(
                f"best_Model_ensemble.pt "
                f"best_train_loss: {best_train_loss} "
                f"best_valid_auc: {best_valid_auc} "
            )

    


    break # 只用5/4的数据做该线性模型的训练样本，这样比较简单

# ======================================== 训练线性模型 =====================================
