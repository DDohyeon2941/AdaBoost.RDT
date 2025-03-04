# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:01:34 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

"""integrate_synthetic_results.py를 그대로 활용, 2024-02-11(일) 오전 11:38"""

"""comparison의 경우, 로그변환이 된 상태로 불러와짐. 따라서 exponential 변환을 취해줘야함, 
    
    prediction_performance_seoul_weekdays_updated_model_model6_minmax.csv 부터 이게 적용됨 [2024-02-21, 오후 1:47]

"""

def get_smape(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum(np.abs(pred_y-real_y) / ((real_y+pred_y+1e-14)/2))/real_y.shape[0]

def get_metrics(real_y, pred_y):
    rmse1 = mse(real_y, pred_y) ** 0.5
    mae1 = mae(real_y, pred_y)
    mape1 = mape(real_y[real_y>0], pred_y[real_y>0])
    smape1 = get_smape(real_y, pred_y)
    smape_zero1 = get_smape(real_y[real_y>0], pred_y[real_y>0])
    return [rmse1, mae1 , mape1, smape1, smape_zero1]

def get_grouper_index(real_y, thr1, thr2):
    gr1 = np.where(real_y<thr1)[0]
    gr2 = np.where(
        (real_y>=thr1)&(real_y<thr2))[0]
    gr3 = np.where(real_y>=thr2)[0]
    return gr1, gr2, gr3

#%%
def get_group_metrics(whole_df, group_index, group_name):
    real_y = whole_df['real'].values

    info_li = []
    metrics_li = []
    for uni_model in whole_df.columns[1:]:
        pred_y = whole_df[uni_model].values
        metrics_1d = get_metrics(real_y[group_index], pred_y[group_index])
        info_li.append([group_name,uni_model])
        metrics_li.append(metrics_1d)


    output1 = pd.DataFrame(data=info_li, columns=['Group','Model'])
    output2 = pd.DataFrame(data=metrics_li, columns=['RMSE','MAE','MAPE','sMAPE','sMAPE_0'])
    return pd.concat([output1, output2], axis=1)


def get_group_metric_df_clean(whole_df, grouper_index):
    group1_df = get_group_metrics(whole_df, grouper_index[0],'Group1')
    group2_df = get_group_metrics(whole_df, grouper_index[1],'Group2')
    all_df = get_group_metrics(whole_df, np.arange(whole_df.shape[0]), 'Whole')
    return pd.concat([group1_df, group2_df, all_df]).reset_index(drop=True)


def get_group_metric_df_extreme(whole_df, grouper_index):
    group1_df = get_group_metrics(whole_df, grouper_index[0],'Group1')
    group2_df = get_group_metrics(whole_df, grouper_index[1],'Group2')
    group3_df = get_group_metrics(whole_df, grouper_index[2],'Group3')
    all_df = get_group_metrics(whole_df, np.arange(whole_df.shape[0]), 'Whole')
    return pd.concat([group1_df, group2_df, group3_df, all_df]).reset_index(drop=True)


def convert_exp_comparison(comparison_df):

    model_list = ['AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']
    for uni_model in model_list:
        comparison_df[uni_model] = np.exp(comparison_df[uni_model].values)-1
    return comparison_df



temp_df = pd.read_csv(r'y_val_cdf_weight_daejeon_weekdays_model8_lasso_avg_threshold_ratio.csv')
temp_df = (temp_df.loc[temp_df['info'] == 'pred_y'])[temp_df.columns[:-1]].reset_index(drop=True)
temp_df1 = convert_exp_comparison(pd.read_csv(r'daejeon_real_pred_weekdays.csv'))


temp_df1
#%%
#concated_df = pd.concat([temp_df, temp_df1[temp_df1.columns[5:]]], axis=1)
concated_df = pd.concat([result_df[result_df.columns[:-1]], temp_df1[temp_df1.columns[5:]]], axis=1)
grouper_idxs = get_grouper_index(concated_df.real.values, 2, 10)

#%%
get_group_metric_df_extreme(concated_df, grouper_idxs).to_csv(r'prediction_performance_daejeon_weekdays_model8_lasso_avg_threshold_ratio.csv', index=False)

