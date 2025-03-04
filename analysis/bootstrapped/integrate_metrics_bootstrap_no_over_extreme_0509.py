# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:21:49 2024

@author: dohyeon
"""



import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae


def get_smape(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum(np.abs(pred_y-real_y) / ((real_y+pred_y)/2))/real_y.shape[0]

def get_metrics(real_y, pred_y):

    idx_zero_real = np.where(real_y==0)[0]
    idx_zero_pred = np.where(pred_y==0)[0]
    idx_both_zero = np.intersect1d(idx_zero_real, idx_zero_pred)
    idx_for_smape = np.setdiff1d(np.arange(real_y.shape[0]), idx_both_zero)


    rmse1 = mse(real_y, pred_y) ** 0.5
    mae1 = mae(real_y, pred_y)
    mape1 = mape(real_y[real_y>0], pred_y[real_y>0])
    smape1 = get_smape(real_y[idx_for_smape], pred_y[idx_for_smape])
    smape_zero1 = get_smape(real_y[real_y>0], pred_y[real_y>0])
    return [rmse1, mae1 , mape1, smape1, smape_zero1]

def get_grouper_index(real_y, thr1, thr2):
    gr1 = np.where(real_y<thr1)[0]
    gr2 = np.where(
        (real_y>=thr1)&(real_y<thr2))[0]
    gr3 = np.where(real_y>=thr2)[0]
    return gr1, gr2, gr3

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



def get_group_metric_df_extreme(whole_df, grouper_index):
    group1_df = get_group_metrics(whole_df, grouper_index[0],'Group1')
    group2_df = get_group_metrics(whole_df, grouper_index[1],'Group2')
    group3_df = get_group_metrics(whole_df, grouper_index[2],'Group3')
    all_df = get_group_metrics(whole_df, np.arange(whole_df.shape[0]), 'Whole')
    return pd.concat([group1_df, group2_df, group3_df, all_df]).reset_index(drop=True)
#%%
if __name__ == "__main__":

    #city_name = 'seoul'
    #day_type = 'weekdays'
    #random_state = 0

    final_result_df = pd.DataFrame()
    for city_name in ['seoul','daejeon']:
        for day_type in ['weekdays', 'weekends']:
            for random_state in range(5):
                temp_df = pd.read_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, random_state))
                if city_name == 'seoul':
                    result_df = get_group_metric_df_extreme(temp_df, get_grouper_index(temp_df.real.values, 4, 40))
                elif city_name == 'daejeon':
                    result_df = get_group_metric_df_extreme(temp_df, get_grouper_index(temp_df.real.values, 2, 10))
                result_df['city_name'] = city_name
                result_df['day_type'] = day_type
                result_df['rs'] = random_state
                final_result_df = pd.concat([final_result_df, result_df])
                print(random_state)

    final_result_df = final_result_df.reset_index(drop=True)[['city_name', 'day_type', 'rs','Group', 'Model', 'RMSE', 'MAE', 'MAPE', 'sMAPE', 'sMAPE_0']]


    final_result_df.to_csv(r'prediction_performance_seoul_daejeon_weekdays_weekends_bootstrap_over_extreme_5.csv', index=False)














