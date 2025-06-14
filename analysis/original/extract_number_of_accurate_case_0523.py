# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:51:02 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np

methods = ['XGBoost', 'MBoost', 'LADBoost', 'GBM', 'AdaBoost', 'AdaBoost_RT']

def get_grouper_index(real_y, thr1, thr2):
    gr1 = np.where(real_y<thr1)[0]
    gr2 = np.where(
        (real_y>=thr1)&(real_y<thr2))[0]
    gr3 = np.where(real_y>=thr2)[0]
    return gr1, gr2, gr3

def get_result_df(methods, group, val1):
    return pd.DataFrame(data={'Model' : methods, 'Group' : [group]*6, 'num': val1})


def get_acc_num(diff_df1, grouper_idx, methods):
    return [((diff_df1['pred'].loc[grouper_idx] - diff_df1[uni_col].loc[grouper_idx])<0).sum() for uni_col in methods]

def get_acc_num_all(diff_df1, methods):
    return [((diff_df1['pred'] - diff_df1[uni_col])<0).sum() for uni_col in methods]


def get_acc_ratio(diff_df1, grouper_idx, methods):
    return [((diff_df1['pred'].loc[grouper_idx] - diff_df1[uni_col].loc[grouper_idx])<0).sum() / grouper_idx.shape[0] for uni_col in methods]

def get_acc_ratio_all(diff_df1, methods):
    return [((diff_df1['pred'] - diff_df1[uni_col])<0).sum() / diff_df1.shape[0] for uni_col in methods]

if __name__ == "__main__":


    final_result_df = pd.DataFrame()
    for city_name in ['seoul','daejeon']:
        for day_type in ['weekdays', 'weekends']:
            for random_state in range(5):
                temp_df = pd.read_csv(r'results/bootstrapped/optimal/%s_%s_%s.csv'%(city_name, day_type, random_state))

                if city_name == 'seoul':
                    idx_123 = get_grouper_index(temp_df.real.values, 4, 40)
                elif city_name == 'daejeon':
                    idx_123 = get_grouper_index(temp_df.real.values, 2, 10)

                diff_df = pd.DataFrame(data={'pred': np.abs(temp_df.real - temp_df.pred),
                                             'XGBoost': np.abs(temp_df.real - temp_df.XGBoost),
                                             'MBoost' : np.abs(temp_df.real - temp_df.MBoost),
                                             'LADBoost' : np.abs(temp_df.real - temp_df.LADBoost),
                                             'GBM' : np.abs(temp_df.real - temp_df.GBM),
                                             'AdaBoost': np.abs(temp_df.real - temp_df.AdaBoost),
                                             'AdaBoost_RT' : np.abs(temp_df.real - temp_df['AdaBoost_RT'])})

                result_df = pd.concat([
                    get_result_df(methods, 'Group0', get_acc_num(diff_df, np.where(temp_df.real==0)[0], methods)),
                    get_result_df(methods, 'Group1', get_acc_num(diff_df, idx_123[0], methods)),
                    get_result_df(methods, 'Group2', get_acc_num(diff_df, idx_123[1], methods)),
                    get_result_df(methods, 'Group3', get_acc_num(diff_df, idx_123[2], methods)),
                    get_result_df(methods, 'All', get_acc_num_all(diff_df,
                                                                    methods))]).reset_index(drop=True)

                result_df['city_name'] = city_name
                result_df['day_type'] = day_type
                result_df['rs'] = random_state
                final_result_df = pd.concat([final_result_df, result_df])
                print(city_name, day_type, random_state)
    final_result_df = final_result_df.reset_index(drop=True)[['city_name', 'day_type', 'rs','Group', 'Model', 'num']]

    #final_result_df.to_csv(r'accurate_case_num_group0_seoul_daejeon_weekdays_weekends_bootstrap_5.csv')
