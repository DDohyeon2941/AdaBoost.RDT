# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:17:15 2022

@author: dohyeon
"""


import pandas as pd
import numpy as np
import time
import pickle
import gzip
import ipdb
import user_utils as uu
from datetime import date
import holidays
import itertools
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
"""
Revision1으로 모듈 수정

작성일: 2023-03-23 [인구통계 변수 중 가구수에서 0인 경우가 발생함]

2019년, 대전의 경우, ID 22번의 경우, 사용하지 않음

"""

def grouping_hours_summer(uni_shour):
    gr1 = [23, 0, 1, 2, 3, 4, 5]
    gr2 = [6, 7, 8, 9, 10, 11]
    gr3 = [12, 13, 14, 15, 16, 17]
    gr4 = [18, 19, 20, 21, 22]

    output = 0
    if uni_shour in gr1:
        output = 1
    elif uni_shour in gr2:
        output = 2
    elif uni_shour in gr3:
        output = 3
    elif uni_shour in gr4:
        output = 4
    return output

def grouping_hours_others(uni_shour):
    gr1 = [19,20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    gr2 = [6, 7, 8, 9, 10, 11]
    gr3 = [12, 13]
    gr4 = [14, 15, 16, 17, 18]

    output = 0
    if uni_shour in gr1:
        output = 1
    elif uni_shour in gr2:
        output = 2
    elif uni_shour in gr3:
        output = 3
    elif uni_shour in gr4:
        output = 4
    return output



def grouping_hours_func(uni_season, uni_hour):
    SUMMER_VAL = 3
    if uni_season == SUMMER_VAL:
        return grouping_hours_summer(uni_hour)
    else:
        return grouping_hours_others(uni_hour)




def masking_peak_hour(uni_season, uni_shour):

    if uni_season in [1,2]:
        peak_hour = 16
    elif uni_season == 3:
        peak_hour = 20
    elif uni_season == 4:
        peak_hour = 17
    output = 0

    if uni_shour == peak_hour:
        output = 1
    return output


def split_x_y_peak(train_df, test_df, sel_cols):
    raw_scaler = StandardScaler()
    raw_scaler.fit(train_df[sel_cols].values)

    train_y = train_df.y.values
    train_x = np.hstack((raw_scaler.transform(train_df[sel_cols].values),
                   np.hstack((pd.get_dummies(train_df.season).values,
                              pd.get_dummies(train_df.tmask).values,
                              pd.get_dummies(train_df.ispeak).values))))
    test_y = test_df.y.values
    test_x = np.hstack((raw_scaler.transform(test_df[sel_cols].values),
                   np.hstack((pd.get_dummies(test_df.season).values,
                              pd.get_dummies(test_df.tmask).values,
                              pd.get_dummies(test_df.ispeak).values))))
    return train_x, train_y, test_x, test_y



def split_x_y(train_df, test_df, sel_cols):
    raw_scaler = StandardScaler()
    raw_scaler.fit(train_df[sel_cols].values)

    train_y = train_df.y.values
    train_x = np.hstack((raw_scaler.transform(train_df[sel_cols].values),
                   np.hstack((pd.get_dummies(train_df.season).values,
                              pd.get_dummies(train_df.tmask).values))))
    test_y = test_df.y.values
    test_x = np.hstack((raw_scaler.transform(test_df[sel_cols].values),
                   np.hstack((pd.get_dummies(test_df.season).values,
                              pd.get_dummies(test_df.tmask).values))))
    return train_x, train_y, test_x, test_y

#%%
if __name__ == "__main__":

    train_pkl, test_pkl = pd.read_pickle(r'2018_total_0317.pkl'), pd.read_pickle( '2019_total_0317.pkl')

    trn_df = train_pkl.loc[((train_pkl['isholy'] >= 1) | (train_pkl['dow'].isin([5,6])))]
    tst_df = test_pkl.loc[((test_pkl['isholy'] >= 1) | (test_pkl['dow'].isin([5,6])))]


    tst_df = (tst_df.reset_index().loc[tst_df.reset_index().ID !=22 ]).set_index(['date','ID'])


    trn_df.loc[:, 'hour'] = trn_df.reset_index().date.dt.hour.values
    tst_df.loc[:, 'hour'] = tst_df.reset_index().date.dt.hour.values
    #%%
    trn_df.loc[:, 'tmask'] = [grouping_hours_func(uni_season=xx, uni_hour=yy) for xx, yy in trn_df[['season','hour']].values]

    tst_df.loc[:, 'tmask'] = [grouping_hours_func(uni_season=xx, uni_hour=yy) for xx, yy in tst_df[['season','hour']].values]
    #%%

    #trn_df.loc[:,'ispeak'] = [masking_peak_hour(uni_season=xx, uni_shour=yy) for xx, yy in trn_df[['season','hour']].values]
    #tst_df.loc[:,'ispeak'] = [masking_peak_hour(uni_season=xx, uni_shour=yy) for xx, yy in tst_df[['season','hour']].values]


    #%%
    used_cols = ['food','cafe','tour','bank','stat_num','uni_dist','subway','bus',
                 'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','solar','temp','wind','log_rain2','humidity']

    a,b,c,d = split_x_y(trn_df, tst_df, used_cols)
    uu.save_gpickle('preprocessed_dataset_0317_holy.pickle', (a, b, c, d))


