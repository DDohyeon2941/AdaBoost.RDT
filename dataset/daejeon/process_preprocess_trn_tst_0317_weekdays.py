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

"""
Revision1으로 모듈 수정

작성일: 2023-03-23 [인구통계 변수 중 가구수에서 0인 경우가 발생함]

2019년, 대전의 경우, ID 22번의 경우, 사용하지 않음

"""

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


def masking_peak_hours(uni_shour):
    peak_hours = [8, 18]

    output = 0

    if uni_shour in peak_hours:
        output = 1
    return output

#%%
if __name__ == "__main__":

    train_pkl, test_pkl = pd.read_pickle(r'2018_total_0317.pkl'), pd.read_pickle(r'2019_total_0317.pkl')

    trn_df = train_pkl.loc[((train_pkl['isholy'] < 1) & (train_pkl['dow'].isin([0,1,2,3,4])))]
    tst_df = test_pkl.loc[((test_pkl['isholy'] < 1) & (test_pkl['dow'].isin([0,1,2,3,4])))]

    tst_df = (tst_df.reset_index().loc[tst_df.reset_index().ID !=22 ]).set_index(['date','ID'])

    #trn_df.loc[:, 'hour'] = trn_df.reset_index().date.dt.hour.values
    #tst_df.loc[:, 'hour'] = tst_df.reset_index().date.dt.hour.values

    #trn_df.loc[:,'ispeak'] = [masking_peak_hours(xx) for xx in trn_df['hour']]
    #tst_df.loc[:,'ispeak'] = [masking_peak_hours(xx) for xx in tst_df['hour']]


    #%%
    used_cols = ['food','cafe','tour','bank','stat_num','uni_dist','subway','bus',
                 'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','solar','temp','wind','log_rain2','humidity']


    a,b,c,d = split_x_y(trn_df, tst_df, used_cols)
    #uu.save_gpickle('preprocessed_dataset_0815.pickle', (a, b, c, d))


    uu.save_gpickle(uu.opj(r'preprocessed_dataset_0317_weekdays.pickle'), (a, b, c, d))


#%%

    """
    tst_pkl = pd.read_pickle(r'D:\project_repository\flow_prediction\dock\experiment\modules\202203\proposed_method\preprocess\전처리_이동거리X_0329\2019_total_0728.pkl')
    tst_pkl = tst_pkl.loc[((tst_pkl['isholy'] < 1) & (tst_pkl['dow'].isin([0,1,2,3,4])))]
    
    used_cols = ['food','cafe','tour','bank','num_of_sta','uni_dist','subway','bus',
                 'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','sun','temp','wind','log_rain2','humidity','cloud','y']
    tst_pkl = tst_pkl[used_cols]
    """