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
작성일: 2023-03-22, latex에 업로드한 자
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
    gr1 = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    gr2 = [6, 7, 8, 9, 10, 11]
    gr3 = [12, 13, 14]
    gr4 = [15, 16, 17, 18, 19]

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

#%%
if __name__ == "__main__":

    train_pkl, test_pkl = pd.read_pickle(r'2018_total_0317.pkl'), pd.read_pickle(r'2019_total_0317.pkl')
    #%%
    #train_pkl, test_pkl = pd.read_pickle(r'2018_total_0728.pkl'), pd.read_pickle(r'2019_total_0728.pkl')

    trn_df_week = train_pkl.loc[((train_pkl['isholy'] < 1) | (train_pkl['dow'].isin([0,1,2,3,4])))]
    trn_df_holy = train_pkl.loc[((train_pkl['isholy'] >= 1) | (train_pkl['dow'].isin([5,6])))]

    trn_df_week.loc[:, 'hour'] = trn_df_week.reset_index().date.dt.hour.values
    trn_df_holy.loc[:, 'hour'] = trn_df_holy.reset_index().date.dt.hour.values

    #trn_df_week.loc[:, 'tmask'] = [grouping_hours_func(uni_season=xx, uni_hour=yy) for xx, yy in trn_df_week[['season','hour']].values]

    #trn_df_holy.loc[:, 'tmask'] = [grouping_hours_func(uni_season=xx, uni_hour=yy) for xx, yy in trn_df_holy[['season','hour']].values]


    #%%
    df1 = trn_df_week.reset_index().copy(deep=True)
    df1.loc[:, 'time'] = df1.date.dt.hour
    df1 = df1.replace({'season':{1:'winter', 2:'spring', 3:'summer', 4:'fall'}})
    df11 = (df1[['season','hour','y']].groupby(['hour','season']).mean().unstack())['y']
    df11

    df2 = trn_df_holy.reset_index().copy(deep=True)
    df2.loc[:, 'time'] = df2.date.dt.hour
    df2 = df2.replace({'season':{1:'winter', 2:'spring', 3:'summer', 4:'fall'}})
    #%%
    fig1, axes1 = plt.subplots(1,2, figsize=(14,6))
    axes1[0].set_title('weekdays')
    axes1[1].set_title('weekends')

    (df1[['season','time','y']].groupby(['time','season']).mean().unstack())['y'].plot(grid=True, xticks=np.arange(24),ax=axes1[0])
    (df2[['season','time','y']].groupby(['time','season']).mean().unstack())['y'].plot(grid=True, xticks=np.arange(24),ax=axes1[1])


    #%% 2022-10-11 Weekday figure

    max_y = 0.9

    fig1, axes1 = plt.subplots(1,1, figsize=(10,6))

    df11[['spring','summer','fall','winter']].plot(grid=False, xticks=np.arange(24), ax=axes1)
    axes1.set_xlabel('Hour', fontsize='x-large')
    axes1.set_ylabel('Demand', fontsize='x-large')
    axes1.fill_between([7,8,9,10], [0,0,0,0], [max_y, max_y, max_y, max_y], color='black', alpha=0.3)
    axes1.fill_between([17,18,19,20], [0,0,0,0], [max_y, max_y, max_y,max_y], color='black', alpha=0.3)
    axes1.legend(loc='upper left', fontsize='x-large')
    #axes1.set_title("Weekdays")
    axes1.set_ylim(0, max_y)
    axes1.set_xlim(0, 23)
    axes1.tick_params(axis='both', labelsize='x-large')
    axes1.legend(fontsize='x-large')

    plt.savefig('Daejeon_Train_Weekday_0322.png', dpi=1024, transparent=True, bbox_inches='tight')
    plt.close()
    #%% 2022-10-11 Weekend figure

    fig1, axes1 = plt.subplots(1,1, figsize=(10,6))
    ((df2[['season','time','y']].groupby(['time','season']).mean().unstack())['y'])[['spring','summer','fall','winter']].plot(grid=False, xticks=np.arange(24), ax=axes1)
    axes1.set_xlabel('Hour', fontsize='x-large')
    axes1.set_ylabel('Demand', fontsize='x-large')
    axes1.fill_between([14, 15, 16, 17, 18,19], [0,0,0,0,0,0], [max_y, max_y, max_y,
                                                               max_y, max_y, max_y], color='black',
                          alpha=0.3)
    axes1.fill_between([18, 19, 20, 21, 22, 23], [0, 0,0,0,0,0], [max_y, max_y, max_y,
                                                               max_y, max_y, max_y],
                          color='green', alpha=0.3)
    axes1.legend(loc='upper left', fontsize='x-large')
    #axes1.set_title("Weekends")
    axes1.set_ylim(0, max_y)
    axes1.set_xlim(0, 23)
    axes1.tick_params(axis='both', labelsize='x-large')
    axes1.legend(fontsize='x-large')

    plt.savefig('Daejeon_Train_Weekend_0322.png', dpi=1024, transparent=True, bbox_inches='tight')
    plt.close()
