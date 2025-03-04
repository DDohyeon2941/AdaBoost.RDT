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

if __name__ == "__main__":

    cols = ['ID', 'date', 'r_date' ,'tot_pop', 'saup', 'jong', '2049', 'single', 'y_ratio',
       'one_ratio', 'tot_ga', 'food', 'cafe', 'tour', 'bank', 'num_of_sta',
       'uni_dist', 'subway', 'bus', 'temp', 'humidity', 'rain',
       'log_rain2', 'wind', 'sun', 'snow', 'dust', 'cloud', 'isholy', 'season',
       'tmask', 'dow']
    #%%

    station_df = pd.read_csv(r'dateinfo_0311.csv')
    p1 = pd.read_pickle(r'2018_features_0311.pkl')
    p2 = pd.read_pickle(r'2018_sel_demands_0311.pkl')


    #%%
    #p2.loc[p2.rent_id.isin(station_df['0'])].groupby(['r_time','rent_id']).count()['age']

    p1=p1[cols]

    p22=p2.loc[p2.rent_id.isin(station_df['0'].dropna().astype(int).values)].groupby(['r_time','rent_id']).count()['age']


    p1.loc[:,'y'] = 0.0

    p12=p1.set_index(['date','ID'])

    p12.loc[p22.index,'y'] = p22.values

    p12.to_pickle(r'2018_total_0311.pkl')
    #%%
    station_df = pd.read_csv(r'dateinfo_0311.csv')
    p3 = pd.read_pickle(r'2019_features_0311.pkl')
    p4 = pd.read_pickle(r'2019_sel_demands_0311.pkl')
    p3=p3[cols]
    p42=p4.loc[p4.rent_id.isin(station_df['1'].values)].groupby(['r_time','rent_id']).count()['fare']
    p3.loc[:,'y'] = 0.0
    p32=p3.set_index(['date','ID'])
    p32.loc[p42.index,'y'] = p42.values
    p32.to_pickle(r'2019_total_0311.pkl')
