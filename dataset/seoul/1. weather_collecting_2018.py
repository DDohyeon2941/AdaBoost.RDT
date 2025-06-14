# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:20:50 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 18:35:23 2019

@author: User
"""

# 날씨 변수 전처리.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import configparser
import sys
import pdb


def fill_wind(df_, n_days):
    null_idx = df_[df_[0].isnull()].index
    candi_arr = np.zeros((null_idx.shape[0], n_days*2))

    for uidx, uni_day in enumerate(np.arange(1,n_days+1,1)):
        after_idx = null_idx + pd.DateOffset(days=uni_day)
        before_idx = null_idx - pd.DateOffset(days=uni_day)

        candi_arr[:, uidx*2] = df_.loc[before_idx].values.squeeze()
        candi_arr[:, uidx*2 +1] = df_.loc[after_idx].values.squeeze()
    df_.loc[null_idx, 0] = np.around(np.nanmean(candi_arr, axis=1),1)
    return df_



def fill_cloud(df_, n_hours):
    null_idx = df_[df_[0].isnull()].index
    candi_arr = np.zeros((null_idx.shape[0], int(n_hours)*2))

    for uidx, uni_hour in enumerate(np.arange(1,n_hours+1,1)):
        after_idx = null_idx + pd.DateOffset(hours=uni_hour)
        before_idx = null_idx - pd.DateOffset(hours=uni_hour)

        candi_arr[:, uidx*2] = df_.loc[before_idx].values.squeeze()
        candi_arr[:, uidx*2 +1] = df_.loc[after_idx].values.squeeze()
    df_.loc[null_idx, 0] = np.around(np.nanmean(candi_arr, axis=1),1)
    return df_




#%%

tst_weather = pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\weather_2018_data.csv')
tst_weather.head(3).T

date_range_idx = pd.date_range(start="20180101", end="20190101", freq='h')[:-1]
date_range_df = pd.DataFrame(index=date_range_idx, data=np.ones(date_range_idx.shape[0])*np.nan)

t1=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\temp_2018.csv', encoding='cp949')
t1=t1[t1.columns[1:]]
t1.columns = ['date', 'temp']
t1=t1[~t1.isnull()['temp']]

new_t1 = date_range_df.copy(deep=True)
new_t1.loc[t1[t1.columns[0]],0] = t1[t1.columns[-1]].values

new_t1.loc[new_t1[0].isnull()]

new_t1 = fill_wind(new_t1, 3).reset_index()
new_t1.columns = ['date', 'temp']
new_t1
#%%

t2=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\wind_2018.csv', encoding='cp949')
t2=t2[t2.columns[1:]]
t2.columns = ['date', 'wind']
t2=t2[~t2.isnull()['wind']]

new_t2 = date_range_df.copy(deep=True)
new_t2.loc[t2[t2.columns[0]],0] = t2[t2.columns[-1]].values
new_t2 = fill_wind(new_t2, 3)
new_t2 = new_t2.reset_index()
new_t2.columns = ['date', 'wind']
new_t2
#%%

t3=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\humidity_2018.csv', encoding='cp949')
t3=t3[t3.columns[1:]]
t3.columns = ['date', 'huminity']
t3.loc[:, 'date'] = pd.to_datetime(t3['date'])
t3 = t3.loc[t3.date <  pd.to_datetime(t3.date.iloc[-1])]

new_t3 = date_range_df.copy(deep=True)

new_t3.loc[t3[t3.columns[0]],0] = t3[t3.columns[-1]].values

new_t3 = fill_cloud(new_t3, 1.0)
new_t3 = new_t3.reset_index()
new_t3.columns = ['date', 'humidity']
new_t3


#%%

t4=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\sun_2018.csv', encoding='cp949')
t4=t4[t4.columns[1:]]
t4.columns = ['date', 'sun']

new_t4 = date_range_df.copy(deep=True)
new_t4.loc[t4[t4.columns[0]],0] = t4[t4.columns[-1]].values
new_t4

new_t4 = new_t4.fillna(0.0)
new_t4 = new_t4.reset_index()
new_t4.columns = ['date', 'sun']

#%%

t5=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\snow_2018.csv', encoding='cp949')
t5=t5[t5.columns[1:]]
t5.columns = ['date', 'snow']

new_t5 = date_range_df.copy(deep=True)
new_t5.loc[t5[t5.columns[0]],0] = t5[t5.columns[-1]].values
new_t5 = new_t5.fillna(0.0)
new_t5 = new_t5.reset_index()
new_t5.columns = ['date', 'snow']


#%%
t6=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\dust_2018.csv', encoding='cp949', skiprows=[0,1,2,3])

t6=t6[t6.columns[2:]]
t6.columns = ['date', 'dust']

new_t6 = date_range_df.copy(deep=True)
new_t6.loc[t6[t6.columns[0]],0] = t6[t6.columns[-1]].values
new_t6 = new_t6.fillna(0.0)
new_t6 = new_t6.reset_index()
new_t6.columns = ['date', 'dust']

#%%
t7=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\rain_2018_hour.csv', encoding='cp949')
t7=t7[t7.columns[2:-2]]
t7.columns = ['date', 'rain']
t7 = t7.fillna(0.0)
t7.loc[:, 'rain_log2'] = np.log(t7.rain.rolling(window=2).mean()+1)
t7 = t7.fillna(0.0)
t7.loc[:, 'date'] = pd.to_datetime(t7['date'])
#%%
t8=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018\cloud_2018.csv', encoding='cp949')
t8=t8[t8.columns[2:]]
t8.columns = ['date', 'cloud']
t8.loc[:, 'date'] = pd.to_datetime(t8['date'])
t8 = t8.astype({'cloud':float})

new_t8 = date_range_df.copy(deep=True)
new_t8.loc[t8[t8.columns[0]],0] = t8[t8.columns[-1]].values
new_t8 = fill_cloud(new_t8, 1.0)
new_t8 = new_t8.reset_index()
new_t8.columns = ['date', 'cloud']
new_t8

#%%




tt1 = pd.merge(pd.merge(new_t1,new_t3), t7)
tt1.loc[:, 'date'] = pd.to_datetime(tt1.date)

tt2 = pd.merge(pd.merge(pd.merge(pd.merge(new_t2, new_t4), new_t5), new_t6), new_t8)
tt2.loc[:, 'date'] = pd.to_datetime(tt2.date)

tt3 = pd.merge(tt1,tt2)

tt3.to_csv(r'weather_2018.csv',index=False)






    
    