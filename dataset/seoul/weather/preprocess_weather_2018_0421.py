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


def fill_wind(df_, n_days):
    col_name = df_.columns[0]
    scale = 2

    null_idx = df_[df_[col_name].isnull()].index
    candi_arr = np.zeros((null_idx.shape[0], n_days*scale))

    for uidx, uni_day in enumerate(np.arange(1,n_days+1,1)):
        after_idx = null_idx + pd.DateOffset(days=uni_day)
        before_idx = null_idx - pd.DateOffset(days=uni_day)

        candi_arr[:, uidx*scale] = df_.loc[before_idx].values.squeeze()
        candi_arr[:, uidx*scale +1] = df_.loc[after_idx].values.squeeze()
    df_.loc[null_idx, col_name] = np.around(np.nanmean(candi_arr, axis=1),1)
    return df_



def fill_cloud(df_, n_hours):
    col_name = df_.columns[0]
    scale = 2
    keep_filling = True
    while keep_filling is True:
        null_idx = df_[df_[col_name].isnull()].index
        candi_arr = np.zeros((null_idx.shape[0], int(n_hours)*scale))
    
        for uidx, uni_hour in enumerate(np.arange(1,n_hours+1,1)):
            after_idx = null_idx + pd.DateOffset(hours=uni_hour)
            before_idx = null_idx - pd.DateOffset(hours=uni_hour)
    
            candi_arr[:, uidx*scale] = df_.loc[before_idx].values.squeeze()
            candi_arr[:, uidx*scale +1] = df_.loc[after_idx].values.squeeze()
        df_.loc[null_idx, col_name] = np.around(np.nanmean(candi_arr, axis=1),1)
        if not df_[col_name].isnull().sum() > 0:
            keep_filling = False

    return df_


weather_dir = r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2018'


date_range_idx = pd.date_range(start="20180101", end="20190101", freq='h')[:-1]
date_range_df = pd.DataFrame(index=date_range_idx, data=np.ones(date_range_idx.shape[0])*np.nan)

#%%

f_name = 'temp'
t1=pd.read_csv(os.path.join(weather_dir, r'%s_2018.csv'%(f_name)), encoding='cp949')
t1=t1[t1.columns[1:]]
t1.columns = ['date', f_name]
t1=t1[~t1.isnull()[f_name]]

new_t1 = date_range_df.copy(deep=True)
new_t1.loc[t1[t1.columns[0]],0] = t1[t1.columns[-1]].values

new_t1.loc[new_t1[0].isnull()]

new_t1 = fill_wind(new_t1, 3).reset_index()
new_t1.columns = ['date', f_name]
new_t1
#%%

f_name = 'wind'
t2=pd.read_csv(os.path.join(weather_dir, r'%s_2018.csv'%(f_name)), encoding='cp949')
t2=t2[t2.columns[1:]]
t2.columns = ['date', f_name]
t2=t2[~t2.isnull()[f_name]]

new_t2 = date_range_df.copy(deep=True)
new_t2.loc[t2[t2.columns[0]],0] = t2[t2.columns[-1]].values
new_t2 = fill_wind(new_t2, 3)
new_t2 = new_t2.reset_index()
new_t2.columns = ['date', f_name]
new_t2
#%%
f_name = 'humidity'
t3=pd.read_csv(os.path.join(weather_dir, r'%s_2018.csv'%(f_name)), encoding='cp949')
t3=t3[t3.columns[1:]]
t3.columns = ['date', 'huminity']
t3.loc[:, 'date'] = pd.to_datetime(t3['date'])
t3 = t3.loc[t3.date <  pd.to_datetime(t3.date.iloc[-1])]

new_t3 = date_range_df.copy(deep=True)

new_t3.loc[t3[t3.columns[0]],0] = t3[t3.columns[-1]].values

new_t3 = fill_cloud(new_t3, 1.0)
new_t3 = new_t3.reset_index()
new_t3.columns = ['date', f_name]
new_t3


#%%
f_name = 'sun'
t4=pd.read_csv(os.path.join(weather_dir, r'%s_2018.csv'%(f_name)), encoding='cp949')
t4=t4[t4.columns[1:]]
t4.columns = ['date', f_name]

new_t4 = date_range_df.copy(deep=True)
new_t4.loc[t4[t4.columns[0]],0] = t4[t4.columns[-1]].values
new_t4

new_t4 = new_t4.fillna(0.0)
new_t4 = new_t4.reset_index()
new_t4.columns = ['date', f_name]

#%%
f_name = 'snow'
t5=pd.read_csv(os.path.join(weather_dir, r'%s_2018.csv'%(f_name)), encoding='cp949')
t5=t5[t5.columns[1:]]
t5.columns = ['date', f_name]

new_t5 = date_range_df.copy(deep=True)
new_t5.loc[t5[t5.columns[0]],0] = t5[t5.columns[-1]].values
new_t5 = new_t5.fillna(0.0)
new_t5 = new_t5.reset_index()
new_t5.columns = ['date', f_name]


#%%
f_name = 'dust'
t6=pd.read_csv(os.path.join(weather_dir, r'%s_2018.csv'%(f_name)), encoding='cp949', skiprows=[0,1,2,3])

t6=t6[t6.columns[2:]]
t6.columns = ['date', f_name]

new_t6 = date_range_df.copy(deep=True)
new_t6.loc[t6[t6.columns[0]],0] = t6[t6.columns[-1]].values
new_t6 = new_t6.fillna(0.0)
new_t6 = new_t6.reset_index()
new_t6.columns = ['date', f_name]

#%%


t7=pd.read_csv(os.path.join(weather_dir, r'rain_2018_hour.csv'), encoding='cp949')
t7=t7[t7.columns[2:-2]]
t7.columns = ['date', 'rain']
t7 = t7.fillna(0.0)
t7.loc[:, 'rain_log2'] = np.log(t7.rain.rolling(window=2).mean()+1)
t7 = t7.fillna(0.0)
t7.loc[:, 'date'] = pd.to_datetime(t7['date'])
#%%
f_name = 'cloud'

t8=pd.read_csv(os.path.join(weather_dir, r'%s_2018.csv'%(f_name)), encoding='cp949')
t8=t8[t8.columns[2:]]
t8.columns = ['date', f_name]
t8.loc[:, 'date'] = pd.to_datetime(t8['date'])
t8 = t8.astype({f_name:float})

new_t8 = date_range_df.copy(deep=True)
new_t8.loc[t8[t8.columns[0]],0] = t8[t8.columns[-1]].values
new_t8 = fill_cloud(new_t8, 1.0)
new_t8 = new_t8.reset_index()
new_t8.columns = ['date', f_name]
new_t8

#%% concat the preprocessed 2018 weather data




tt1 = pd.merge(pd.merge(new_t1,new_t3), t7)
tt1.loc[:, 'date'] = pd.to_datetime(tt1.date)

tt2 = pd.merge(pd.merge(pd.merge(pd.merge(new_t2, new_t4), new_t5), new_t6), new_t8)
tt2.loc[:, 'date'] = pd.to_datetime(tt2.date)

tt3 = pd.merge(tt1,tt2)


#tt3.to_csv(r'weather_2018.csv',index=False)


