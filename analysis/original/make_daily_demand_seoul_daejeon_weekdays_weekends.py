# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:55:20 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


temp_df = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')



gb_date_df = temp_df.groupby(['date']).sum()[['real', 'pred', 'median_pred', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']]



gb_date_hour_df = temp_df.groupby(['date','hour']).sum()[['real', 'pred', 'median_pred', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']]




ext_station = np.array([ 113,  152,  202,  203,  204,  205,  206,  207,  210,  211,  212,
          217,  222,  247,  248,  259,  272,  409,  412,  418,  419,  502,
          563,  565,  567,  578,  583,  723, 1153, 1195, 1210, 1222, 1243,
          1278, 1295, 1308, 1834, 1839, 1851, 1906, 1911, 1961, 1986, 2002,
          2102, 2219, 2255, 2316, 2348, 2615, 2701, 3533])

ext_date = temp_df.loc[temp_df.real>=40].date.unique()


#%%


fig1, axes1 = plt.subplots(1,1, figsize=(10,8))

temp_df.loc[temp_df.station.isin(ext_station)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='b')

temp_df.loc[~temp_df.station.isin(ext_station)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='r', )

axes1.set_xlabel('Date', fontsize='xx-large')
axes1.set_ylabel('Daily Average Demand', fontsize='xx-large')
axes1.tick_params(axis='both', labelsize='x-large')

plt.legend(['Station with extreme events (52)','Station without extreme events (1442)'], fontsize='large', loc='upper left')


plt.savefig('seoul_weekdays_daily_demand.png', dpi=1024, transparent=True)
plt.close()
#%%


"""seoul weekends"""

#%%


temp_df2 = pd.read_csv(r'seoul_weekends_real_pred_prop_median_comparison_demand.csv')

ext_station_seoul_ends = temp_df2.loc[temp_df2.real>=40].station.unique()

#%%


fig1, axes1 = plt.subplots(1,1, figsize=(10,8))

temp_df2.loc[temp_df2.station.isin(
ext_station_seoul_ends)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='b')

temp_df2.loc[~temp_df2.station.isin(
ext_station_seoul_ends)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='r', )

axes1.set_xlabel('Date', fontsize='xx-large')
axes1.set_ylabel('Daily Average Demand', fontsize='xx-large')
axes1.tick_params(axis='both', labelsize='x-large')

plt.legend(['Station with extreme events (42)','Station without extreme events (1452)'], fontsize='large', loc='upper left')

plt.savefig('seoul_weekends_daily_demand.png', dpi=1024, transparent=True)
plt.close()


#%%


"""daejeon"""


#%%

temp_df1 = pd.read_csv(r'daejeon_weekdays_real_pred_prop_median_comparison_demand.csv')

ext_station_daejeon = temp_df1.loc[temp_df1.real>=10].station.unique()
ext_date_daejeon = temp_df1.loc[temp_df1.real>=10].date.unique()

#%%
fig1, axes1 = plt.subplots(1,1, figsize=(10,8))

temp_df1.loc[temp_df1.station.isin(ext_station_daejeon)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='b')

temp_df1.loc[~temp_df1.station.isin(ext_station_daejeon)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='r', )

axes1.set_xlabel('Date', fontsize='xx-large')
axes1.set_ylabel('Daily Average Demand', fontsize='xx-large')
axes1.tick_params(axis='both', labelsize='x-large')

plt.legend(['Station with extreme events (27)','Station without extreme events (195)'], fontsize='large', loc='upper left')

plt.savefig('daejeon_weekdays_daily_demand.png', dpi=1024, transparent=True)
plt.close()

#%%

"""daejeon weekends"""


#%%

temp_df3 = pd.read_csv(r'daejeon_weekends_real_pred_prop_median_comparison_demand.csv')

ext_station_daejeon_ends = temp_df3.loc[temp_df3.real>=10].station.unique()

#%%
fig1, axes1 = plt.subplots(1,1, figsize=(10,8))

temp_df3.loc[temp_df3.station.isin(ext_station_daejeon_ends)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='b')

temp_df3.loc[~temp_df3.station.isin(ext_station_daejeon_ends)].groupby('date').mean().drop(columns=['station','hour'])['real'].plot(ax=axes1,rot=45, linewidth=2.0, c='r', )

axes1.set_xlabel('Date', fontsize='xx-large')
axes1.set_ylabel('Daily Average Demand', fontsize='xx-large')
axes1.tick_params(axis='both', labelsize='x-large')

plt.legend(['Station with extreme events (22)','Station without extreme events (200)'], fontsize='large', loc='upper left')



plt.savefig('daejeon_weekends_daily_demand.png', dpi=1024, transparent=True)
plt.close()










