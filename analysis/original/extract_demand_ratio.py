# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:21:56 2024

@author: dohyeon
"""

import pandas as pd

station_df = pd.read_csv(r'preprocess\dateinfo_0317.csv')
temp_df = pd.read_pickle(r'preprocess\2018_sel_demands_0317.pkl')

temp_df['rr_time'] = temp_df['return_time'].dt.floor('1h')

temp_df['rr_date'] = temp_df['return_time'].dt.date

temp_df.loc[temp_df.rent_id.isin(station_df['0'].dropna().astype(int).values)].groupby(['r_time','rent_id']).count()['rent_time']
#%%


station_df = pd.read_csv(r'preprocess\dateinfo_0317.csv')
temp_df = pd.read_pickle(r'preprocess\2019_sel_demands_0317.pkl')

temp_df['rr_time'] = temp_df['return_time'].dt.floor('1h')

temp_df['rr_date'] = temp_df['return_time'].dt.date


#%%
gb_rent_df = temp_df.loc[temp_df.rent_id.isin(station_df['0'].dropna().astype(int).values)].groupby(['r_date','rent_id']).count()['rent_time'].unstack()


gb_return_df = temp_df.loc[temp_df.return_id.isin(station_df['0'].dropna().astype(int).values)].groupby(['rr_date','return_id']).count()['rent_time'].unstack().iloc[:-1]

rent_ratio_df = gb_rent_df.fillna(0) / (gb_rent_df.fillna(0) + gb_return_df.fillna(0))
#%%

gb_rent_df = temp_df.loc[temp_df.rent_id.isin(station_df['1'].values)].groupby(['r_date','rent_id']).count()['rent_time'].unstack()


gb_return_df = temp_df.loc[temp_df.return_id.isin(station_df['1'].values)].groupby(['rr_date','return_id']).count()['rent_time'].unstack().iloc[:-1]

rent_ratio_df = gb_rent_df / (gb_rent_df + gb_return_df)


#%%

gb_rent_df1 = temp_df.loc[temp_df.rent_id.isin(station_df['1'].values)].groupby(['r_time','rent_id']).count()['rent_time'].unstack()


gb_rent_df1


gb_return_df1 = temp_df.loc[temp_df.return_id.isin(station_df['1'].values)].groupby(['rr_time','return_id']).count()['rent_time'].unstack().iloc[:-3]

rent_ratio_df1 = gb_rent_df1.fillna(0) / (gb_rent_df1.fillna(0) + gb_return_df1.fillna(0))

rent_ratio_df1.index = rent_ratio_df1.index.floor('1h')

#%%
import numpy as np
demand_df = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')
demand_df1 = pd.read_csv(r'seoul_weekends_real_pred_prop_median_comparison_demand.csv')
np.sort(demand_df.loc[demand_df.real>=40].station.unique())

pd.to_datetime(demand_df.date.unique())
#%%

rent_ratio_df[np.sort(demand_df.loc[demand_df.real>=40].station.unique())]


demand_df.loc[demand_df.real>=40].date.unique()

#%%
import seaborn as sns

sns.heatmap(rent_ratio_df.loc[pd.to_datetime(demand_df.date.unique()).date][np.sort(demand_df.loc[demand_df.real>=40].station.unique())], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)


sns.heatmap(rent_ratio_df.loc[pd.to_datetime(demand_df.loc[demand_df.real>=40].date.unique()).date][np.sort(demand_df.loc[demand_df.real>=40].station.unique())], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)



sns.heatmap(rent_ratio_df.loc[pd.to_datetime(demand_df.date.unique()).date][np.setdiff1d(demand_df.station.unique(), np.sort(demand_df.loc[demand_df.real>=40].station.unique()))], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)



sns.heatmap(rent_ratio_df.loc[pd.to_datetime(demand_df.loc[demand_df.real>=40].date.unique()).date][np.setdiff1d(demand_df.station.unique(), np.sort(demand_df.loc[demand_df.real>=40].station.unique()))], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)


#%%
sns.heatmap(rent_ratio_df.loc[pd.to_datetime(demand_df1.date.unique()[demand_df1.date.unique()!='2019-09-07']).date][np.sort(demand_df1.loc[demand_df1.real>=40].station.unique())], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)

sns.heatmap(rent_ratio_df.loc[pd.to_datetime(demand_df1.date.unique()[demand_df1.date.unique()!='2019-09-07']).date][np.setdiff1d(demand_df1.station.unique(), np.sort(demand_df1.loc[demand_df1.real>=40].station.unique()))], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)
#%%

"""daejeon"""

#%%

station_df1 = pd.read_csv(r'tashu\station_info_2019.csv')
temp_df1 = pd.read_pickle(r'tashu\sel_demand_2019.pkl')



temp_df1['r_time'] = pd.to_datetime(temp_df1['rent_time']).dt.floor('1h')
temp_df1['r_date'] = pd.to_datetime(temp_df1['rent_time']).dt.date

temp_df1['rr_time'] = pd.to_datetime(temp_df1['return_time']).dt.floor('1h')
temp_df1['rr_date'] = pd.to_datetime(temp_df1['return_time']).dt.date

#%%


gb_rent_df11 = temp_df1.loc[temp_df1.rent_id.isin(station_df1['0'].values)].groupby(['r_date','rent_id']).count()['rent_time'].unstack()

gb_return_df11 = temp_df1.loc[temp_df1.return_id.isin(station_df1['0'].values)].groupby(['rr_date','return_id']).count()['rent_time'].unstack().iloc[:-1]

rent_ratio_df11 = gb_rent_df11 / (gb_rent_df11 + gb_return_df11)


#%%

demand_df2 = pd.read_csv(r'daejeon_weekdays_real_pred_prop_median_comparison_demand.csv')
demand_df3 = pd.read_csv(r'daejeon_weekends_real_pred_prop_median_comparison_demand.csv')

#%% 대전 주중

sns.heatmap(rent_ratio_df11.loc[pd.to_datetime(demand_df2.loc[demand_df2.real>=10].date.unique()).date][np.sort(demand_df2.loc[demand_df2.real>=10].station.unique())], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)

sns.heatmap(rent_ratio_df11.loc[pd.to_datetime(demand_df2.loc[demand_df2.real>=10].date.unique()).date][np.setdiff1d(demand_df2.station.unique(), np.sort(demand_df2.loc[demand_df2.real>=10].station.unique()))], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)

#%%
sns.heatmap(rent_ratio_df11.loc[pd.to_datetime(demand_df3.loc[demand_df3.real>=10].date.unique()).date][np.sort(demand_df3.loc[demand_df3.real>=10].station.unique())], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)

sns.heatmap(rent_ratio_df11.loc[pd.to_datetime(demand_df3.loc[demand_df3.real>=10].date.unique()).date][np.setdiff1d(demand_df3.station.unique(), np.sort(demand_df3.loc[demand_df3.real>=10].station.unique()))], annot=False, cmap='viridis', vmin=0.5, vmax=1.0)



