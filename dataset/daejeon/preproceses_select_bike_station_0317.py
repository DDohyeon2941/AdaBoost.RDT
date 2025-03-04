# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:59:46 2023

@author: dohyeon
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp_pkl = pd.read_csv(r'2018.csv')
temp_pkl1 = pd.read_csv(r'2019v2.csv')
#%%
temp_pkl = temp_pkl.dropna().reset_index(drop=True)
temp_pkl.columns = ['rent_id', 'rent_time', 'return_id', 'return_time', 'travel_distance', 'fare']
#%%
temp_pkl.loc[:,'rent_time'] = pd.to_datetime(temp_pkl['rent_time'], format='%Y%m%d%H%M%S')
temp_pkl.loc[:,'return_time'] = pd.to_datetime(temp_pkl['return_time'], format='%Y%m%d%H%M%S')


#%%

temp_pkl1.loc[:, 'date'] = pd.to_datetime(temp_pkl1.rent_time)
temp_pkl1.loc[:, 'date1'] = [xx.floor('1H') for xx in temp_pkl1.date]
temp_pkl1.loc[:, 'date2'] = [xx.floor('1D') for xx in temp_pkl1.date]
temp_pkl1.loc[:, 'return_time'] = pd.to_datetime(temp_pkl1.return_time)
temp_pkl1.loc[:, 'duration'] = [xx-yy for xx, yy in temp_pkl1[['return_time','date']].values]
temp_pkl1.loc[:, 'duration'] = [xx.total_seconds() for xx in temp_pkl1['duration']]
temp_pkl1=temp_pkl1.loc[np.where((temp_pkl1['duration'].values>=180)&(temp_pkl1['duration'].values<=3600*4))[0]].reset_index(drop=True)



#%%

temp_pkl = temp_pkl.astype({'rent_id':'int', 'return_id':'int'})
temp_pkl.loc[:, 'date'] = pd.to_datetime(temp_pkl.rent_time)
temp_pkl.loc[:, 'day'] = [xx.day for xx in temp_pkl.date]
temp_pkl.loc[:, 'hour'] = [xx.hour for xx in temp_pkl.date]
temp_pkl.loc[:, 'month'] = [xx.month for xx in temp_pkl.date]
temp_pkl.loc[:, 'date1'] = [xx.floor('1H') for xx in temp_pkl.date]
temp_pkl.loc[:, 'date2'] = [xx.floor('1D') for xx in temp_pkl.date]
temp_pkl.loc[:, 'return_time'] = pd.to_datetime(temp_pkl.return_time)
temp_pkl.loc[:, 'duration'] = [xx-yy for xx, yy in temp_pkl[['return_time','date']].values]
temp_pkl.loc[:, 'duration'] = [xx.total_seconds() for xx in temp_pkl['duration']]


temp_pkl=temp_pkl.loc[np.where((temp_pkl['duration'].values>=180)&(temp_pkl['duration'].values<=3600*4))[0]].reset_index(drop=True)


#%%

#temp_pkl.groupby(['month','day','hour', 'rent_id']).count()['rent_id'].plot()



tt1= temp_pkl.groupby(['date2', 'rent_id']).count()['month'].unstack().stack().unstack()
tt2 = temp_pkl1.groupby(['date2', 'rent_id']).count()['fare'].unstack().stack().unstack()

#%%
tt11 = 364-tt1.isnull().sum(axis=0)
tt21 = 365-tt2.isnull().sum(axis=0)

trn_sel_ind = tt11[tt11>=250].index.values
tst_sel_ind = tt21[tt21>=250].index.values

#%%
(temp_pkl.loc[temp_pkl.rent_id.isin(trn_sel_ind )]).groupby('date1').count()['rent_id'].plot(figsize=(20,6))
(temp_pkl1.loc[temp_pkl1.rent_id.isin(tst_sel_ind )]).groupby('date1').count()['rent_id'].plot(figsize=(20,6))


#%%

"""아래 두 그래프를 보여드리고, 타슈의 경우 대상 정류소를 선정하는 과정에서 운영일수에 대한 임계값을 250일로 줄이기로 함"""

#%% 연도별, 대여소별 운영일수 그래프

plt.plot(tt11.sort_values().values, label='2018'), plt.plot(tt21.sort_values().values, label='2019'), plt.legend()


#%%  연도별, 대여소별 일평균 대여량 그래프

plt.plot(tt1.mean(axis=0).sort_values().values, label='2018'), plt.plot(tt2.mean(axis=0).sort_values().values, label='2019'), plt.legend()


#%%


temp_pkl = temp_pkl.loc[(temp_pkl.rent_id.isin(trn_sel_ind))&(temp_pkl.return_id.isin(trn_sel_ind))].reset_index(drop=True)

temp_pkl1 = temp_pkl1.loc[(temp_pkl1.rent_id.isin(tst_sel_ind))&(temp_pkl1.return_id.isin(tst_sel_ind))].reset_index(drop=True)


temp_pkl.to_pickle(r'sel_demand_2018.pkl')
temp_pkl1.to_pickle(r'sel_demand_2019.pkl')


#%%
def get_stat_date(aaa):
    bin_list = []
    for uni_id, uni_df in aaa.groupby(['rent_id']):
        uni_row = [uni_id, uni_df['date2'].values[0], uni_df['date2'].values[-1]]
        bin_list.append(uni_row)
    return bin_list

#%%

pd.DataFrame(data=get_stat_date(temp_pkl)).to_csv(r'station_info_2018.csv', index=False)
pd.DataFrame(data=get_stat_date(temp_pkl1)).to_csv(r'station_info_2019.csv', index=False)

#%%

poi_pop_500 = pd.read_csv(r'geo/geo_500m.csv')

poi_pop_500.ID.unique()


trn_sel_ind.shape
tst_sel_ind.shape


np.intersect1d(poi_pop_500.ID.unique(), trn_sel_ind).shape
np.intersect1d(poi_pop_500.ID.unique(), tst_sel_ind).shape






