# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:30:23 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp_df = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')

temp_df

#%%

fig1, axes1 = plt.subplots(1,1, figsize=(8,8))
temp_df.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).plot(rot=45, ax=axes1, linewidth=3.0)

axes1.set_xlabel('Date', fontsize='xx-large')
axes1.set_ylabel('Daily Aggregated Demand', fontsize='xx-large')
axes1.tick_params(axis='both', labelsize='x-large')

plt.legend(['Real','AdaBoost_RDT','AdaBoost','AdaBoost_RT','GBM','XGBoost','MBoost','LADBoost'], fontsize='large', loc='upper left')

#%%

temp_df01 = pd.DataFrame(data=(temp_df.groupby(['date']).sum()['real'].values.reshape(-1,1)-temp_df.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).values) / (temp_df.groupby(['date']).sum()['real'].values.reshape(-1,1)+1e-10), index=temp_df.groupby(['date']).sum().index,columns=temp_df.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).columns)


sd_ext_date = temp_df.loc[temp_df.real>=40].date.unique()
#%%
fig1, axes1 = plt.subplots(1,1, figsize=(12,12))

bp1=axes1.boxplot(temp_df01.loc[sd_ext_date].drop(columns='real').values,showfliers=False, positions=[0.35,1.35,2.35, 3.35, 4.35, 5.35, 6.35], widths=0.25)

bp2=axes1.boxplot(temp_df01.loc[np.setdiff1d(temp_df.date.unique(), sd_ext_date)].drop(columns='real').values, positions=[0.65,1.65,2.65, 3.65, 4.65, 5.65, 6.65,], widths=0.25, showfliers=False)

axes1.set_xticks([0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5])
axes1.set_yticks([-4,-3,-2,-1,0,1])

axes1.set_xticklabels(['AdaBoost_RDT', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost'], fontsize='xx-large', rotation=45)
axes1.tick_params(axis='both', labelsize='xx-large')

plt.setp(bp1['boxes'], color='b', linewidth=2.5)
plt.setp(bp2['boxes'], color='R', linewidth=2.5)

plt.setp(bp1['whiskers'], color='b', linewidth=2.5)
plt.setp(bp2['whiskers'], color='R', linewidth=2.5)


plt.setp(bp1['caps'], color='b', linewidth=2.5)
plt.setp(bp2['caps'], color='R', linewidth=2.5)

plt.setp(bp1['medians'], color='b', linewidth=2.5)
plt.setp(bp2['medians'], color='R', linewidth=2.5)

axes1.set_ylabel('Distribution of Residual Ratio', fontsize='xx-large')
axes1.set_xlabel('Model', fontsize='x-large')

axes1.plot([], c='b', label='Date with extreme events (150)', linewidth=2.5)
axes1.plot([], c='r', label='Date without extreme events (97)', linewidth=2.5)
axes1.legend(fontsize='xx-large')

plt.savefig('seoul_weekdays_daily_residual_ratio.png', dpi=1024, transparent=True, tightlayout=True)
plt.close()

#sum: 4.38
#mean 2.83
#%%

"""Seoul Weekends"""

#%%

temp_df1 = pd.read_csv(r'seoul_weekends_real_pred_prop_median_comparison_demand.csv')
#%%

fig1, axes1 = plt.subplots(1,1, figsize=(8,8))
temp_df1.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).plot(rot=45, ax=axes1, linewidth=3.0)

axes1.set_xlabel('Date', fontsize='xx-large')
axes1.set_ylabel('Daily Aggregated Demand', fontsize='xx-large')
axes1.tick_params(axis='both', labelsize='x-large')

#plt.legend(['Real','AdaBoost_RDT','AdaBoost','AdaBoost_RT','GBM','XGBoost','MBoost','LADBoost'], fontsize='large', loc='upper left')


#%%


temp_df1.groupby(['station']).sum().drop(columns=['hour','median_pred'])


#%%


temp_df11 = pd.DataFrame(data=(temp_df1.groupby(['date']).sum()['real'].values.reshape(-1,1)-temp_df1.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).values) / (temp_df1.groupby(['date']).sum()['real'].values.reshape(-1,1)+1e-10), index=temp_df1.groupby(['date']).sum().index,columns=temp_df1.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).columns)

se_ext_date = temp_df1.loc[temp_df1.real>=40].date.unique()


#%%

fig1, axes1 = plt.subplots(1,1, figsize=(12,12))

bp1=axes1.boxplot(temp_df11.loc[se_ext_date].drop(columns='real').values,showfliers=False, positions=[0.35,1.35,2.35, 3.35, 4.35, 5.35, 6.35], widths=0.25)

bp2=axes1.boxplot(temp_df11.loc[np.setdiff1d(temp_df1.date.unique(), se_ext_date)].drop(columns='real').values, positions=[0.65,1.65,2.65, 3.65, 4.65, 5.65, 6.65,], widths=0.25, showfliers=False)


axes1.set_xticks([0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5])
axes1.set_xticklabels(['AdaBoost_RDT', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost'], fontsize='xx-large', rotation=45)
axes1.tick_params(axis='both', labelsize='xx-large')

plt.setp(bp1['boxes'], color='b', linewidth=2.5)
plt.setp(bp2['boxes'], color='R', linewidth=2.5)

plt.setp(bp1['whiskers'], color='b', linewidth=2.5)
plt.setp(bp2['whiskers'], color='R', linewidth=2.5)


plt.setp(bp1['caps'], color='b', linewidth=2.5)
plt.setp(bp2['caps'], color='R', linewidth=2.5)

plt.setp(bp1['medians'], color='b', linewidth=2.5)
plt.setp(bp2['medians'], color='R', linewidth=2.5)

axes1.set_ylabel('Distribution of Residual Ratio', fontsize='xx-large')
axes1.set_xlabel('Model', fontsize='xx-large')

axes1.plot([], c='b', label='Date with extreme events (66)', linewidth=2.5)
axes1.plot([], c='r', label='Date without extreme events (52)', linewidth=2.5)
axes1.legend(fontsize='xx-large')

plt.savefig('seoul_weekends_daily_residual_ratio.png', dpi=1024, transparent=True, tightlayout=True)
plt.close()


#sum: 4.29
#mean: 3.38
#%%

"""daejeon weekdays"""

#%%

temp_df2 = pd.read_csv(r'daejeon_weekdays_real_pred_prop_median_comparison_demand.csv')

temp_df21 = pd.DataFrame(data=(temp_df2.groupby(['date']).sum()['real'].values.reshape(-1,1)-temp_df2.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).values) / (temp_df2.groupby(['date']).sum()['real'].values.reshape(-1,1)+1e-10), index=temp_df2.groupby(['date']).sum().index,columns=temp_df2.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).columns)


dd_ext_date = temp_df2.loc[temp_df2.real>=10].date.unique()

#%%
fig1, axes1 = plt.subplots(1,1, figsize=(12,12))

bp1=axes1.boxplot(temp_df21.loc[dd_ext_date].drop(columns='real').values,showfliers=False, positions=[0.35,1.35,2.35, 3.35, 4.35, 5.35, 6.35], widths=0.25)

bp2=axes1.boxplot(temp_df21.loc[np.setdiff1d(temp_df2.date.unique(), dd_ext_date)].drop(columns='real').values, positions=[0.65,1.65,2.65, 3.65, 4.65, 5.65, 6.65,], widths=0.25, showfliers=False)

axes1.set_xticks([0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5])
#axes1.set_yticks([-4,-3,-2,-1,0,1])

axes1.set_xticklabels(['AdaBoost_RDT', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost'], fontsize='xx-large', rotation=45)
axes1.tick_params(axis='both', labelsize='xx-large')

plt.setp(bp1['boxes'], color='b', linewidth=2.5)
plt.setp(bp2['boxes'], color='R', linewidth=2.5)

plt.setp(bp1['whiskers'], color='b', linewidth=2.5)
plt.setp(bp2['whiskers'], color='R', linewidth=2.5)


plt.setp(bp1['caps'], color='b', linewidth=2.5)
plt.setp(bp2['caps'], color='R', linewidth=2.5)

plt.setp(bp1['medians'], color='b', linewidth=2.5)
plt.setp(bp2['medians'], color='R', linewidth=2.5)

axes1.set_ylabel('Distribution of Residual Ratio', fontsize='xx-large')
axes1.set_xlabel('Model', fontsize='xx-large')

axes1.plot([], c='b', label='Date with extreme events (106)', linewidth=2.5)
axes1.plot([], c='r', label='Date without extreme events (141)', linewidth=2.5)
axes1.legend(fontsize='xx-large')

plt.savefig('daejeon_weekdays_daily_residual_ratio.png', dpi=1024, transparent=True, tightlayout=True)
plt.close()


#sum: 1.68
#mean: 2.23

#%%

"""daejeon weekends"""

#%%

temp_df3 = pd.read_csv(r'daejeon_weekends_real_pred_prop_median_comparison_demand.csv')

temp_df31 = pd.DataFrame(data=(temp_df3.groupby(['date']).sum()['real'].values.reshape(-1,1)-temp_df3.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).values) / (temp_df3.groupby(['date']).sum()['real'].values.reshape(-1,1)+1e-10), index=temp_df3.groupby(['date']).sum().index,columns=temp_df3.groupby(['date']).sum().drop(columns=['station','hour','median_pred']).columns)


de_ext_date = temp_df3.loc[temp_df3.real>=10].date.unique()

#%%


fig1, axes1 = plt.subplots(1,1, figsize=(12,12))

bp1=axes1.boxplot(temp_df31.loc[de_ext_date].drop(columns='real').values,showfliers=False, positions=[0.35,1.35,2.35, 3.35, 4.35, 5.35, 6.35], widths=0.25)

bp2=axes1.boxplot(temp_df31.loc[np.setdiff1d(temp_df3.date.unique(), de_ext_date)].drop(columns='real').values, positions=[0.65,1.65,2.65, 3.65, 4.65, 5.65, 6.65,], widths=0.25, showfliers=False)

axes1.set_xticks([0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5])
#axes1.set_yticks([-4,-3,-2,-1,0,1])

axes1.set_xticklabels(['AdaBoost_RDT', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost'], fontsize='xx-large', rotation=45)
axes1.tick_params(axis='both', labelsize='xx-large')

plt.setp(bp1['boxes'], color='b', linewidth=2.5)
plt.setp(bp2['boxes'], color='R', linewidth=2.5)

plt.setp(bp1['whiskers'], color='b', linewidth=2.5)
plt.setp(bp2['whiskers'], color='R', linewidth=2.5)


plt.setp(bp1['caps'], color='b', linewidth=2.5)
plt.setp(bp2['caps'], color='R', linewidth=2.5)

plt.setp(bp1['medians'], color='b', linewidth=2.5)
plt.setp(bp2['medians'], color='R', linewidth=2.5)

axes1.set_ylabel('Distribution of Residual Ratio', fontsize='xx-large')
axes1.set_xlabel('Model', fontsize='xx-large')

axes1.plot([], c='b', label='Date with extreme events (74)', linewidth=2.5)
axes1.plot([], c='r', label='Date without extreme events (44)', linewidth=2.5)
axes1.legend(fontsize='xx-large')

plt.savefig('daejeon_weekends_daily_residual_ratio.png', dpi=1024, transparent=True, tightlayout=True)
plt.close()

#sum: 6.36
#mean: 3.78
#%%


def get_real_pred_zero_ratio(whole_df):
    new_df = whole_df.loc[whole_df.real==0]
    new_li = []
    for xx in ['pred',  'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']:
        new_li.append((new_df[xx] == 0).sum() / new_df.shape[0])
    return new_li

#%%
pd.DataFrame(data={'seoul_weekdays':get_real_pred_zero_ratio(temp_df),
 'seoul_weekends':get_real_pred_zero_ratio(temp_df1),
 'daejeon_weekdays':get_real_pred_zero_ratio(temp_df2),
 'daejeon_weekends':get_real_pred_zero_ratio(temp_df3)}, index=['AdaBoost_RDT',  'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']).to_csv(r'real_pred_zero_ratio_seoul_daejeon_weekdays_weekends.csv')


#%%
import user_utils as uu




_, trn_y, _, _ = uu.load_gpickle(r'preprocess/daejeon/preprocessed_dataset_0317_weekdays.pickle')



trn_daejeon_weekdays_df = pd.read_csv(r'preprocess/daejeon/index_daejeon_trn_weekdays.csv')
trn_daejeon_weekdays_df['real'] = trn_y

np.sort(trn_daejeon_weekdays_df.loc[trn_daejeon_weekdays_df.real>=10].station.unique())

trn_daejeon_weekdays_df.station.nunique()

temp_df2.station.nunique()

np.sort(temp_df2.loc[temp_df2.real>=10].station.unique())



























