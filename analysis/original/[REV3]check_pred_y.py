# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:31:34 2024

@author: dohyeon
"""


import pandas as pd

import numpy as np
import user_utils as uu
import matplotlib.pyplot as plt

#temp_df = pd.read_csv(r'G:\dock\resubmit\bootstrapped\proposed\seoul\weekdays\y_val_1.csv')
X_trn , y_trn , X_tst , y_tst = uu.load_gpickle(r'preprocess/preprocessed_dataset_0317_weekdays.pickle')

X_trn[:,21:]

temp_df1 = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')

used_cols = ['food','cafe','tour','bank','num_of_sta','uni_dist','subway','bus',
             'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','sun','temp','wind','log_rain2','humidity']

temp_df1.columns

temp_df1[['real','pred', 'median_pred']]


temp_df1

gr1_df = (temp_df1.loc[temp_df1.real<4])[['real','pred', 'median_pred']]
zero_df = (temp_df1.loc[temp_df1.real==0])[['real','pred', 'median_pred']]
one_df = (temp_df1.loc[temp_df1.real==1])[['real','pred', 'median_pred']]
ext_df = (temp_df1.loc[temp_df1.real>=40])[['real','pred', 'median_pred']]


non_ext_df = ext_df.loc[ext_df.pred<3]
acc_ext_df = ext_df.loc[ext_df.pred>=20]


trn_gr1_df = X_trn[np.where(y_trn<4)[0],:]

trn_ext_df = X_trn[np.where(y_trn>=40)[0],:]


#%%


acc_zero_df = X_tst[zero_df.loc[zero_df['pred']==0].index.values,:]



non_zero_df = X_tst[zero_df.loc[zero_df['pred']>0].index.values,:]
#%%
sel_num = 5
#plt.boxplot({1:X_tst[gr1_df.index.values, sel_num],2:X_tst[ext_df.index.values, sel_num]}.values(), showfliers=False)


plt.boxplot({1:acc_zero_df[:, sel_num], 2:non_zero_df[:, sel_num], 3:X_tst[ext_df.index.values, sel_num], 4:X_trn[:, sel_num]}.values(), showfliers=False)
plt.xticks([1,2,3,4], labels=['acc_zero', 'over_zero', 'ext_tst','ext_trn'])

#%% 2024-09-13
sel_num = 4

plt.boxplot({1:X_tst[acc_ext_df.index.values, sel_num], 2:X_tst[non_ext_df.index.values, sel_num], 3:X_tst[ext_df.index.values, sel_num], 4:trn_ext_df[:, sel_num], 5:trn_gr1_df[:, sel_num]}.values(), showfliers=False)

plt.xticks([1,2,3,4,5], labels=['acc_ext', 'under_ext', 'ext_tst','ext_trn', 'gr1_trn'])



#%% 2024-09-13

temp_df1.loc[(temp_df1.real>=40)&(temp_df1.pred<4)].station.value_counts()


trn_low_ext_df = X_trn[temp_df2.loc[temp_df2.ID.isin([2701, 1243])].index.values,:]

tst_low_ext_df = X_tst[temp_df1.loc[temp_df1.station.isin([2701, 1243])].index.values,:]

#%%

sel_num = 21
plt.boxplot({1: tst_low_ext_df[:,sel_num], 2:trn_gr1_df[:, sel_num],3:trn_ext_df[:, sel_num], 4:X_tst[acc_ext_df.index.values, sel_num]}.values(), showfliers=False)

plt.xticks([1,2,3,4], labels=['tst_low_ext', 'trn_gr1', 'trn_ext','tst_acc_ext'])
plt.title('%s'%(used_cols[sel_num]))

#%%
"""1:winter, 2:spring, 3: summer, 4:fall

    gr1 = [7, 8, 9]
    gr2 = [10, 11, 12, 13, 14, 15, 16]
    gr3 = [17, 18, 19]
    gr4 = [20, 21, 22, 23]
    gr5 = [0, 1, 2, 3, 4, 5, 6]
"""
sel_num = 29

plt.boxplot({1: tst_low_ext_df[:,sel_num], 2:trn_gr1_df[:, sel_num],3:trn_ext_df[:, sel_num], 4:X_tst[acc_ext_df.index.values, sel_num]}.values(), showfliers=False)

plt.xticks([1,2,3,4], labels=['tst_low_ext', 'trn_gr1', 'trn_ext','tst_acc_ext'])






#%%

temp_df2 = pd.read_csv(r'preprocess/seoul_train_weekdays_date_id_y.csv')

#%%



temp_df1.loc[temp_df1.real>=40][['real','pred','median_pred']].plot()


temp_df1.loc[temp_df1.station==419][['real','pred','median_pred']].plot()
temp_df1.loc[temp_df1.station==207][['real','pred','median_pred']].plot()

temp_df1.loc[temp_df1.station==113][['real','pred','median_pred']].plot()

temp_df1.loc[temp_df1.station==1222][['real','pred','median_pred','AdaBoost']].plot()
