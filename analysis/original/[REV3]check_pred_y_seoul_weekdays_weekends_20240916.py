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


temp_df1 = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')

temp_df2 = pd.read_csv(r'preprocess/seoul_train_weekdays_date_id_y.csv')

used_cols = ['food','cafe','tour','bank','num_of_sta','uni_dist','subway','bus',
             'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','sun','temp','wind','log_rain2','humidity']


tst_ext_df = (temp_df1.loc[temp_df1.real>=40])[['real','pred', 'median_pred']]

tst_low_ext_df = tst_ext_df.loc[tst_ext_df.pred<4]
tst_high_ext_df = tst_ext_df.loc[tst_ext_df.pred>=20]


trn_ext_df = X_trn[np.where(y_trn>=40)[0],:]
#%%

sel_num = 20

plt.boxplot({1: X_tst[tst_low_ext_df.index.values, sel_num], 2: X_tst[tst_high_ext_df.index.values, sel_num],3:trn_ext_df[:, sel_num]}.values(), showfliers=False)
plt.xticks([1,2,3], labels=['tst_low_ext', 'tst_high_ext', 'trn_ext'])
plt.title('%s'%(used_cols[sel_num]))




#%%

"""weekends"""

#%%


X_trn , y_trn , X_tst , y_tst = uu.load_gpickle(r'preprocess/preprocessed_dataset_0317_holy.pickle')


temp_df1 = pd.read_csv(r'seoul_weekends_real_pred_prop_median_comparison_demand.csv')

temp_df2 = pd.read_csv(r'preprocess/seoul_train_weekends_date_id_y.csv')


#%%

tst_ext_df = (temp_df1.loc[temp_df1.real>=40])[['real','pred', 'median_pred']]

tst_low_ext_df = tst_ext_df.loc[tst_ext_df.pred<4]
tst_high_ext_df = tst_ext_df.loc[tst_ext_df.pred>=20]


trn_ext_df = X_trn[np.where(y_trn>=40)[0],:]
#%%

sel_num = 20

plt.boxplot({1: X_tst[tst_low_ext_df.index.values, sel_num], 2: X_tst[tst_high_ext_df.index.values, sel_num],3:trn_ext_df[:, sel_num]}.values(), showfliers=False)
plt.xticks([1,2,3], labels=['tst_low_ext', 'tst_high_ext', 'trn_ext'])
plt.title('%s'%(used_cols[sel_num]))




