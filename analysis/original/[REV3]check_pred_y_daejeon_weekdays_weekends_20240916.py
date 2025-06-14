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
X_trn , y_trn , X_tst , y_tst = uu.load_gpickle(r'tashu/preprocessed_dataset_0317_weekdays.pickle')


temp_df1 = pd.read_csv(r'daejeon_weekdays_real_pred_prop_median_comparison_demand.csv')

temp_df2 = pd.read_csv(r'tashu/index_daejeon_trn_weekdays.csv')

used_cols = ['food','cafe','tour','bank','num_of_sta','uni_dist','subway','bus',
             'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','sun','temp','wind','log_rain2','humidity']



tst_ext_df = (temp_df1.loc[temp_df1.real>=10])[['real','pred', 'median_pred']]

tst_low_ext_df = tst_ext_df.loc[tst_ext_df.pred<2]
tst_high_ext_df = tst_ext_df.loc[tst_ext_df.pred>=5]


trn_ext_df = X_trn[np.where(y_trn>=10)[0],:]
#%%

sel_num = 20
plt.boxplot({1: X_tst[tst_low_ext_df.index.values, sel_num], 2: X_tst[tst_high_ext_df.index.values, sel_num],3:trn_ext_df[:, sel_num]}.values(), showfliers=False)
plt.xticks([1,2,3], labels=['tst_low_ext', 'tst_high_ext', 'trn_ext'])
plt.title('%s'%(used_cols[sel_num]))


#%%

"""weekends"""

#%%


X_trn , y_trn , X_tst , y_tst = uu.load_gpickle(r'tashu/preprocessed_dataset_0317_holy.pickle')

used_cols = ['food','cafe','tour','bank','num_of_sta','uni_dist','subway','bus',
             'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','sun','temp','wind','log_rain2','humidity']
temp_df1 = pd.read_csv(r'daejeon_weekends_real_pred_prop_median_comparison_demand.csv')

temp_df2 = pd.read_csv(r'tashu/index_daejeon_trn_weekends.csv')


#%%

tst_ext_df = (temp_df1.loc[temp_df1.real>=10])[['real','pred', 'median_pred']]

tst_low_ext_df = tst_ext_df.loc[tst_ext_df.pred<2]
tst_high_ext_df = tst_ext_df.loc[tst_ext_df.pred>=5]


trn_ext_df = X_trn[np.where(y_trn>=10)[0],:]
#%%

sel_num = 20
plt.boxplot({1: X_tst[tst_low_ext_df.index.values, sel_num], 2: X_tst[tst_high_ext_df.index.values, sel_num],3:trn_ext_df[:, sel_num]}.values(), showfliers=False)
plt.xticks([1,2,3], labels=['tst_low_ext', 'tst_high_ext', 'trn_ext'])
plt.title('%s'%(used_cols[sel_num]))

