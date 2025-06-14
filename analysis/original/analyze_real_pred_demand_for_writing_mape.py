# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:52:30 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sd_df = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')
se_df = pd.read_csv(r'seoul_weekends_real_pred_prop_median_comparison_demand.csv')

dd_df = pd.read_csv(r'daejeon_weekdays_real_pred_prop_median_comparison_demand.csv')
de_df = pd.read_csv(r'daejeon_weekends_real_pred_prop_median_comparison_demand.csv')

#%%

['pred', 'median_pred', 'AdaBoost',
       'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']

def cal_mape(y_true, y_pred):
    return np.abs(y_true - y_pred) / y_true

def get_mape_dict(real_pred_df):
    model_names = ['pred', 'median_pred', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']
    new_dict = {}
    for m_idx, m_name in enumerate(model_names):
        new_dict[m_idx] = cal_mape(real_pred_df['real'].values, real_pred_df[m_name].values)
    return new_dict


def cal_smape(y_true, y_pred):
    return np.abs(y_true - y_pred) / ((y_true+y_pred)/2)

def get_smape_dict(real_pred_df):
    model_names = ['pred', 'median_pred', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']
    new_dict = {}
    for m_idx, m_name in enumerate(model_names):
        new_real_pred_df = real_pred_df.loc[~((real_pred_df['real']==0)&(real_pred_df[m_name]==0))]
        new_dict[m_idx] = cal_smape(new_real_pred_df['real'].values, new_real_pred_df[m_name].values)
    return new_dict

def get_smape0_dict(real_pred_df):
    model_names = ['pred', 'median_pred', 'AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']
    new_dict = {}
    for m_idx, m_name in enumerate(model_names):
        new_dict[m_idx] = cal_smape(real_pred_df['real'].values, real_pred_df[m_name].values)
    return new_dict


cal_mape(sd_df.real[sd_df.real>0], sd_df.pred[sd_df.real>0])


plt.boxplot(get_mape_dict(sd_df.loc[(sd_df.real>0)&(sd_df.real<4)]).values(), showfliers=False, notch=True, showmeans=True)

plt.boxplot(get_mape_dict(se_df.loc[(se_df.real>0)&(se_df.real<4)]).values(), showfliers=False, notch=True, showmeans=True)

plt.boxplot(get_mape_dict(dd_df.loc[(dd_df.real>0)&(dd_df.real<2)]).values(), showfliers=False, notch=True, showmeans=True)


plt.boxplot(get_mape_dict(de_df.loc[(de_df.real>0)&(de_df.real<2)]).values(), showfliers=False, notch=True, showmeans=True)


#%%


sd_df.loc[~((sd_df.real==0)&(sd_df.pred==0))]



plt.boxplot(get_smape_dict(sd_df.loc[sd_df.real<4]).values(), showmeans=True)

plt.boxplot(get_smape0_dict(sd_df.loc[(sd_df.real<4)&(sd_df.real>0)]).values(), showmeans=True, showfliers=True)

#%% seoul

sd_df[['real','pred','median_pred']]

plt.hist((sd_df.iloc[np.intersect1d(np.where(sd_df.real<4)[0], np.where((sd_df.real - sd_df.median_pred) < 0)[0])]).real)

plt.hist((sd_df.iloc[np.intersect1d(np.where(sd_df.real<4)[0], np.where((sd_df.real - sd_df.median_pred) > 0)[0])]).real)

###
plt.hist((se_df.iloc[np.intersect1d(np.where(se_df.real<4)[0], np.where((se_df.real - se_df.median_pred) < 0)[0])]).real)

plt.hist((se_df.iloc[np.intersect1d(np.where(se_df.real<4)[0], np.where((se_df.real - se_df.median_pred) > 0)[0])]).real)


#%% daejeon

plt.hist((dd_df.iloc[np.intersect1d(np.where(dd_df.real<2)[0], np.where((dd_df.real - dd_df.median_pred) < 0)[0])]).real)

plt.hist((dd_df.iloc[np.intersect1d(np.where(dd_df.real<2)[0], np.where((dd_df.real - dd_df.median_pred) > 0)[0])]).real)

###
plt.hist((de_df.iloc[np.intersect1d(np.where(de_df.real<2)[0], np.where((de_df.real - de_df.median_pred) < 0)[0])]).real)

plt.hist((de_df.iloc[np.intersect1d(np.where(de_df.real<2)[0], np.where((de_df.real - de_df.median_pred) > 0)[0])]).real)











