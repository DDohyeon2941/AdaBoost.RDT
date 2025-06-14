# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:27:06 2024

@author: dohyeon
"""


import user_utils as uu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


weekdays_obj = uu.load_gpickle(uu.get_full_path(r'results//fitted_models//proposed_revision//seoul', 'seoul_weekdays_model8_4_4_3.0_1.0_0.75_0.001.pickle' ))

weekends_obj = uu.load_gpickle(uu.get_full_path(r'results//fitted_models//proposed_revision//seoul', 'seoul_weekends_model8_4_4_3.5_0.75_0.75_0.01.pickle' ))



sd_arr = np.array([np.sum([1 if yy>29 else 0 for yy in xx.dt_obj.tree_.feature]) for xx in weekdays_obj.estimators_])/15

se_arr = np.array([np.sum([1 if yy>29 else 0 for yy in xx.dt_obj.tree_.feature]) for xx in weekends_obj.estimators_])/15



plt.plot(sd_arr, c='b'), plt.plot(se_arr, c='r')


weekdays_obj1 = uu.load_gpickle(uu.get_full_path(r'results//results//seoul', 'seoul_weekdays_model8_4_4_3.0_1.0_0.75_0.001.pickle' ))

weekdays_obj1['tst_cdf_weight']

weekdays_df = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')

weekdays_df[['real','pred']]


weekdays_df.real<4

plt.hist(weekdays_obj1['tst_cdf_weight'][(weekdays_df.real[weekdays_df.real<4] - weekdays_df.pred[weekdays_df.real<4]) < 0], density=True)


plt.hist(weekdays_obj1['tst_cdf_weight'][[weekdays_df.real<4]][(weekdays_df.real[weekdays_df.real<4] - weekdays_df.pred[weekdays_df.real<4]) < 0])


plt.hist(weekdays_obj1['tst_cdf_weight'][[weekdays_df.real>=40]][(weekdays_df.real[weekdays_df.real>=40] - weekdays_df.pred[weekdays_df.real>=40]) < 0])

plt.hist(weekdays_obj1['tst_cdf_weight'][[(weekdays_df.real<40) & (weekdays_df.real>=4)]][(weekdays_df.real[(weekdays_df.real<40) & (weekdays_df.real>=4)] - weekdays_df.pred[(weekdays_df.real<40) & (weekdays_df.real>=4)]) < 0])


(weekdays_df.real[weekdays_df.real<4] - weekdays_df.pred[weekdays_df.real<4]) < 0

weekdays_obj1['tst_cdf_weight'][np.where(weekdays_df.real<4)[0]]


over_idx = np.where((weekdays_df.real - weekdays_df.pred)<0)[0]
under_idx = np.where((weekdays_df.real - weekdays_df.pred)>0)[0]

idx1 = np.where(weekdays_df.real<4)[0]
idx2 = np.where((weekdays_df.real>=4)&(weekdays_df.real<40))[0]
idx3 = np.where(weekdays_df.real>=40)[0]


plt.hist(weekdays_obj1['tst_cdf_weight'][np.intersect1d(over_idx, idx1)])
