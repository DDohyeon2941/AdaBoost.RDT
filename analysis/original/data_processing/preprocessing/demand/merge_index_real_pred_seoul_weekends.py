# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:19:53 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np
import os
import user_utils as uu


if __name__ == "__main__":
    dataset_path = r'preprocess/preprocessed_dataset_0317_holy.pickle'
    _ , _ , _ , tst_y = uu.load_gpickle(dataset_path)
    holy_obj = uu.load_gpickle(r'results/results/weekends/seoul/holy_50_4_0317_step23_3.0_positive_12_False_Ridge.pickle')['pred_y']
    comparison_obj = uu.load_gpickle(r'[TAB 6.]seoul_comparison_week_holy_optimal_pred_y_revision1_0317.pickle')['weekends']
    index_df = pd.read_csv(r'preprocess/index_seoul_tst_weekends.csv')

    index_df['real'] = np.log(tst_y+1)
    index_df['pred'] = holy_obj


    index_df['AdaBoost'] = comparison_obj['AdaBoost']
    index_df['AdaBoost_RT'] = comparison_obj['AdaBoost_RT']
    index_df['GBM'] = comparison_obj['L2Loss']
    index_df['XGBoost'] = comparison_obj['xgboost']
    index_df['MBoost'] = comparison_obj['mboost']
    index_df['LADBoost'] = comparison_obj['l1loss']


    index_df['AdaBoost'].loc[index_df['AdaBoost']<0] = 0
    index_df['AdaBoost_RT'].loc[index_df['AdaBoost_RT']<0] = 0
    index_df['GBM'].loc[index_df['GBM']<0] = 0
    index_df['XGBoost'].loc[index_df['XGBoost']<0] = 0
    index_df['MBoost'].loc[index_df['MBoost']<0] = 0
    index_df['LADBoost'].loc[index_df['LADBoost']<0] = 0

    index_df.to_csv(r'seoul_real_pred_weekends.csv', index=False)



