# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:26:35 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np

def indexing_no_extreme(whole_df, index_df):
    new_result_df = pd.DataFrame()
    for random_state in index_df.columns:
        extreme_idx = np.where(whole_df.y.values>=10)[0]
        new_result_df[random_state] = np.sort(np.setdiff1d(index_df[random_state].values, extreme_idx))
    return new_result_df

def indexing_over_extreme(whole_df, index_df):
    new_result_df = pd.DataFrame()
    for random_state in index_df.columns:
        extreme_idx = np.where(whole_df.y.values>=10)[0]
        new_result_df[random_state] = np.sort(np.union1d(index_df[random_state].values, extreme_idx))
    return new_result_df


if __name__ == "__main__":
    trn_df = pd.read_csv(r'2018_train_daejeon_weekdays_20240418.csv')
    trn_df1 = pd.read_csv(r'2018_train_daejeon_weekends_20240418.csv')

    idx_df = pd.read_csv(r'bootstrap_index_daejeon_weekdays_20240424.csv')
    idx_df1 = pd.read_csv(r'bootstrap_index_daejeon_weekends_20240424.csv')


    indexing_no_extreme(trn_df, idx_df).to_csv(r'bootstrap_index_daejeon_weekdays_no_extreme_20240501.csv', index=False)
    indexing_no_extreme(trn_df1, idx_df1).to_csv(r'bootstrap_index_daejeon_weekends_no_extreme_20240501.csv', index=False)



    indexing_over_extreme(trn_df, idx_df).to_csv(r'bootstrap_index_daejeon_weekdays_over_extreme_20240501.csv', index=False)
    indexing_over_extreme(trn_df1, idx_df1).to_csv(r'bootstrap_index_daejeon_weekends_over_extreme_20240501.csv', index=False)
