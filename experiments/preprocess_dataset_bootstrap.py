# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:24:22 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def split_x_y(train_df, test_df, sel_cols):
    raw_scaler = StandardScaler()
    raw_scaler.fit(train_df[sel_cols].values)

    train_y = train_df.y.values
    train_x = np.hstack((raw_scaler.transform(train_df[sel_cols].values),
                   np.hstack((pd.get_dummies(train_df.season).values,
                              pd.get_dummies(train_df.tmask).values))))
    test_y = test_df.y.values
    test_x = np.hstack((raw_scaler.transform(test_df[sel_cols].values),
                   np.hstack((pd.get_dummies(test_df.season).values,
                              pd.get_dummies(test_df.tmask).values))))
    return train_x, train_y, test_x, test_y

def split_x_y_main(train_df, test_df, index_df, rs):
    used_cols = ['food','cafe','tour','bank','stat_num','uni_dist','subway','bus',
             'tot_pop','saup','jong','y_ratio','one_ratio','tot_ga','dust','snow','solar','temp','wind','log_rain2','humidity']
    return split_x_y(train_df.iloc[index_df[rs]], test_df, used_cols)

if __name__ == "__main__":

    trn_df = pd.read_csv(r'2018_train_daejeon_weekdays_20240418.csv')
    tst_df = pd.read_csv(r'2019_test_daejeon_weekdays_20240418.csv')
    idx_df = pd.read_csv(r'bootstrap_index_daejeon_weekdays_20240418.csv')

    trn_x, trn_y, tst_x, tst_y = split_x_y_main(trn_df, tst_df, idx_df, str(1))
