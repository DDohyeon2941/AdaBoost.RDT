# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:14:55 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np

def get_new_idxs2(whole_arr, thr1, thr2, thr3):
    gr1 = np.where(whole_arr<thr1)[0]
    gr2 = np.where(
        (whole_arr>=thr1)&(whole_arr<thr2))[0]
    gr3 = np.where(
        (whole_arr>=thr2)&(whole_arr<thr3))[0]

    gr4 = np.where(whole_arr>=thr3)[0]
    return gr1, gr2, gr3, gr4

def sampling_index(rs, index):
    np.random.seed(rs); sampled_index = np.random.choice(index, size=index.shape[0], replace=True)
    return sampled_index

def sampling_grouper_index(rs, index1, index2, index3, index4):
    sampled_idx1 = sampling_index(rs, index1)
    sampled_idx2 = sampling_index(rs, index2)
    sampled_idx3 = sampling_index(rs, index3)
    sampled_idx4 = sampling_index(rs, index4)
    output = np.concatenate([sampled_idx1, sampled_idx2,
                             sampled_idx3, sampled_idx4])
    return np.sort(output)

#[ 0,  6,  8, 22, 23, 27, 39, 42, 46, 52, 56, 64, 65, 68, 75, 90, 95, 96, 97, 98]

def get_bootstrap_index_df(main_df, thr1, thr2, thr3):
    idx1, idx2, idx3, idx4 = get_new_idxs2(main_df.y.values, thr1, thr2, thr3)
    random_states = [ 0,  6,  8, 22, 23, 27, 39, 42, 46, 52, 56, 64, 65, 68, 75, 90, 95, 96, 97, 98]
    index_df = pd.DataFrame()
    for i_idx, i_rs in enumerate(random_states):
        index_df[i_idx] = sampling_grouper_index(i_rs, idx1, idx2, idx3, idx4)
    return index_df
if __name__ == "__main__":

    train_pkl = pd.read_pickle(r'2018_total_0317.pkl')

    trn_df = train_pkl.loc[((train_pkl['isholy'] < 1) & (train_pkl['dow'].isin([0,1,2,3,4])))]
    trn_df1 = train_pkl.loc[((train_pkl['isholy'] >= 1) | (train_pkl['dow'].isin([5,6])))]


    sd_df = get_bootstrap_index_df(trn_df, 4, 10, 40)
    se_df = get_bootstrap_index_df(trn_df1, 4, 10, 40)

    sd_df.to_csv(r'bootstrap_index_seoul_weekdays_20240418.csv', index=False)
    se_df.to_csv(r'bootstrap_index_seoul_weekends_20240418.csv', index=False)
