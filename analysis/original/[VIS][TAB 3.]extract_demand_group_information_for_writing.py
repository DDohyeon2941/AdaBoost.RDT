# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:43:35 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np
import user_utils as uu


def get_group_index(whole_arr, thr1, thr2):
    gr1 = np.where(whole_arr<thr1)[0]
    gr2 = np.where(
        (whole_arr>=thr1)&(whole_arr<thr2))[0]
    gr3 = np.where(whole_arr>=thr2)[0]
    return gr1, gr2, gr3

def count_ratio_group(idx1, idx2, idx3, iscount=True):
    one_size = idx1.shape[0]
    two_size = idx2.shape[0]
    three_size = idx3.shape[0]
    all_size = one_size + two_size + three_size

    if iscount:
        return np.array([one_size, two_size, three_size, all_size])
    else:
        return np.array([one_size / all_size, two_size / all_size, three_size / all_size])


def count_ratio_group_string(idx1, idx2, idx3):
    one_size = idx1.shape[0]
    two_size = idx2.shape[0]
    three_size = idx3.shape[0]
    all_size = one_size + two_size + three_size

    str1 = str(one_size) + '(%.3f'%((one_size / all_size)*100)+'%)'
    str2 = str(two_size) + '(%.3f'%((two_size / all_size)*100)+'%)'
    str3 = str(three_size) + '(%.3f'%((three_size / all_size)*100)+'%)'

    return [str1, str2, str3, str(all_size)]




#%%
if __name__ == "__main__":
    """서울 먼저"""

    _, trn_y, _, tst_y = uu.load_gpickle(r'preprocess/preprocessed_dataset_0317_weekdays.pickle')


    count_ratio_group(*get_group_index(trn_y, 4, 40), iscount=True)
    np.round(count_ratio_group(*get_group_index(trn_y, 4, 40), iscount=False)*100,3)


    count_ratio_group(*get_group_index(tst_y, 4, 40), iscount=True)
    np.round(count_ratio_group(*get_group_index(tst_y, 4, 40), iscount=False)*100,3)

    #%%


    _, trn_y, _, tst_y = uu.load_gpickle(r'preprocess/preprocessed_dataset_0317_holy.pickle')


    count_ratio_group_string(*get_group_index(trn_y, 4, 40))
    count_ratio_group_string(*get_group_index(tst_y, 4, 40))

    #%%
    """여기서 부터 대전"""

    _, trn_y, _, tst_y = uu.load_gpickle(r'tashu/preprocessed_dataset_0317_weekdays.pickle')


    count_ratio_group_string(*get_group_index(trn_y, 2, 10))
    count_ratio_group_string(*get_group_index(tst_y, 2, 10))


    #%%
    _, trn_y, _, tst_y = uu.load_gpickle(r'tashu/preprocessed_dataset_0317_holy.pickle')


    count_ratio_group_string(*get_group_index(trn_y, 2, 10))
    count_ratio_group_string(*get_group_index(tst_y, 2, 10))







