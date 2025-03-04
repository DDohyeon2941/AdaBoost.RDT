# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:57:12 2021

@author: dohyeon
"""

import time
import itertools
import numpy as np
import os

from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import stable_cumsum
import user_utils as uu


"""

실험결과파일 생성 O

1. comparison_week_holy_optimal_pred_y.pickle

"""

def get_model_name(file_name):
    special_name = 'AdaBoost_RT'
    if not special_name in file_name:
        return file_name.split('_')[0]
    else:
        return special_name


def get_model_name_holy(file_name):
    special_name = 'AdaBoost_RT'
    if not special_name in file_name:
        return file_name.split('_')[1]
    else:
        return special_name


#%%
if __name__ == "__main__":


    week_dir = r'optimal_comparison_weekdays/daejeon'
    week_dict = {}
    for file_name1 in os.listdir(week_dir):
        temp_path = uu.opj(week_dir, file_name1)
        temp_pred_y = uu.load_gpickle(temp_path)
        temp_model_name = get_model_name(file_name1)
        week_dict[temp_model_name] = temp_pred_y


    #uu.save_gpickle('daejeon_comparison_week_holy_optimal_pred_y_revision1_0317.pickle', {'weekdays':week_dict})

#%%

    holy_dir = r'optimal_comparison_weekends/daejeon'
    holy_dict = {}
    for file_name1 in os.listdir(holy_dir):
        temp_path = uu.opj(holy_dir, file_name1)
        temp_pred_y = uu.load_gpickle(temp_path)
        temp_model_name = get_model_name_holy(file_name1)
        holy_dict[temp_model_name] = temp_pred_y

    holy_dict

    uu.save_gpickle('daejeon_comparison_week_holy_optimal_pred_y_revision1_0317.pickle', {'weekdays':week_dict, 'weekends':holy_dict})














