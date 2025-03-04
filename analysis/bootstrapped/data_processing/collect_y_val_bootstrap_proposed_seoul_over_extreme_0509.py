# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:31:11 2024

@author: dohyeon
"""


import pandas as pd
import itertools

city_name = 'seoul'
day_type = 'weekdays'
avg_thresholds = [0.25, 0.50, 0.75, 1.00]
avg_ratios = [0.50, 0.75]


def collect_y_val(c_random_state):

    global city_name
    global day_type
    global avg_thresholds
    global avg_ratios

    temp_result = pd.DataFrame()
    for ii, (avg_threshold, avg_ratio) in enumerate(itertools.product(avg_thresholds, avg_ratios)):
        temp_y_val = pd.read_csv(r'results/bootstrapped/proposed/%s/%s/y_val_over_extreme_%s_%s_%s.csv'%(city_name, day_type, c_random_state, str(avg_threshold), str(avg_ratio)))
        if ii == 0:
            temp_result = temp_y_val
        else:
            temp_result['%s_%s'%(str(avg_threshold), str(avg_ratio))] = temp_y_val['%s_%s'%(str(avg_threshold), str(avg_ratio))]
        print(ii)
    return temp_result
#%%
if __name__ == "__main__":

    collect_y_val(0).to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_over_extreme_%s.csv'%(city_name, day_type, 0), index=False)
    collect_y_val(1).to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_over_extreme_%s.csv'%(city_name, day_type, 1), index=False)
    collect_y_val(2).to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_over_extreme_%s.csv'%(city_name, day_type, 2), index=False)
    collect_y_val(3).to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_over_extreme_%s.csv'%(city_name, day_type, 3), index=False)
    collect_y_val(4).to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_over_extreme_%s.csv'%(city_name, day_type, 4), index=False)
    #%%

