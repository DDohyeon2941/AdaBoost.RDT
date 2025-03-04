# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:34:53 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np

def get_merged_y_val_no_extreme(city_name, day_type, alpha, beta, c_random_state):

    comp_df = pd.read_csv(r'results/bootstrapped/comparison/%s/%s/y_val_no_extreme_%s.csv'%(city_name, day_type, c_random_state))
    prop_df = pd.read_csv(r'results/bootstrapped/proposed/%s/%s/y_val_no_extreme_%s.csv'%(city_name, day_type, c_random_state))
    comp_df['pred'] = prop_df['%s_%s'%(str(alpha), str(beta))]
    return comp_df[['real', 'pred', 'XGBoost', 'MBoost', 'LADBoost', 'GBM', 'AdaBoost', 'AdaBoost_RT']]

def get_merged_y_val_over_extreme(city_name, day_type, alpha, beta, c_random_state):

    comp_df = pd.read_csv(r'results/bootstrapped/comparison/%s/%s/y_val_over_extreme_%s.csv'%(city_name, day_type, c_random_state))
    prop_df = pd.read_csv(r'results/bootstrapped/proposed/%s/%s/y_val_over_extreme_%s.csv'%(city_name, day_type, c_random_state))
    comp_df['pred'] = prop_df['%s_%s'%(str(alpha), str(beta))]
    return comp_df[['real', 'pred', 'XGBoost', 'MBoost', 'LADBoost', 'GBM', 'AdaBoost', 'AdaBoost_RT']]


if __name__ == "__main__":
    city_name = 'seoul'
    day_type = 'weekdays'
    sd_params = [(0.75, 0.50), (1.0, 0.50), (1.0, 0.50), (1.0, 0.50), (1.0, 0.75)]

    #%% seoul, weekdays

    get_merged_y_val_no_extreme(city_name, day_type, sd_params[0][0], sd_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 0), index=False)


    get_merged_y_val_no_extreme(city_name, day_type, sd_params[1][0], sd_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, sd_params[2][0], sd_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, sd_params[3][0], sd_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, sd_params[4][0], sd_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 4), index=False)

    #%% seoul, weekends

    city_name = 'seoul'
    day_type = 'weekends'

    se_params = [(1.0, 0.50), (1.0, 0.50), (0.50, 0.50), (0.50, 0.50), (0.75, 0.75)]

    get_merged_y_val_no_extreme(city_name, day_type, se_params[0][0], se_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 0), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, se_params[1][0], se_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, se_params[2][0], se_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, se_params[3][0], se_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, se_params[4][0], se_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 4), index=False)

    #%% daejeon, weekdays

    city_name = 'daejeon'
    day_type = 'weekdays'

    dd_params = [(1.0, 0.50), (1.0, 0.50), (0.75, 0.50), (1.0, 0.50), (1.0, 0.50)]

    get_merged_y_val_no_extreme(city_name, day_type, dd_params[0][0], dd_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 0), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, dd_params[1][0], dd_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, dd_params[2][0], dd_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, dd_params[3][0], dd_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, dd_params[4][0], dd_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 4), index=False)

    #%% daejeon, weekends


    city_name = 'daejeon'
    day_type = 'weekends'

    de_params = [(1.0, 0.50), (1.0, 0.50), (1.0, 0.50), (1.0, 0.50), (1.0, 0.50)]

    get_merged_y_val_no_extreme(city_name, day_type, de_params[0][0], de_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 0), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, de_params[1][0], de_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, de_params[2][0], de_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, de_params[3][0], de_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_no_extreme(city_name, day_type, de_params[4][0], de_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_no_extreme_%s.csv'%(city_name, day_type, 4), index=False)


    #%%

    """over extreme"""
    """over extreme"""
    """over extreme"""

    #%%

    city_name = 'seoul'
    day_type = 'weekdays'
    sd_params = [(0.75, 0.75), (1.0, 0.75), (0.75, 0.75), (1.0, 0.75), (1.0, 0.75)]

    #%% seoul, weekdays

    get_merged_y_val_over_extreme(city_name, day_type, sd_params[0][0], sd_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 0), index=False)


    get_merged_y_val_over_extreme(city_name, day_type, sd_params[1][0], sd_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, sd_params[2][0], sd_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, sd_params[3][0], sd_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, sd_params[4][0], sd_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 4), index=False)

    #%% seoul, weekends

    city_name = 'seoul'
    day_type = 'weekends'

    se_params = [(0.75, 0.75), (0.75, 0.50), (0.75, 0.75), (0.75, 0.50), (0.75, 0.75)]

    get_merged_y_val_over_extreme(city_name, day_type, se_params[0][0], se_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 0), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, se_params[1][0], se_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, se_params[2][0], se_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, se_params[3][0], se_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, se_params[4][0], se_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 4), index=False)

    #%% daejeon, weekdays

    city_name = 'daejeon'
    day_type = 'weekdays'

    dd_params = [(1.0, 0.50), (1.0, 0.75), (1.00, 0.75), (1.0, 0.75), (1.0, 0.75)]

    get_merged_y_val_over_extreme(city_name, day_type, dd_params[0][0], dd_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 0), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, dd_params[1][0], dd_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, dd_params[2][0], dd_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, dd_params[3][0], dd_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, dd_params[4][0], dd_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 4), index=False)

    #%% daejeon, weekends


    city_name = 'daejeon'
    day_type = 'weekends'

    de_params = [(1.0, 0.75), (1.0, 0.75), (1.0, 0.75), (1.0, 0.50), (1.0, 0.75)]

    get_merged_y_val_over_extreme(city_name, day_type, de_params[0][0], de_params[0][1], 0).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 0), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, de_params[1][0], de_params[1][1], 1).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 1), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, de_params[2][0], de_params[2][1], 2).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 2), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, de_params[3][0], de_params[3][1], 3).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 3), index=False)

    get_merged_y_val_over_extreme(city_name, day_type, de_params[4][0], de_params[4][1], 4).to_csv(r'results/bootstrapped/optimal/%s_%s_over_extreme_%s.csv'%(city_name, day_type, 4), index=False)

