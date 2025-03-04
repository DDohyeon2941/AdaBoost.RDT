# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:51:21 2024

@author: dohyeon
"""



import pandas as pd
import numpy as np
import user_utils as uu


info_dict = {}

info_dict1 = {}
seoul_dir = r'results//fitted_models//proposed_revision//seoul'
daejeon_dir = r'results//fitted_models//proposed_revision//daejeon'
if __name__ == "__main__":


    _ , _ , X_tst , y_tst = uu.load_gpickle(r'preprocess/preprocessed_dataset_0317_weekdays.pickle')

    reg1 = uu.load_gpickle(uu.get_full_path(seoul_dir, 'seoul_weekdays_model8_4_4_3.0_1.0_0.75_0.001.pickle'))
    aa,_,_,_ = reg1.obj_predict_elements1(X_tst, len(reg1.estimators_), False)
    tst_real_resid_2d = np.log(y_tst+1).reshape(-1,1) - aa
    tst_avg_real_resid = np.mean(tst_real_resid_2d, axis=1)

    info_dict['weekdays'] = tst_avg_real_resid

    #%%

    reg1 = None
    aa = None
    tst_real_resid_2d = None
    tst_avg_real_resid = None


    #%%

    _ , _ , X_tst , y_tst = uu.load_gpickle(r'preprocess/preprocessed_dataset_0317_holy.pickle')

    reg1 = uu.load_gpickle(uu.get_full_path(seoul_dir, 'seoul_weekends_model8_4_4_3.5_0.75_0.75_0.01.pickle'))
    aa,_,_,_ = reg1.obj_predict_elements1(X_tst, len(reg1.estimators_), False)
    tst_real_resid_2d = np.log(y_tst+1).reshape(-1,1) - aa
    tst_avg_real_resid = np.mean(tst_real_resid_2d, axis=1)

    info_dict['weekends'] = tst_avg_real_resid


    #uu.save_gpickle(r'results//results//seoul//tst_avg_real_resid.pickle', info_dict)

    #test_obj = uu.load_gpickle(r'results//results//seoul//tst_avg_real_resid.pickle')
    #%%

    reg1 = None
    aa = None
    tst_real_resid_2d = None
    tst_avg_real_resid = None



    #%% daejeon weekdays


    _ , _ , X_tst , y_tst = uu.load_gpickle(r'preprocess/daejeon/preprocessed_dataset_0317_weekdays.pickle')

    reg1 = uu.load_gpickle(uu.get_full_path(daejeon_dir, 'daejeon_weekdays_model8_4_4_2.0_1.0_0.75_0.01.pickle'))
    aa,_,_,_ = reg1.obj_predict_elements1(X_tst, len(reg1.estimators_), False)
    tst_real_resid_2d = np.log(y_tst+1).reshape(-1,1) - aa
    tst_avg_real_resid = np.mean(tst_real_resid_2d, axis=1)

    info_dict1['weekdays'] = tst_avg_real_resid

    #%%
    reg1 = None
    aa = None
    tst_real_resid_2d = None
    tst_avg_real_resid = None
    #%%
    _ , _ , X_tst , y_tst = uu.load_gpickle(r'preprocess/daejeon/preprocessed_dataset_0317_holy.pickle')

    reg1 = uu.load_gpickle(uu.get_full_path(daejeon_dir, 'daejeon_weekends_model8_4_4_2.5_1.0_0.75_0.01.pickle'))
    aa,_,_,_ = reg1.obj_predict_elements1(X_tst, len(reg1.estimators_), False)
    tst_real_resid_2d = np.log(y_tst+1).reshape(-1,1) - aa
    tst_avg_real_resid = np.mean(tst_real_resid_2d, axis=1)

    info_dict1['weekends'] = tst_avg_real_resid


    #uu.save_gpickle(r'results//results//daejeon//tst_avg_real_resid.pickle', info_dict1)

















