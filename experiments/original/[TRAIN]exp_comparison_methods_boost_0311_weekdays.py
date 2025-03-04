# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:57:12 2021

@author: dohyeon
"""
#import ipdb
import time
import numpy as np
import pickle
import gzip
import user_utils as uu
#from analysis_utils import *
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost.sklearn import XGBRegressor
#from robust_boosting_algorithm1 import GradientBoostingRegressor1
from robust_boosting_algorithm import GradientBoostingRegressor1

from sklearn.tree import DecisionTreeRegressor
from noise_robust_models_0418 import AdaBoost_RT

#기존 py파일명에서 _1만 추가됨, 약한학습자 수에 대한 추가실험 GRB: 50, Adaboost:100
# 최대나무깊이에 대한 추가실험: 3,4
#np.seterr(all='ignore')

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

@logging_time
def exp_main(lr_name, X, y):
    temp_reg = lr_name.fit(X, y)
    return temp_reg

def rt_predict(X, fitted_obj):

    # y_preds.shape is (max_iter, len(X))
    y_preds = np.array([reg.predict(X) for reg in fitted_obj.estimators_])
    
    # weighted majority vote
    y_pred = np.sum(np.log(1.0/fitted_obj.betas_)[:,None] * y_preds, axis=0) / np.log(1.0/fitted_obj.betas_).sum()

    return y_pred



def avg_predict(X, fitted_obj):

    # y_preds.shape is (max_iter, len(X))
    y_preds = np.array([reg.predict(X) for reg in fitted_obj.estimators_])
    max_est, _ = y_preds.shape
    #ipdb.set_trace()
    # weighted majority vote
    #y_pred = y_preds.T @ fitted_obj.estimator_weights_
    y_pred = (y_preds.T @ fitted_obj.estimator_weights_[:max_est]) / np.sum(fitted_obj.estimator_weights_[:max_est])

    return y_pred

#%%


if __name__ == "__main__":


    with gzip.open('preprocess/preprocessed_dataset_0317_weekdays.pickle', 'rb') as d:
        data=pickle.load(d)
        trn_x , trn_y , tst_x , tst_y = data

    #%%
    print('start max depth 4')
    exp_day = '0317'
    n_est, max_d = [50, 4]

    add_name = ''
    #%%


    reg2=exp_main(XGBRegressor(max_depth=max_d, n_estimators=n_est), trn_x, np.log(trn_y+1))
    uu.save_gpickle(r'%sxgboost_%s_%s_%s_pred_y.pickle'%(add_name, n_est, max_d, exp_day), reg2.predict(tst_x))
    uu.save_gpickle(r'%sxgboost_%s_%s_%s.pickle'%(add_name, n_est, max_d, exp_day), reg2)
    reg2=None
    #%%
    reg2=exp_main(GradientBoostingRegressor1(max_depth=max_d, n_estimators=n_est, loss='huber'), trn_x, np.log(trn_y+1))
    uu.save_gpickle(r'%smboost_%s_%s_%s_pred_y.pickle'%(add_name, n_est, max_d, exp_day), reg2.predict(tst_x))
    uu.save_gpickle(r'%smboost_%s_%s_%s.pickle'%(add_name, n_est, max_d, exp_day), reg2)
    reg2=None

    #%%

    reg2=exp_main(GradientBoostingRegressor1(max_depth=max_d, n_estimators=n_est, loss='absolute_error'), trn_x, np.log(trn_y+1))
    uu.save_gpickle(r'%sl1loss_%s_%s_%s.pickle'%(add_name, n_est, max_d, exp_day), reg2)
    uu.save_gpickle(r'%sl1loss_%s_%s_%s_pred_y.pickle'%(add_name, n_est, max_d, exp_day), reg2.predict(tst_x))
    reg2=None

    #%%

    reg2=exp_main(GradientBoostingRegressor(max_depth=max_d, n_estimators=n_est), trn_x, np.log(trn_y+1))
    uu.save_gpickle(r'%sl2Loss_%s_%s_%s.pickle'%(add_name,n_est, max_d, exp_day), reg2)
    uu.save_gpickle(r'%sL2Loss_%s_%s_%s_pred_y.pickle'%(add_name,n_est, max_d, exp_day), reg2.predict(tst_x))
    reg2=None


    #%%
    ada_n_est = 50

    reg2=exp_main(AdaBoostRegressor(base_estimator= DecisionTreeRegressor(max_depth=max_d), n_estimators=ada_n_est), trn_x, np.log(trn_y+1))

    pred_y = avg_predict(tst_x, reg2)

    uu.save_gpickle(r'%sAdaBoost_%s_%s_%s_pred_y.pickle'%(add_name, ada_n_est, max_d, exp_day), pred_y)
    uu.save_gpickle(r'%sAdaBoost_%s_%s_%s.pickle'%(add_name, ada_n_est, max_d, exp_day), reg2)
    reg2=None

    #%%

    reg2=exp_main(AdaBoost_RT(base_estimator= DecisionTreeRegressor(max_depth=max_d), n_estimators=ada_n_est, pie=0.5), trn_x, np.log(trn_y+1))

    rt_pred = rt_predict(tst_x, reg2)

    uu.save_gpickle(r'%sAdaBoost_RT_%s_%s_%s_pred_y.pickle'%(add_name, ada_n_est, max_d, exp_day), rt_pred)
    uu.save_gpickle(r'%sAdaBoost_RT_%s_%s_%s.pickle'%(add_name, ada_n_est, max_d, exp_day), reg2)
    reg2=None







