# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:57:12 2021

@author: dohyeon
"""
import time
import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost.sklearn import XGBRegressor
from robust_boosting_algorithm import GradientBoostingRegressor1
from sklearn.tree import DecisionTreeRegressor
from noise_robust_models_0418 import AdaBoost_RT
from preprocess_dataset_bootstrap import split_x_y_main

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

    day_types = ['weekdays','weekends']
    city_names = ['daejeon', 'seoul']

    for day_type, city_name in itertools.product(day_types, city_names):
        trn_df = pd.read_csv(r'preprocess/bootstrap/2018_train_%s_%s_20240418.csv'%(city_name, day_type))
        tst_df = pd.read_csv(r'preprocess/bootstrap/2019_test_%s_%s_20240418.csv'%(city_name, day_type))
        idx_df = pd.read_csv(r'preprocess/bootstrap/bootstrap_index_%s_%s_20240424.csv'%(city_name, day_type))
        n_est, max_d = [50, 4]
        ada_n_est = 50
    
    
        for random_state in np.arange(5):
            X_trn, y_trn, X_tst, y_tst = split_x_y_main(trn_df, tst_df, idx_df, str(random_state))
        
    
            result_df = pd.DataFrame({'real':y_tst})
        
            ###
            reg2=exp_main(XGBRegressor(max_depth=max_d, n_estimators=n_est), X_trn, np.log(y_trn+1))
            y_pred = reg2.predict(X_tst)
            y_pred[y_pred<0] = 0
            y_pred = np.exp(y_pred)-1
        
            result_df['XGBoost'] = y_pred
            ###
            reg2=exp_main(GradientBoostingRegressor1(max_depth=max_d, n_estimators=n_est, loss='huber'), X_trn, np.log(y_trn+1))
            y_pred = reg2.predict(X_tst)
            y_pred[y_pred<0] = 0
            y_pred = np.exp(y_pred)-1
            result_df['MBoost'] = y_pred
        
            ###
    
            reg2=exp_main(GradientBoostingRegressor1(max_depth=max_d, n_estimators=n_est, loss='absolute_error'), X_trn, np.log(y_trn+1))
            y_pred = reg2.predict(X_tst)
            y_pred[y_pred<0] = 0
            y_pred = np.exp(y_pred)-1
            result_df['LADBoost'] = y_pred
        
            ###
    
            reg2=exp_main(GradientBoostingRegressor(max_depth=max_d, n_estimators=n_est), X_trn, np.log(y_trn+1))
            y_pred = reg2.predict(X_tst)
            y_pred[y_pred<0] = 0
            y_pred = np.exp(y_pred)-1
            result_df['GBM'] = y_pred
        
        
            ###
    
            reg2=exp_main(AdaBoostRegressor(base_estimator= DecisionTreeRegressor(max_depth=max_d), n_estimators=ada_n_est), X_trn, np.log(y_trn+1))
        
            y_pred = avg_predict(X_tst, reg2)
            y_pred[y_pred<0] = 0
            y_pred = np.exp(y_pred)-1
            result_df['AdaBoost'] = y_pred
        
            ###
    
            reg2=exp_main(AdaBoost_RT(base_estimator= DecisionTreeRegressor(max_depth=max_d), n_estimators=ada_n_est, pie=0.5), X_trn, np.log(y_trn+1))
        
            y_pred = rt_predict(X_tst, reg2)
            y_pred[y_pred<0] = 0
            y_pred = np.exp(y_pred)-1
            result_df['AdaBoost_RT'] = y_pred
    
            ###
            result_df.to_csv(r'results/bootstrapped/comparison/%s/%s/y_val_%s.csv'%(city_name, day_type, random_state),index=False)
            print('save %s - %s %s th results csv file'%(city_name, day_type, random_state))
            print('-'*50)
        
