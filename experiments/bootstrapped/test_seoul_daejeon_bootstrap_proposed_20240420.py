# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:02:40 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np
import itertools
import user_utils as uu
from models_0708_8  import hierarchical_Estimator, Noise_corrector, Residual_DT, AdaBoostRegressor_ModelTree
from preprocess_dataset_bootstrap import split_x_y_main



if __name__ == "__main__":




    #day_types = ['weekdays','weekends']
    #city_names = ['daejeon', 'seoul']

    day_types = ['weekends',]
    city_names = ['daejeon',]
    #city_names = ['seoul',]


    avg_thresholds = [0.25, 0.50, 0.75, 1.00]
    avg_ratios = [0.50, 0.75]
    num_rs = 5
    n_est, max_d = [50, 4]
    max_d_2 = 4

    weighted_fit=True

    total_round = len(day_types) * len(city_names) * len(avg_thresholds) * len(avg_ratios) * num_rs

    i_round = 0
    for day_type, city_name in itertools.product(day_types, city_names):
        trn_df = pd.read_csv(r'preprocess/bootstrap/2018_train_%s_%s_20240418.csv'%(city_name, day_type))
        tst_df = pd.read_csv(r'preprocess/bootstrap/2019_test_%s_%s_20240418.csv'%(city_name, day_type))
        idx_df = pd.read_csv(r'preprocess/bootstrap/bootstrap_index_%s_%s_20240424.csv'%(city_name, day_type))


        if (city_name == 'seoul') & (day_type == 'weekdays'):
            thr_val, alpha = 3.0, 0.001
        elif (city_name == 'seoul') & (day_type == 'weekends'):
            thr_val, alpha = 3.5, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekdays'):
            thr_val, alpha = 2.0, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekends'):
            thr_val, alpha = 2.5, 0.01

        for random_state in np.arange(num_rs):
            X_trn, y_trn, X_tst, y_tst = split_x_y_main(trn_df, tst_df, idx_df, str(random_state))

            result_df = pd.DataFrame({'real':y_tst})
            result_df1 = pd.DataFrame({'real':y_tst})
            #result_df2 = pd.DataFrame({'real':y_tst})

            for avg_threshold, avg_ratio in itertools.product(avg_thresholds, avg_ratios):
                print('%s / %s round' % (i_round, total_round))

        
                reg1=None
                reg1 = AdaBoostRegressor_ModelTree(base_estimator=hierarchical_Estimator(max_depth=max_d,
                                                                                         avg_threshold=avg_threshold,
                                                                                         avg_ratio=avg_ratio,
                                                                                         alpha=alpha),
                                                   n_estimators=n_est,
                                                   noise_obj=Noise_corrector(is_corrected=True,
                                                                             outlier_thr=thr_val),
                                                   resid_dt_obj=Residual_DT(with_residual_dt=True,
                                                                            max_depth=max_d_2,
                                                                            weighted_fit=weighted_fit)
                                                   )
                for tt in reg1.__dict__.items():
                    if not (isinstance(tt[1], Noise_corrector) or isinstance(tt[1], Residual_DT)) :
                        print(tt)
                    else:
                        print(tt[0], tt[1].__dict__)
        
        
                print("start\n==================================")
                print(uu.get_datetime())
                reg1.fit(X_trn, np.log(y_trn+1), showtimes=True)
                print("trained\n==================================")
                print(uu.get_datetime())
                pred_y1 = reg1.predict(X_tst)
                result_df['%s_%s'%(str(avg_threshold), str(avg_ratio))] = pred_y1
                result_df1['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.cdf_weight
                #result_df2['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.median_pred

                i_round += 1



            result_df['info'] = 'pred_y'
            result_df1['info'] = 'cdf_weight'
            #result_df2['info'] = 'median_pred_y'

            pd.concat([result_df, result_df1]).reset_index(drop=True).to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_cdf_weight_%s.csv'%(city_name, day_type, random_state),index=False)

        print('%s-%s finish'%(city_name, day_type))




#%%


    day_types = ['weekends','weekdays',]
    city_names = ['seoul',]


    avg_thresholds = [0.25, 0.50, 0.75, 1.00]
    avg_ratios = [0.50, 0.75]
    num_rs = 5
    n_est, max_d = [50, 4]
    max_d_2 = 4

    weighted_fit=True

    total_round = len(day_types) * len(city_names) * len(avg_thresholds) * len(avg_ratios) * num_rs

    i_round = 0
    for day_type, city_name in itertools.product(day_types, city_names):
        trn_df = pd.read_csv(r'preprocess/bootstrap/2018_train_%s_%s_20240418.csv'%(city_name, day_type))
        tst_df = pd.read_csv(r'preprocess/bootstrap/2019_test_%s_%s_20240418.csv'%(city_name, day_type))
        idx_df = pd.read_csv(r'preprocess/bootstrap/bootstrap_index_%s_%s_20240424.csv'%(city_name, day_type))


        if (city_name == 'seoul') & (day_type == 'weekdays'):
            thr_val, alpha = 3.0, 0.001
        elif (city_name == 'seoul') & (day_type == 'weekends'):
            thr_val, alpha = 3.5, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekdays'):
            thr_val, alpha = 2.0, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekends'):
            thr_val, alpha = 2.5, 0.01

        for random_state in np.arange(num_rs):
            X_trn, y_trn, X_tst, y_tst = split_x_y_main(trn_df, tst_df, idx_df, str(random_state))

            result_df = pd.DataFrame({'real':y_tst})
            result_df1 = pd.DataFrame({'real':y_tst})
            #result_df2 = pd.DataFrame({'real':y_tst})

            for avg_threshold, avg_ratio in itertools.product(avg_thresholds, avg_ratios):
                print('%s / %s round' % (i_round, total_round))

        
                reg1=None
                reg1 = AdaBoostRegressor_ModelTree(base_estimator=hierarchical_Estimator(max_depth=max_d,
                                                                                         avg_threshold=avg_threshold,
                                                                                         avg_ratio=avg_ratio,
                                                                                         alpha=alpha),
                                                   n_estimators=n_est,
                                                   noise_obj=Noise_corrector(is_corrected=True,
                                                                             outlier_thr=thr_val),
                                                   resid_dt_obj=Residual_DT(with_residual_dt=True,
                                                                            max_depth=max_d_2,
                                                                            weighted_fit=weighted_fit)
                                                   )
                for tt in reg1.__dict__.items():
                    if not (isinstance(tt[1], Noise_corrector) or isinstance(tt[1], Residual_DT)) :
                        print(tt)
                    else:
                        print(tt[0], tt[1].__dict__)
        
        
                print("start\n==================================")
                print(uu.get_datetime())
                reg1.fit(X_trn, np.log(y_trn+1), showtimes=True)
                print("trained\n==================================")
                print(uu.get_datetime())
                pred_y1 = reg1.predict(X_tst)
                result_df['%s_%s'%(str(avg_threshold), str(avg_ratio))] = pred_y1
                result_df1['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.cdf_weight
                #result_df2['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.median_pred

                i_round += 1



            result_df['info'] = 'pred_y'
            result_df1['info'] = 'cdf_weight'
            #result_df2['info'] = 'median_pred_y'

            pd.concat([result_df, result_df1]).reset_index(drop=True).to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_cdf_weight_%s.csv'%(city_name, day_type, random_state),index=False)

        print('%s-%s finish'%(city_name, day_type))


    #%%


    day_types = ['weekdays',]
    city_names = ['seoul',]
    avg_thresholds = [0.25, 0.50, 0.75, 1.00]
    avg_ratios = [0.50, 0.75]


    random_state = 2
    n_est, max_d = [50, 4]
    max_d_2 = 4

    weighted_fit = True

    total_round = len(day_types) * len(city_names) * len(avg_thresholds) * len(avg_ratios)

    i_round = 0
    for day_type, city_name in itertools.product(day_types, city_names):
        trn_df = pd.read_csv(r'preprocess/bootstrap/2018_train_%s_%s_20240418.csv'%(city_name, day_type))
        tst_df = pd.read_csv(r'preprocess/bootstrap/2019_test_%s_%s_20240418.csv'%(city_name, day_type))
        idx_df = pd.read_csv(r'preprocess/bootstrap/bootstrap_index_%s_%s_20240424.csv'%(city_name, day_type))

        if (city_name == 'seoul') & (day_type == 'weekdays'):
            thr_val, alpha = 3.0, 0.001
        elif (city_name == 'seoul') & (day_type == 'weekends'):
            thr_val, alpha = 3.5, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekdays'):
            thr_val, alpha = 2.0, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekends'):
            thr_val, alpha = 2.5, 0.01

        X_trn, y_trn, X_tst, y_tst = split_x_y_main(trn_df, tst_df, idx_df, str(random_state))

        #result_df1 = pd.DataFrame({'real':y_tst})
        #result_df2 = pd.DataFrame({'real':y_tst})

        for avg_threshold, avg_ratio in itertools.product(avg_thresholds, avg_ratios):
            result_df = pd.DataFrame({'real':y_tst})

            print('%s / %s round' % (i_round, total_round))

            reg1=None
            reg1 = AdaBoostRegressor_ModelTree(base_estimator=hierarchical_Estimator(max_depth=max_d,
                                                                                     avg_threshold=avg_threshold,
                                                                                     avg_ratio=avg_ratio,
                                                                                     alpha=alpha),
                                               n_estimators=n_est,
                                               noise_obj=Noise_corrector(is_corrected=True,
                                                                         outlier_thr=thr_val),
                                               resid_dt_obj=Residual_DT(with_residual_dt=True,
                                                                        max_depth=max_d_2,
                                                                        weighted_fit=weighted_fit)
                                               )
            for tt in reg1.__dict__.items():
                if not (isinstance(tt[1], Noise_corrector) or isinstance(tt[1], Residual_DT)) :
                    print(tt)
                else:
                    print(tt[0], tt[1].__dict__)
    
    
            print("start\n==================================")
            print(uu.get_datetime())
            reg1.fit(X_trn, np.log(y_trn+1), showtimes=True)
            print("trained\n==================================")
            print(uu.get_datetime())
            pred_y1 = reg1.predict(X_tst)
            result_df['%s_%s'%(str(avg_threshold), str(avg_ratio))] = pred_y1
            #result_df1['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.cdf_weight
            #result_df2['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.median_pred

            i_round += 1
            result_df.to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_%s_%s_%s.csv'%(city_name, day_type, random_state, str(avg_threshold), str(avg_ratio)),index=False)

        #result_df['info'] = 'pred_y'
        #result_df1['info'] = 'cdf_weight'
        #result_df2['info'] = 'median_pred_y'


    print('%s-%s finish'%(city_name, day_type))



#%%

    day_types = ['weekdays',]
    city_names = ['seoul',]
    avg_thresholds = [0.25, 0.50, 0.75, 1.00]
    avg_ratios = [0.50, 0.75]


    #random_state = 2
    n_est, max_d = [50, 4]
    max_d_2 = 4

    weighted_fit = True

    total_round = len(day_types) * len(city_names) * len(avg_thresholds) * len(avg_ratios) * 3

    i_round = 0
    for day_type, city_name in itertools.product(day_types, city_names):
        trn_df = pd.read_csv(r'preprocess/bootstrap/2018_train_%s_%s_20240418.csv'%(city_name, day_type))
        tst_df = pd.read_csv(r'preprocess/bootstrap/2019_test_%s_%s_20240418.csv'%(city_name, day_type))
        idx_df = pd.read_csv(r'preprocess/bootstrap/bootstrap_index_%s_%s_20240424.csv'%(city_name, day_type))

        if (city_name == 'seoul') & (day_type == 'weekdays'):
            thr_val, alpha = 3.0, 0.001
        elif (city_name == 'seoul') & (day_type == 'weekends'):
            thr_val, alpha = 3.5, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekdays'):
            thr_val, alpha = 2.0, 0.01
        elif (city_name == 'daejeon') & (day_type == 'weekends'):
            thr_val, alpha = 2.5, 0.01

        for random_state in [2,3,4]:
            X_trn, y_trn, X_tst, y_tst = split_x_y_main(trn_df, tst_df, idx_df, str(random_state))
    
            #result_df1 = pd.DataFrame({'real':y_tst})
            #result_df2 = pd.DataFrame({'real':y_tst})
    
            for avg_threshold, avg_ratio in itertools.product(avg_thresholds, avg_ratios):
                result_df = pd.DataFrame({'real':y_tst})
    
                print('%s / %s round' % (i_round, total_round))
    
                reg1=None
                reg1 = AdaBoostRegressor_ModelTree(base_estimator=hierarchical_Estimator(max_depth=max_d,
                                                                                         avg_threshold=avg_threshold,
                                                                                         avg_ratio=avg_ratio,
                                                                                         alpha=alpha),
                                                   n_estimators=n_est,
                                                   noise_obj=Noise_corrector(is_corrected=True,
                                                                             outlier_thr=thr_val),
                                                   resid_dt_obj=Residual_DT(with_residual_dt=True,
                                                                            max_depth=max_d_2,
                                                                            weighted_fit=weighted_fit)
                                                   )
                for tt in reg1.__dict__.items():
                    if not (isinstance(tt[1], Noise_corrector) or isinstance(tt[1], Residual_DT)) :
                        print(tt)
                    else:
                        print(tt[0], tt[1].__dict__)


                print("start\n==================================")
                print(uu.get_datetime())
                reg1.fit(X_trn, np.log(y_trn+1), showtimes=True)
                print("trained\n==================================")
                print(uu.get_datetime())
                pred_y1 = reg1.predict(X_tst)
                result_df['%s_%s'%(str(avg_threshold), str(avg_ratio))] = pred_y1
                #result_df1['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.cdf_weight
                #result_df2['%s_%s'%(str(avg_threshold), str(avg_ratio))] = reg1.median_pred
    
                i_round += 1
                result_df.to_csv(r'results/bootstrapped/proposed/%s/%s/y_val_%s_%s_%s.csv'%(city_name, day_type, random_state, str(avg_threshold), str(avg_ratio)),index=False)

            #result_df['info'] = 'pred_y'
            #result_df1['info'] = 'cdf_weight'
            #result_df2['info'] = 'median_pred_y'


    print('%s-%s finish'%(city_name, day_type))



























        


