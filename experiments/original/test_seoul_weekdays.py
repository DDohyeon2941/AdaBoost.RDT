# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:55:49 2024

@author: dohyeon
"""

    #%% 따릉이 데이터


import pandas as pd
import numpy as np
import user_utils as uu
from models_0708_8  import hierarchical_Estimator, Noise_corrector, Residual_DT, AdaBoostRegressor_ModelTree
import itertools



if __name__ == "__main__":

    X_trn , y_trn , X_tst , y_tst = uu.load_gpickle(r'preprocess/preprocessed_dataset_0317_weekdays.pickle')

    n_est = 50
    max_d_2 = 4
    max_ds = [4, ]
    #thr_vals = [3.0, 3.5, 4.0]
    thr_vals = [3.0, ]
    avg_thresholds = [0.25, 0.50, 0.75, 1.00]
    avg_ratios = [0.25, 0.50, 0.75]
    alpha= 0.001

    weighted_fit=True

    result_df = pd.DataFrame({'real':y_tst})
    result_df1 = pd.DataFrame({'real':y_tst})


    total_round = len(thr_vals) * len(avg_thresholds) * len(avg_ratios)
    for i_round, (max_d, thr_val, avg_threshold, avg_ratio) in enumerate(list(itertools.product(max_ds, thr_vals, avg_thresholds, avg_ratios))):

        print('%s / %s round' % (i_round, total_round))
        print("start threshold: %s \n==================================" % (str(thr_val)))


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
        result_df['%s_%s_%s'%(str(thr_val), str(avg_threshold), str(avg_ratio))] = pred_y1
        result_df1['%s_%s_%s'%(str(thr_val), str(avg_threshold), str(avg_ratio))] = reg1.cdf_weight

        uu.save_gpickle(r'results//fitted_models//proposed_revision//seoul//seoul_weekdays_model8_%s_%s_%s_%s_%s_%s.pickle'%(max_d, max_d_2, str(thr_val), avg_threshold, avg_ratio, str(alpha)), reg1)


    #%%

    result_df['info'] = 'pred_y'
    result_df1['info'] = 'cdf_weight'

    #pd.concat([result_df, result_df1]).reset_index(drop=True).to_csv(r'y_val_cdf_weight_seoul_weekdays_model8_lasso_avg_threshold_ratio.csv', index=False)


