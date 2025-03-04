# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:54:57 2024

@author: dohyeon
"""
#%% seoul

import user_utils as uu
import numpy as np
import os

#os.listdir(r'results//fitted_models//proposed_revision//seoul')


seoul_dir = r'results//fitted_models//proposed_revision//seoul'

#[xx for xx in os.listdir(seoul_dir) if 'weekdays' in xx]
#temp_obj = uu.load_gpickle(uu.get_full_path(seoul_dir, 'seoul_weekdays_model8_4_4_3.0_1.0_0.25_0.001.pickle'))

"""
#uu.open_dir(r'results//results//seoul')
#uu.open_dir(r'results//results//daejeon')

temp_obj.median_pred
temp_obj.avg_pred_resid
temp_obj.cdf_weight

np.sum(temp_obj.noise_obj.outliers_mask_2d_[:,:len(temp_obj.estimators_)], axis=1)
np.mean(temp_obj.noise_obj.residual_2d_[:, :len(temp_obj.estimators_)], axis=1)

np.sum(temp_obj.resid_dt_obj.worst_mask_2d_[:, :len(temp_obj.estimators_)], axis=1)
np.mean(temp_obj.resid_dt_obj.pred_residual_2d_[:,:len(temp_obj.estimators_)], axis=1)


np.array([np.sum([1 for xx_key, xx_val in xx.lr_dict.items() if xx_val[1].__str__() == 'Lasso(alpha=0.001)']) for xx in temp_obj.estimators_])
"""

#paths = [xx for xx in os.listdir(seoul_dir) if 'weekdays' in xx]

for fname in os.listdir(seoul_dir):
    temp_obj = None
    info_dict = None
    temp_obj = uu.load_gpickle(uu.get_full_path(seoul_dir, fname))
    if 'weekdays' in fname:
        info_dict = {'tst_median_pred':temp_obj.median_pred,
                     'tst_avg_pred_resid':temp_obj.avg_pred_resid,
                     'tst_cdf_weight':temp_obj.cdf_weight,
    
         'trn_outlier_cnt':np.sum(temp_obj.noise_obj.outliers_mask_2d_[:,:len(temp_obj.estimators_)], axis=1),
         'trn_avg_real_resid':np.mean(temp_obj.noise_obj.residual_2d_[:, :len(temp_obj.estimators_)], axis=1),
         'trn_avg_pred_resid':np.mean(temp_obj.resid_dt_obj.pred_residual_2d_[:,:len(temp_obj.estimators_)], axis=1),
         'trn_pos_resid_cnt':np.sum(temp_obj.resid_dt_obj.worst_mask_2d_[:, :len(temp_obj.estimators_)], axis=1),
         'trn_lr_cnt':np.array([np.sum([1 for xx_key, xx_val in xx.lr_dict.items() if xx_val[1].__str__() == 'Lasso(alpha=0.001)']) for xx in temp_obj.estimators_]),
         'trn_times':temp_obj.times_,
         'trn_est_errors':temp_obj.estimator_errors_,
         'trn_est_weights':temp_obj.estimator_weights_}
    else:
        info_dict = {'tst_median_pred':temp_obj.median_pred,
                     'tst_avg_pred_resid':temp_obj.avg_pred_resid,
                     'tst_cdf_weight':temp_obj.cdf_weight,
    
         'trn_outlier_cnt':np.sum(temp_obj.noise_obj.outliers_mask_2d_[:,:len(temp_obj.estimators_)], axis=1),
         'trn_avg_real_resid':np.mean(temp_obj.noise_obj.residual_2d_[:, :len(temp_obj.estimators_)], axis=1),
         'trn_avg_pred_resid':np.mean(temp_obj.resid_dt_obj.pred_residual_2d_[:,:len(temp_obj.estimators_)], axis=1),
         'trn_pos_resid_cnt':np.sum(temp_obj.resid_dt_obj.worst_mask_2d_[:, :len(temp_obj.estimators_)], axis=1),
         'trn_lr_cnt':np.array([np.sum([1 for xx_key, xx_val in xx.lr_dict.items() if xx_val[1].__str__() == 'Lasso(alpha=0.01)']) for xx in temp_obj.estimators_]),
         'trn_times':temp_obj.times_,
         'trn_est_errors':temp_obj.estimator_errors_,
         'trn_est_weights':temp_obj.estimator_weights_}

    uu.save_gpickle(uu.get_full_path(r'results//results//seoul', fname), info_dict)
#%%



#%% daejeon




import user_utils as uu
import numpy as np
import os

#os.listdir(r'results//fitted_models//proposed_revision//daejeon')


daejeon_dir = r'results//fitted_models//proposed_revision//daejeon'





for fname in os.listdir(daejeon_dir):
    temp_obj = None
    info_dict = None
    temp_obj = uu.load_gpickle(uu.get_full_path(daejeon_dir, fname))

    info_dict = {'tst_median_pred':temp_obj.median_pred,
                 'tst_avg_pred_resid':temp_obj.avg_pred_resid,
                 'tst_cdf_weight':temp_obj.cdf_weight,

     'trn_outlier_cnt':np.sum(temp_obj.noise_obj.outliers_mask_2d_[:,:len(temp_obj.estimators_)], axis=1),
     'trn_avg_real_resid':np.mean(temp_obj.noise_obj.residual_2d_[:, :len(temp_obj.estimators_)], axis=1),
     'trn_avg_pred_resid':np.mean(temp_obj.resid_dt_obj.pred_residual_2d_[:,:len(temp_obj.estimators_)], axis=1),
     'trn_pos_resid_cnt':np.sum(temp_obj.resid_dt_obj.worst_mask_2d_[:, :len(temp_obj.estimators_)], axis=1),
     'trn_lr_cnt':np.array([np.sum([1 for xx_key, xx_val in xx.lr_dict.items() if xx_val[1].__str__() == 'Lasso(alpha=0.01)']) for xx in temp_obj.estimators_]),
     'trn_times':temp_obj.times_,
     'trn_est_errors':temp_obj.estimator_errors_,
     'trn_est_weights':temp_obj.estimator_weights_}


    uu.save_gpickle(uu.get_full_path(r'results//results//daejeon', fname), info_dict)











