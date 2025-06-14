# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:15:08 2024

@author: dohyeon
"""

import user_utils as uu
import pandas as pd
import numpy as np


def convert_exp_comparison(comparison_df):

    model_list = ['AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']
    for uni_model in model_list:
        comparison_df[uni_model] = np.exp(comparison_df[uni_model].values)-1
    return comparison_df[['AdaBoost', 'AdaBoost_RT', 'GBM', 'XGBoost', 'MBoost', 'LADBoost']]

#%% seoul, weekdays


temp_df = pd.read_csv(r'preprocess/index_seoul_tst_weekdays.csv')

temp_df1 = convert_exp_comparison(pd.read_csv(r'seoul_real_pred_weekdays.csv'))


temp_df2 = pd.read_csv(r'y_val_cdf_weight_seoul_weekdays_model8_lasso_avg_threshold_ratio.csv')


new_temp_df2 = temp_df2.loc[temp_df2['info']=='pred_y'][['real','3.0_1.0_0.75']]
new_temp_df2.columns = ['real', 'pred']


weekdays_obj = uu.load_gpickle(uu.get_full_path(r'results//results//seoul', 'seoul_weekdays_model8_4_4_3.0_1.0_0.75_0.001.pickle' ))



concated_df = pd.concat([temp_df, new_temp_df2],axis=1)
concated_df['median_pred'] = weekdays_obj['tst_median_pred']

concated_df = pd.concat([concated_df, temp_df1], axis=1)

concated_df.to_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv', index=False)

#%% seoul, weekends

temp_df = pd.read_csv(r'preprocess/index_seoul_tst_weekends.csv')

temp_df1 = convert_exp_comparison(pd.read_csv(r'seoul_real_pred_weekends.csv'))


temp_df2 = pd.read_csv(r'y_val_cdf_weight_seoul_weekends_model8_lasso_avg_threshold_ratio.csv')


new_temp_df2 = temp_df2.loc[temp_df2['info']=='pred_y'][['real','3.5_0.75_0.75_0.01']]
new_temp_df2.columns = ['real', 'pred']


weekends_obj = uu.load_gpickle(uu.get_full_path(r'results//results//seoul', 'seoul_weekends_model8_4_4_3.5_0.75_0.75_0.01.pickle'))


concated_df = pd.concat([temp_df, new_temp_df2],axis=1)
concated_df['median_pred'] = weekends_obj['tst_median_pred']

concated_df = pd.concat([concated_df, temp_df1], axis=1)

concated_df.to_csv(r'seoul_weekends_real_pred_prop_median_comparison_demand.csv', index=False)


#%%

"""daejeon"""

#%% daejeon, weekdays


temp_df = pd.read_csv(r'preprocess/daejeon/index_daejeon_tst_weekdays.csv')

temp_df1 = convert_exp_comparison(pd.read_csv(r'daejeon_real_pred_weekdays.csv'))


temp_df2 = pd.read_csv(r'y_val_cdf_weight_daejeon_weekdays_model8_lasso_avg_threshold_ratio.csv')


new_temp_df2 = temp_df2.loc[temp_df2['info']=='pred_y'][['real','2.0_1.0_0.75_0.01']]
new_temp_df2.columns = ['real', 'pred']


weekdays_obj = uu.load_gpickle(uu.get_full_path(r'results//results//daejeon', 'daejeon_weekdays_model8_4_4_2.0_1.0_0.75_0.01.pickle'))



concated_df = pd.concat([temp_df, new_temp_df2],axis=1)
concated_df['median_pred'] = weekdays_obj['tst_median_pred']

concated_df = pd.concat([concated_df, temp_df1], axis=1)

concated_df.to_csv(r'daejeon_weekdays_real_pred_prop_median_comparison_demand.csv', index=False)




#%% daejeon, weekends



temp_df = pd.read_csv(r'preprocess/daejeon/index_daejeon_tst_weekends.csv')

temp_df1 = convert_exp_comparison(pd.read_csv(r'daejeon_real_pred_weekends.csv'))


temp_df2 = pd.read_csv(r'y_val_cdf_weight_daejeon_weekends_model8_lasso_avg_threshold_ratio.csv')


new_temp_df2 = temp_df2.loc[temp_df2['info']=='pred_y'][['real','2.5_1.0_0.75_0.01']]
new_temp_df2.columns = ['real', 'pred']


weekends_obj = uu.load_gpickle(uu.get_full_path(r'results//results//daejeon', 'daejeon_weekends_model8_4_4_2.5_1.0_0.75_0.01.pickle' ))


concated_df = pd.concat([temp_df, new_temp_df2],axis=1)
concated_df['median_pred'] = weekends_obj['tst_median_pred']

concated_df = pd.concat([concated_df, temp_df1], axis=1)

concated_df.to_csv(r'daejeon_weekends_real_pred_prop_median_comparison_demand.csv', index=False)































