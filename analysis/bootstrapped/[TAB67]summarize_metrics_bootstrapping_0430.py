# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:47:23 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np
from scipy import stats


def strtolatex(str1):
    if str1 == 'better':
        output = r'\uparrow'
    elif str1 == 'worse':
        output = r'\downarrow'
    elif (str1 == '0') or (str1 == 0):
        output = r''
    return output


def merge_avg_ttest(avg_val, ttest_val):

    tt_str = ''
    for aidx, (aa, bb) in enumerate(zip(avg_val, ttest_val)):
        if aidx < 9:
            if not r'\textbf' in aa :
                tt_str += aa.replace(" ", "").split('&')[0] + '$^{%s}$'%(strtolatex(bb))+ ' & '
            else:
                tt_str += '\\textbf{' + aa.replace(" ", "").split("\\textbf{")[1].replace("}", "").replace("&", "") + '$^{%s}$'%(strtolatex(bb)) +'}' + ' & '
        else:
            if not r'\textbf' in aa :
                tt_str += aa.replace(" ", "").replace('\\\\','').split('&')[0] + '$^{%s}$'%(strtolatex(bb))+ ' \\\\ '
            else:
                tt_str += '\\textbf{' + (aa.replace(" ", "").replace('\\\\','').split("\\textbf{")[1].replace("}", "") + '$^{%s}$'%(strtolatex(bb)) +'}') + '\\\\'

    return tt_str


if __name__ == "__main__":
    temp_df = pd.read_csv(r'prediction_performance_seoul_daejeon_weekdays_weekends_bootstrap_5.csv')
    city_name = 'daejeon'

    avg_df = temp_df.groupby(['city_name','day_type','Group','Model']).mean().drop(columns='rs').loc[city_name].unstack(0).round(4).loc[
        [('Group1','pred'),('Group1','AdaBoost'),('Group1','AdaBoost_RT'),('Group1','GBM'), ('Group1','XGBoost'),('Group1','MBoost'),('Group1','LADBoost'),
         ('Group2','pred'),('Group2','AdaBoost'),('Group2','AdaBoost_RT'),('Group2','GBM'), ('Group2','XGBoost'),('Group2','MBoost'),('Group2','LADBoost'),
         ('Group3','pred'),('Group3','AdaBoost'),('Group3','AdaBoost_RT'),('Group3','GBM'), ('Group3','XGBoost'),('Group3','MBoost'),('Group3','LADBoost'),
         ('Whole','pred'),('Whole','AdaBoost'),('Whole','AdaBoost_RT'),('Whole','GBM'), ('Whole','XGBoost'),('Whole','MBoost'),('Whole','LADBoost')]]



    std_df = temp_df.groupby(['city_name','day_type','Group','Model']).std().drop(columns='rs').loc[city_name].unstack(0).round(4).loc[
        [('Group1','pred'),('Group1','AdaBoost'),('Group1','AdaBoost_RT'),('Group1','GBM'), ('Group1','XGBoost'),('Group1','MBoost'),('Group1','LADBoost'),
         ('Group2','pred'),('Group2','AdaBoost'),('Group2','AdaBoost_RT'),('Group2','GBM'), ('Group2','XGBoost'),('Group2','MBoost'),('Group2','LADBoost'),
         ('Group3','pred'),('Group3','AdaBoost'),('Group3','AdaBoost_RT'),('Group3','GBM'), ('Group3','XGBoost'),('Group3','MBoost'),('Group3','LADBoost'),
         ('Whole','pred'),('Whole','AdaBoost'),('Whole','AdaBoost_RT'),('Whole','GBM'), ('Whole','XGBoost'),('Whole','MBoost'),('Whole','LADBoost')]]


    rows = []
    for (idx, row) in avg_df.iterrows():
        rows.append(row.astype(str))
        rows.append(std_df.applymap(lambda x: f'{x:.4f}').loc[idx])

    seoul_df = pd.DataFrame(rows, index=avg_df.index.repeat(2))
    seoul_df.to_csv(r'prediction_performance_for_latex_bootstrap_%s_same_ratio.csv'%(city_name))


    #%%
    bin_bin_list = []
    for uni_args, metric_df in temp_df.groupby(['city_name','day_type','Group']):
        #bin_list = []
        for uni_model in metric_df.Model.unique()[1:]:
            for uni_metric in ['RMSE', 'MAE','MAPE', 'sMAPE', 'sMAPE_0']:
                arr1= metric_df.loc[metric_df['Model']=='pred'][uni_metric].values
                arr2= metric_df.loc[metric_df['Model']==uni_model][uni_metric].values
                tstat, pval = stats.ttest_rel(arr2, arr1)
                if pval >0.05:
                    output = 0
                else:
                    if tstat < 0 : output = 'worse'
                    else: output= 'better'
                bin_bin_list.append(list(uni_args) + [uni_model, uni_metric, output])

    #%%

    bin_bin_df= pd.DataFrame(data=bin_bin_list, columns=['city_name','day_type','Group','Model','metric','eval'])
    bin_bin_df

    ttest_df = bin_bin_df.loc[bin_bin_df.city_name == city_name].drop(columns='city_name').pivot(index=['Group','Model'],columns=['metric','day_type'], values='eval')[[('RMSE','weekdays'),('RMSE','weekends'),('MAE','weekdays'),('MAE','weekends'),('MAPE','weekdays'),('MAPE','weekends'),('sMAPE','weekdays'),('sMAPE','weekends'),('sMAPE_0','weekdays'),('sMAPE_0','weekends') ]]

    ttest_df = ttest_df.loc[[('Group1','AdaBoost'),('Group1','AdaBoost_RT'),('Group1','GBM'), ('Group1','XGBoost'),('Group1','MBoost'),('Group1','LADBoost'),
                             ('Group2','AdaBoost'),('Group2','AdaBoost_RT'),('Group2','GBM'), ('Group2','XGBoost'),('Group2','MBoost'),('Group2','LADBoost'),
                             ('Group3','AdaBoost'),('Group3','AdaBoost_RT'),('Group3','GBM'), ('Group3','XGBoost'),('Group3','MBoost'),('Group3','LADBoost'),
                             ('Whole','AdaBoost'),('Whole','AdaBoost_RT'),('Whole','GBM'), ('Whole','XGBoost'),('Whole','MBoost'),('Whole','LADBoost')]]


    #file1 = open(r'latex_test1.txt', "r").readlines()
    file1 = open(r'prediction_performance_bootstrap_same_ratio_%s_latex.txt'%(city_name), "r").readlines()
    file1 = [xx.replace('\n', '') for xx in file1]



    start_val = 55
    for xx in range(ttest_df.shape[0]):
        if xx % 6 == 0:
            if xx>0:
                start_val += 26
        print('%s th'%(xx))
        print(merge_avg_ttest(file1[start_val+xx*26:start_val+xx*26+10], ttest_df.loc[ttest_df.index[xx]].values))
        print('                                                                                 ')



