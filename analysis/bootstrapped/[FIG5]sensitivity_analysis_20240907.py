# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:37:21 2024

@author: dohyeon
"""



import pandas as pd

import numpy as np
import user_utils as uu
import matplotlib.pyplot as plt



temp_df= pd.read_csv(r'X:\prediction_performance_seoul_weekdays_model8_lasso_avg_threshold_ratio.csv')

temp_df1= pd.read_csv(r'X:\prediction_performance_seoul_weekends_model8_lasso_avg_threshold_ratio.csv')

temp_df2= pd.read_csv(r'X:\prediction_performance_daejeon_weekdays_model8_lasso_avg_threshold_ratio.csv')


temp_df3= pd.read_csv(r'X:\prediction_performance_daejeon_weekends_model8_lasso_avg_threshold_ratio.csv')


temp_df3.Model.unique()



new_temp_df = (temp_df.loc[temp_df['Model'].isin(['3.0_0.25_0.25', '3.0_0.25_0.5',
                                   '3.0_0.25_0.75', '3.0_0.5_0.25','3.0_0.5_0.5',
                                   '3.0_0.5_0.75', '3.0_0.75_0.25',
                                   '3.0_0.75_0.5', '3.0_0.75_0.75',
                                   '3.0_1.0_0.25', '3.0_1.0_0.5',
                                   '3.0_1.0_0.75'])]).reset_index(drop=True)

new_temp_df1 = (temp_df1.loc[temp_df1['Model'].isin(['3.5_0.25_0.25_0.01', '3.5_0.25_0.5_0.01', '3.5_0.25_0.75_0.01',
       '3.5_0.5_0.25_0.01', '3.5_0.5_0.5_0.01', '3.5_0.5_0.75_0.01',
       '3.5_0.75_0.25_0.01', '3.5_0.75_0.5_0.01', '3.5_0.75_0.75_0.01',
       '3.5_1.0_0.25_0.01', '3.5_1.0_0.5_0.01', '3.5_1.0_0.75_0.01'])]).reset_index(drop=True)

new_temp_df2 = (temp_df2.loc[temp_df2['Model'].isin(['2.0_0.25_0.25_0.01', '2.0_0.25_0.5_0.01', '2.0_0.25_0.75_0.01',
       '2.0_0.5_0.25_0.01', '2.0_0.5_0.5_0.01', '2.0_0.5_0.75_0.01',
       '2.0_0.75_0.25_0.01', '2.0_0.75_0.5_0.01', '2.0_0.75_0.75_0.01',
       '2.0_1.0_0.25_0.01', '2.0_1.0_0.5_0.01', '2.0_1.0_0.75_0.01'])]).reset_index(drop=True)

new_temp_df3 = (temp_df3.loc[temp_df3['Model'].isin(['2.5_0.25_0.25_0.01', '2.5_0.25_0.5_0.01', '2.5_0.25_0.75_0.01',
       '2.5_0.5_0.25_0.01', '2.5_0.5_0.5_0.01', '2.5_0.5_0.75_0.01',
       '2.5_0.75_0.25_0.01', '2.5_0.75_0.5_0.01', '2.5_0.75_0.75_0.01',
       '2.5_1.0_0.25_0.01', '2.5_1.0_0.5_0.01', '2.5_1.0_0.75_0.01'])]).reset_index(drop=True)



new_temp_df.loc[:, 'alpha']  = [xx.split('_')[1] for xx in new_temp_df['Model']]
new_temp_df.loc[:, 'beta']  = [xx.split('_')[2] for xx in new_temp_df['Model']]

###
new_temp_df1.loc[:, 'alpha']  = [xx.split('_')[1] for xx in new_temp_df1['Model']]
new_temp_df1.loc[:, 'beta']  = [xx.split('_')[2] for xx in new_temp_df1['Model']]

###
new_temp_df2.loc[:, 'alpha']  = [xx.split('_')[1] for xx in new_temp_df2['Model']]
new_temp_df2.loc[:, 'beta']  = [xx.split('_')[2] for xx in new_temp_df2['Model']]

###
new_temp_df3.loc[:, 'alpha']  = [xx.split('_')[1] for xx in new_temp_df3['Model']]
new_temp_df3.loc[:, 'beta']  = [xx.split('_')[2] for xx in new_temp_df3['Model']]

#%%


import seaborn as sns
# Group by alpha and beta, and calculate the average RMSE, MAE, etc.
grouped_data = new_temp_df.groupby(['Group', 'alpha', 'beta']).mean().reset_index()

# Loop through each group to create heatmaps for RMSE, MAE, etc.
for group in grouped_data['Group'].unique():
    if group in ['Group1', 'Group3']:
        group_data = grouped_data[grouped_data['Group'] == group]
    
        # Create a pivot table for RMSE to create a heatmap
        rmse_pivot = group_data.pivot('alpha', 'beta', 'MAPE')
    
        # Plot heatmap for RMSE for this group
        #plt.figure(figsize=(8, 6))
        fig1 ,axes1 = plt.subplots(1,1, figsize=(8,6))
        if group == 'Group1':
            sns.heatmap(rmse_pivot, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.66, vmin=0.45, ax=axes1)
        elif group == 'Group3':
            sns.heatmap(rmse_pivot, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.81, vmin=0.43, ax=axes1)

        #plt.title(f'Sensitivity of MAPE to Alpha and Beta for {group}')
        plt.xlabel('beta(β)', fontsize=25)
        axes1.tick_params(labelsize=25)

        #plt.xticks(fontsize=20)
        #plt.yticks(fontsize=20)
        plt.ylabel('alpha(α)', fontsize=25)
        plt.savefig('Seoul_Weekdays_%s_MAPE_20240915.png'%(group), dpi=1024, transparent=True, bbox_inches='tight')
        plt.show()

#%%
import seaborn as sns
# Group by alpha and beta, and calculate the average RMSE, MAE, etc.
grouped_data1 = new_temp_df1.groupby(['Group', 'alpha', 'beta']).mean().reset_index()

# Loop through each group to create heatmaps for RMSE, MAE, etc.
for group in grouped_data1['Group'].unique():
    if group in ['Group1', 'Group3']:
    
        group_data1 = grouped_data1[grouped_data1['Group'] == group]
    
        # Create a pivot table for RMSE to create a heatmap
        rmse_pivot1 = group_data1.pivot('alpha', 'beta', 'MAPE')
    
        # Plot heatmap for RMSE for this group
        plt.figure(figsize=(8, 6))
        if group == 'Group1':
            sns.heatmap(rmse_pivot1, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.66, vmin=0.45)
        elif group == 'Group3':
            sns.heatmap(rmse_pivot1, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.81, vmin=0.43)

        #plt.title(f'Sensitivity of MAPE to Alpha and Beta for {group}')
        plt.xlabel('beta(β)', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylabel('alpha(α)', fontsize=25)
        plt.savefig('Seoul_Weekends_%s_MAPE_20240915.png'%(group), dpi=1024, transparent=True, bbox_inches='tight')
        plt.show()

#%%

# Group by alpha and beta, and calculate the average RMSE, MAE, etc.
grouped_data2 = new_temp_df2.groupby(['Group', 'alpha', 'beta']).mean().reset_index()

# Loop through each group to create heatmaps for RMSE, MAE, etc.
for group in grouped_data2['Group'].unique():
    if group in ['Group1', 'Group3']:
    
        group_data2 = grouped_data2[grouped_data2['Group'] == group]
    
        # Create a pivot table for RMSE to create a heatmap
        rmse_pivot2 = group_data2.pivot('alpha', 'beta', 'MAPE')
    
        # Plot heatmap for RMSE for this group
        plt.figure(figsize=(8, 6))
        if group == 'Group1':
            sns.heatmap(rmse_pivot2, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.66, vmin=0.45)
        elif group == 'Group3':
            sns.heatmap(rmse_pivot2, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.81, vmin=0.43)

        #plt.title(f'Sensitivity of MAPE to Alpha and Beta for {group}')
        plt.xlabel('beta(β)', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylabel('alpha(α)', fontsize=25)
        plt.savefig('Daejeon_Weekdays_%s_MAPE_20240915.png'%(group), dpi=1024, transparent=True, bbox_inches='tight')
        plt.show()



#%%

# Group by alpha and beta, and calculate the average RMSE, MAE, etc.
grouped_data3 = new_temp_df3.groupby(['Group', 'alpha', 'beta']).mean().reset_index()

# Loop through each group to create heatmaps for RMSE, MAE, etc.
for group in grouped_data3['Group'].unique():
    if group in ['Group1', 'Group3']:
    
        group_data3 = grouped_data3[grouped_data3['Group'] == group]
    
        # Create a pivot table for RMSE to create a heatmap
        rmse_pivot3 = group_data3.pivot('alpha', 'beta', 'MAPE')
    
        # Plot heatmap for RMSE for this group
        plt.figure(figsize=(8, 6))
        if group == 'Group1':
            sns.heatmap(rmse_pivot3, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.66, vmin=0.45)
        elif group == 'Group3':
            sns.heatmap(rmse_pivot3, annot=True, cmap="YlGnBu", fmt=".3g", annot_kws={"fontsize":20}, vmax=0.81, vmin=0.43)

        #plt.title(f'Sensitivity of MAPE to Alpha and Beta for {group}')
        plt.xlabel('beta(β)', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.ylabel('alpha(α)', fontsize=25)
        plt.savefig('Daejeon_Weekends_%s_MAPE_20240915.png'%(group), dpi=1024, transparent=True, bbox_inches='tight')
        plt.show()














