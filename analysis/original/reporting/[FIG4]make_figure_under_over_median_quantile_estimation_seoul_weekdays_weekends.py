# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:07:06 2024

@author: dohyeon
"""


import numpy as np
import pandas as pd
import user_utils as uu
import matplotlib.pyplot as plt


def get_new_idxs2(whole_arr, thr1, thr2):
    gr1 = np.where(whole_arr<thr1)[0]
    gr2 = np.where(
        (whole_arr>=thr1)&(whole_arr<thr2))[0]
    gr3 = np.where(whole_arr>=thr2)[0]
    return gr1, gr2, gr3

def get_group_median_idxs(real_y, info_dict, group_idxs):

    under_idx = np.where( (real_y - info_dict['tst_median_pred']) > 0)[0]
    over_idx = np.where( (real_y - info_dict['tst_median_pred']) < 0)[0]

    gr1_under, gr1_over = np.intersect1d(group_idxs[0], under_idx), np.intersect1d(group_idxs[0], over_idx)
    gr2_under, gr2_over = np.intersect1d(group_idxs[1], under_idx), np.intersect1d(group_idxs[1], over_idx)
    gr3_under, gr3_over = np.intersect1d(group_idxs[2], under_idx), np.intersect1d(group_idxs[2], over_idx)

    print(gr3_over.shape)
    return dict(zip(['gr1_under','gr1_over','gr2_under','gr2_over','gr3_under','gr3_over'],
                    [gr1_under, gr1_over, gr2_under, gr2_over, gr3_under, gr3_over]))


def make_cdf_weight_dict(real_y, info_dict, group_idxs):

    grp_median_idxs = get_group_median_idxs(real_y, info_dict, group_idxs)

    cdf_weight_dict = {1:info_dict['tst_cdf_weight'][grp_median_idxs['gr1_over']],
                       2:info_dict['tst_cdf_weight'][grp_median_idxs['gr1_under']],
                       3:info_dict['tst_cdf_weight'][grp_median_idxs['gr2_over']],
                       4:info_dict['tst_cdf_weight'][grp_median_idxs['gr2_under']],
                       5:info_dict['tst_cdf_weight'][grp_median_idxs['gr3_under']]}


    ratio_dict = dict(zip(['gr1_over', 'gr2_over','gr3_over'],
                      [np.round((grp_median_idxs['gr1_over'].shape[0] / group_idxs[0].shape[0])*100,2),
                       np.round((grp_median_idxs['gr2_over'].shape[0] / group_idxs[1].shape[0])*100,2),
                       np.round((grp_median_idxs['gr3_over'].shape[0] / group_idxs[2].shape[0])*100,2)]))
    print(ratio_dict)
    return cdf_weight_dict



def make_cdf_weight_dict1(real_y, info_dict, group_idxs):

    grp_median_idxs = get_group_median_idxs(real_y, info_dict, group_idxs)

    cdf_weight_dict = {1:info_dict['tst_cdf_weight'][grp_median_idxs['gr1_over']],
                       2:info_dict['tst_cdf_weight'][grp_median_idxs['gr1_under']],
                       3:info_dict['tst_cdf_weight'][grp_median_idxs['gr2_over']],
                       4:info_dict['tst_cdf_weight'][grp_median_idxs['gr2_under']],
                       5:info_dict['tst_cdf_weight'][grp_median_idxs['gr3_over']],
                       6:info_dict['tst_cdf_weight'][grp_median_idxs['gr3_under']]}


    ratio_dict = dict(zip(['gr1_over', 'gr2_over','gr3_over'],
                      [np.round((grp_median_idxs['gr1_over'].shape[0] / group_idxs[0].shape[0])*100,2),
                       np.round((grp_median_idxs['gr2_over'].shape[0] / group_idxs[1].shape[0])*100,2),
                       np.round((grp_median_idxs['gr3_over'].shape[0] / group_idxs[2].shape[0])*100,2)]))
    print(ratio_dict)
    return cdf_weight_dict


def make_figure(cdf_weight_dict, x_labels):

    boxprops = dict(linewidth=2.0)
    medianprops = dict(linewidth=2.0)
    whiskerprops = dict(linewidth=2.0)
    capprops = dict(linewidth=2.0)


    fig1, axes1 = plt.subplots(1,1,figsize=(7,7))
    axes1.boxplot(cdf_weight_dict.values(), showfliers=False, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axes1.set_xticklabels(x_labels)
    axes1.set_yticks(np.arange(0.0,1.1,0.1))
    axes1.set_ylim(ymin=0.00, ymax=1.05)
    axes1.tick_params(axis='both', labelsize='xx-large')
    axes1.set_ylabel(r"$Q(i)$", fontsize='xx-large')

def make_figure1(cdf_weight_dict, x_labels):
    boxprops = dict(linewidth=2.0)
    medianprops = dict(linewidth=2.0)
    whiskerprops = dict(linewidth=2.0)
    capprops = dict(linewidth=2.0)


    fig1, axes1 = plt.subplots(1,1,figsize=(7,7))
    axes1.boxplot(cdf_weight_dict.values(), showfliers=False, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
    axes1.set_xticks(np.arange(1,7,1))
    axes1.set_xticklabels(x_labels)
    axes1.set_yticks(np.arange(0.0,1.1,0.1))
    axes1.set_ylim(ymin=0.00, ymax=1.05)
    axes1.tick_params(axis='both', labelsize='xx-large')
    axes1.set_ylabel(r"$Q(i)$", fontsize='xx-large')


#%%
"""seoul weekdays"""


#%% seoul, weekdays

dataset_path = r'preprocess/preprocessed_dataset_0317_weekdays.pickle'
_ , _ , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 4, 40)

weekdays_obj = uu.load_gpickle(uu.get_full_path(r'results//results//seoul', 'seoul_weekdays_model8_4_4_3.0_1.0_0.75_0.001.pickle' ))

weekdays_dict = make_cdf_weight_dict(tst_y, weekdays_obj, tst_idxs)
xticks_lab1 = ['Group1\nOver\n(75.81%)', 'Group1\nUnder\n(24.19%)','Group2\nOver\n(0.16%)', 'Group2\nUnder\n(99.84%)', 'Group3\nUnder\n(100.00%)' ]

xticks_lab1_round = ['Group1\nOver\n(75.8%)', 'Group1\nUnder\n(24.2%)','Group2\nOver\n(0.2%)', 'Group2\nUnder\n(99.8%)', 'Group3\nUnder\n(100.0%)' ]


make_figure(weekdays_dict, xticks_lab1_round)
plt.savefig('seoul_weekdays_cdf_weight_over_under_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()


#%%


"""seoul weekends"""



#%% seoul weekends


dataset_path = r'preprocess/preprocessed_dataset_0317_holy.pickle'

_ , _ , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 4, 40)

weekends_obj = uu.load_gpickle(uu.get_full_path(r'results//results//seoul', 'seoul_weekends_model8_4_4_3.5_0.75_0.75_0.01.pickle' ))

weekends_dict = make_cdf_weight_dict(tst_y, weekends_obj, tst_idxs)

xticks_lab1 = ['Group1\nOver\n(76.65%)', 'Group1\nUnder\n(23.35%)','Group2\nOver\n(0.15%)', 'Group2\nUnder\n(99.85%)', 'Group3\nUnder\n(100.00%)' ]


xticks_lab1_round = ['Group1\nOver\n(76.7%)', 'Group1\nUnder\n(23.3%)','Group2\nOver\n(0.2%)', 'Group2\nUnder\n(99.8%)', 'Group3\nUnder\n(100.0%)' ]


make_figure(weekends_dict, xticks_lab1_round)

plt.savefig('seoul_weekends_cdf_weight_over_under_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()

#%%


#%%


"""daejeon weekdays"""

#%%



dataset_path = r'preprocess/daejeon/preprocessed_dataset_0317_weekdays.pickle'

_ , _ , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 2, 10)

weekdays_obj = uu.load_gpickle(uu.get_full_path(r'results//results//daejeon', 'daejeon_weekdays_model8_4_4_2.0_1.0_0.75_0.01.pickle' ))

weekdays_dict = make_cdf_weight_dict(tst_y, weekdays_obj, tst_idxs)

xticks_lab1 = ['Group1\nOver\n(94.81%)', 'Group1\nUnder\n(5.19%)','Group2\nOver\n(1.34%)', 'Group2\nUnder\n(98.66%)', 'Group3\nUnder\n(100.00%)' ]

xticks_lab1_round = ['Group1\nOver\n(94.8%)', 'Group1\nUnder\n(5.2%)','Group2\nOver\n(1.3%)', 'Group2\nUnder\n(98.7%)', 'Group3\nUnder\n(100.0%)' ]


make_figure(weekdays_dict, xticks_lab1_round)
plt.savefig('daejeon_weekdays_cdf_weight_over_under_0416.png', dpi=1024, transparent=True)
plt.close()



#%%
#weekdays_group_idxs_dict = get_group_median_idxs(tst_y, weekdays_obj, tst_idxs)
#plt.hist(tst_y[weekdays_group_idxs_dict['gr2_over']])



#%%

"""daejeon weekends"""

#%%



dataset_path = r'preprocess/daejeon/preprocessed_dataset_0317_holy.pickle'

_ , _ , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 2, 10)

weekends_obj = uu.load_gpickle(uu.get_full_path(r'results//results//daejeon', 'daejeon_weekends_model8_4_4_2.5_1.0_0.75_0.01.pickle' ))

weekends_dict = make_cdf_weight_dict1(tst_y, weekends_obj, tst_idxs)
xticks_lab1 = ['Group1\nOver\n(95.91%)', 'Group1\nUnder\n(4.09%)','Group2\nOver\n(1.81%)', 'Group2\nUnder\n(98.19%)', 'Group3\nOver\n(0.30%)', 'Group3\nUnder\n(99.70%)' ]

xticks_lab1_round = ['Group1\nOver\n(95.9%)', 'Group1\nUnder\n(4.1%)','Group2\nOver\n(1.8%)', 'Group2\nUnder\n(98.2%)', 'Group3\nOver\n(0.3%)', 'Group3\nUnder\n(99.7%)' ]


make_figure1(weekends_dict, xticks_lab1_round)
plt.savefig('daejeon_weekends_cdf_weight_over_under_0416.png', dpi=1024, transparent=True)
plt.close()
#%%

#weekends_group_idxs_dict = get_group_median_idxs(tst_y, weekends_obj, tst_idxs)
#plt.hist(tst_y[weekends_group_idxs_dict['gr2_under']])


