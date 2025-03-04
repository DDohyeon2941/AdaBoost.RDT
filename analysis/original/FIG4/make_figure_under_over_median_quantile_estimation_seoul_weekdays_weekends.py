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



선별과정이 왜 필요하냐?
leaf node를 구성하는 분포의 평균값은 대체로 낮은 편이라고 가정, 이 경우, 대여량이 낮은 경우가 대다수를 차지하면, 대여량이 높은 경우는 이상치처럼 작동하게 되어, 대여량이 낮은쪽과 높은쪽 모두 잘 맞추지 못하게 된다. 이러한 경우를 방지하기 위해서 대여량이 낮은 경우가 대다수를 차지하면, 선형회귀모델을 학습하지 않고, 타겟값의 평균값을 예측값으로 선정한다.

그러면 모든 leaf node에서 선형회귀모델을 학습하거나, 학습하지 않을때와 비교해야 되는데....

여기서 알고 싶은 것은, 왜 굳이 선형회귀모델을 학습할 leaf node를 선정했냐? 그건 이론적으로 설명이 가능함

실험결과를 통해서 보고 싶은건? 그래서 대여량이 낮은 경우에는, 평균 대여량이 낮은 곳으로 잘 배정이 되었는지,
대여량이 높은 경우에는 평균 대여량이 높은 곳으로 잘 배정이 되었는지?


우리가 의도한 대로 모델이 학습되었느냐, 선형회귀모델이 학습된 경우는 대여량의 평균값이 비교적 높고 (대여량의 분포가 크고), 그렇지 않은 경우는 대여량의 평균값이 비교적 낮고


결국에는 예측성능을 잘 맞추기 위한 방안. 그러니까 이렇게 함으로써, group1과 extreme event에 대해서 각각 잘 맞췄냐를 봐야함
그리고 이게 테스트셋에 대해서도 잘 맞췄는지를 봐야하고
그러면 group별로 평균값을 사용해서 예측했는지, 선형회귀를 사용했는지를 통계적으로 본다면? 그리고 그렇게 했을 때


