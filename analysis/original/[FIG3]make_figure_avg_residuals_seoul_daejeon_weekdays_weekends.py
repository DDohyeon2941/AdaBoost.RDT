# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:27:44 2024

@author: dohyeon
"""

#%%
import numpy as np
import user_utils as uu
import matplotlib.pyplot as plt


def get_new_idxs2(whole_arr, thr1, thr2):
    gr1 = np.where(whole_arr<thr1)[0]
    gr2 = np.where(
        (whole_arr>=thr1)&(whole_arr<thr2))[0]
    gr3 = np.where(whole_arr>=thr2)[0]
    return gr1, gr2, gr3



def plot_outcounts_two1(out_count_1d, out_count_1d1, event_idxs, sf, xlab='#Iteration', ylab='Outlier Counts',ylab1='Outlier Counts',figsize=(5,5), yticks=None, yticks1=None, tl=True):
    """Boxplot for outliers"""
    fig1, axes1 = plt.subplots(1,1,figsize=figsize, tight_layout=tl)
    axes2 = axes1.twinx()
    bp1=axes1.boxplot({1:out_count_1d[event_idxs[0]],
                   2:out_count_1d[event_idxs[1]],
                   3:out_count_1d[event_idxs[2]]}.values(),showfliers=sf, positions=[0.35,1.35,2.35], widths=0.25)

    bp2=axes2.boxplot({1:out_count_1d1[event_idxs[0]],
                   2:out_count_1d1[event_idxs[1]],
                   3:out_count_1d1[event_idxs[2]]}.values(),showfliers=sf, positions=[0.65,1.65,2.65], widths=0.25)

    axes1.set_xlabel(xlab, fontsize='xx-large')
    axes1.set_ylabel(ylab, fontsize='xx-large')
    axes1.set_yticks(yticks)
    axes1.tick_params(axis='both', labelsize='xx-large')


    axes2.set_xlabel(xlab, fontsize='xx-large')
    axes2.set_ylabel(ylab1, fontsize='xx-large')
    axes2.set_yticks(yticks1)

    axes2.tick_params(axis='both', labelsize='xx-large')



    plt.setp(bp1['boxes'], color='b', linewidth=2.5)
    plt.setp(bp2['boxes'], color='R', linewidth=2.5)

    plt.setp(bp1['whiskers'], color='b', linewidth=2.5)
    plt.setp(bp2['whiskers'], color='R', linewidth=2.5)


    plt.setp(bp1['caps'], color='b', linewidth=2.5)
    plt.setp(bp2['caps'], color='R', linewidth=2.5)

    plt.setp(bp1['medians'], color='b', linewidth=2.5)
    plt.setp(bp2['medians'], color='R', linewidth=2.5)

    axes1.set_xticks([0.5,1.5,2.5])
    axes1.set_xticklabels(['Group1','Group2','Group3'], fontsize='xx-large')


    axes1.plot([], c='b', label='Real', linewidth=2.5)
    axes1.plot([], c='r', label='Pred', linewidth=2.5)
    axes1.legend(fontsize='xx-large')

    return fig1, axes1


#%% seoul, weekdays

dataset_path = r'preprocess/preprocessed_dataset_0317_weekdays.pickle'

_ , trn_y , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 4, 40)
trn_idxs = get_new_idxs2(trn_y, 4, 40)

week_obj = uu.load_gpickle(uu.get_full_path(r'results//results//seoul', 'seoul_weekdays_model8_4_4_3.0_1.0_0.75_0.001.pickle' ))
avg_resid_obj = uu.load_gpickle(r'results//results//seoul//tst_avg_real_resid.pickle')


plot_outcounts_two1(week_obj['trn_avg_real_resid'], week_obj['trn_avg_pred_resid'],  trn_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 3.00, 0.50),yticks1=np.arange(-0.10, 0.25, 0.05),figsize=(6.5,6.5))

plt.savefig('seoul_weekdays_trn_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()

plot_outcounts_two1(avg_resid_obj['weekdays'], week_obj['tst_avg_pred_resid'],  tst_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 3.75, 1.00),yticks1=np.arange(-0.10, 0.25, 0.05),figsize=(6.5,6.5))

plt.savefig('seoul_weekdays_tst_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()
#%% seoul, weekends

dataset_path = r'preprocess/preprocessed_dataset_0317_holy.pickle'

_ , trn_y , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 4, 40)
trn_idxs = get_new_idxs2(trn_y, 4, 40)

weekends_obj = uu.load_gpickle(uu.get_full_path(r'results//results//seoul', 'seoul_weekends_model8_4_4_3.5_0.75_0.75_0.01.pickle' ))
avg_resid_obj = uu.load_gpickle(r'results//results//seoul//tst_avg_real_resid.pickle')

##
plot_outcounts_two1(weekends_obj['trn_avg_real_resid'], weekends_obj['trn_avg_pred_resid'],  trn_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 3.00, 0.50),yticks1=np.arange(-0.15, 0.30, 0.05),figsize=(6.5,6.5))


plt.savefig('seoul_weekends_trn_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()

##
plot_outcounts_two1(avg_resid_obj['weekends'], weekends_obj['tst_avg_pred_resid'],  tst_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 3.75, 1.00),yticks1=np.arange(-0.15, 0.45, 0.10),figsize=(6.5,6.5))


plt.savefig('seoul_weekends_tst_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()
#%%



"""
seoul to daejeon

"""



#%% daejeon weekdays


dataset_path = r'preprocess/daejeon/preprocessed_dataset_0317_weekdays.pickle'

_ , trn_y , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 2, 10)
trn_idxs = get_new_idxs2(trn_y, 2, 10)

weekdays_obj = uu.load_gpickle(uu.get_full_path(r'results//results//daejeon', 'daejeon_weekdays_model8_4_4_2.0_1.0_0.75_0.01.pickle' ))
avg_resid_obj = uu.load_gpickle(r'results//results//daejeon//tst_avg_real_resid.pickle')


plot_outcounts_two1(weekdays_obj['trn_avg_real_resid'], weekdays_obj['trn_avg_pred_resid'],  trn_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 3.00, 0.50),yticks1=np.arange(-0.15, 0.25, 0.05),figsize=(6.5,6.5))

plt.savefig('daejeon_weekdays_trn_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()



plot_outcounts_two1(avg_resid_obj['weekdays'], weekdays_obj['tst_avg_pred_resid'],  tst_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 2.25, 0.50),yticks1=np.arange(-0.15, 0.25, 0.05),figsize=(6.5,6.5))

plt.savefig('daejeon_weekdays_tst_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()


#%% daejeon, weekends

dataset_path = r'preprocess/daejeon/preprocessed_dataset_0317_holy.pickle'

_ , trn_y , _ , tst_y = uu.load_gpickle(dataset_path)

tst_idxs = get_new_idxs2(tst_y, 2, 10)
trn_idxs = get_new_idxs2(trn_y, 2, 10)

weekends_obj = uu.load_gpickle(uu.get_full_path(r'results//results//daejeon', 'daejeon_weekends_model8_4_4_2.5_1.0_0.75_0.01.pickle' ))
avg_resid_obj = uu.load_gpickle(r'results//results//daejeon//tst_avg_real_resid.pickle')



plot_outcounts_two1(weekends_obj['trn_avg_real_resid'], weekends_obj['trn_avg_pred_resid'],  trn_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 3.00, 0.50),yticks1=np.arange(-0.15, 0.25, 0.05),figsize=(6.5,6.5))


plt.savefig('daejeon_weekends_trn_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()


plot_outcounts_two1(avg_resid_obj['weekends'], weekdays_obj['tst_avg_pred_resid'],  tst_idxs, False, "", "Real Average Residuals","Predicted Average Residuals", yticks=np.arange(-1.75, 2.25, 0.50),yticks1=np.arange(-0.15, 0.25, 0.05),figsize=(6.5,6.5))



plt.savefig('daejeon_weekends_tst_avg_0416.png', dpi=1024, transparent=True,bbox_inches='tight')
plt.close()







