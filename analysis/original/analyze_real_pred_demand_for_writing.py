# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:25:47 2024

@author: dohyeon
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sd_df = pd.read_csv(r'seoul_weekdays_real_pred_prop_median_comparison_demand.csv')
se_df = pd.read_csv(r'seoul_weekends_real_pred_prop_median_comparison_demand.csv')

dd_df = pd.read_csv(r'daejeon_weekdays_real_pred_prop_median_comparison_demand.csv')
de_df = pd.read_csv(r'daejeon_weekends_real_pred_prop_median_comparison_demand.csv')



"""Group1에 대해서 성능이 낮은 이유를 찾음"""

#%%
sd_df.loc[(sd_df.real==0)&(sd_df.pred<=0.5)]
sd_df.loc[(sd_df.real==0)&(sd_df.median_pred<=0.5)]

## ratio of correctly predicted case
sd_df.loc[(sd_df.real==0)&(sd_df.pred<=0.5)].shape[0] / sd_df.loc[(sd_df.real==0)].shape[0]
sd_df.loc[(sd_df.real==0)&(sd_df.median_pred<=0.5)].shape[0] / sd_df.loc[(sd_df.real==0)].shape[0]


#%%
sd_df.loc[(sd_df.real==1)]

sd_df.loc[(sd_df.real==1)&((sd_df.pred>=0.5)&(sd_df.pred<1.5))]
sd_df.loc[(sd_df.real==1)&((sd_df.median_pred>=0.5)&(sd_df.median_pred<1.5))]

## ratio of correctly predicted case
sd_df.loc[(sd_df.real==1)&((sd_df.pred>=0.5)&(sd_df.pred<1.5))].shape[0] / sd_df.loc[(sd_df.real==1)].shape[0]
sd_df.loc[(sd_df.real==1)&((sd_df.median_pred>=0.5)&(sd_df.median_pred<1.5))].shape[0] / sd_df.loc[(sd_df.real==1)].shape[0]


#%%

sd_df.loc[(sd_df.real==2)&((sd_df.pred>=1.5)&(sd_df.pred<2.5))]
sd_df.loc[(sd_df.real==2)&((sd_df.median_pred>=1.5)&(sd_df.median_pred<2.5))]

## ratio of correctly predicted case
sd_df.loc[(sd_df.real==2)&((sd_df.pred>=1.5)&(sd_df.pred<2.5))].shape[0] / sd_df.loc[(sd_df.real==2)].shape[0]
sd_df.loc[(sd_df.real==2)&((sd_df.median_pred>=1.5)&(sd_df.median_pred<2.5))].shape[0] / sd_df.loc[(sd_df.real==2)].shape[0]


#%%

sd_df.loc[(sd_df.real==3)&((sd_df.pred>=2.5)&(sd_df.pred<3.5))]
sd_df.loc[(sd_df.real==3)&((sd_df.median_pred>=2.5)&(sd_df.median_pred<3.5))]


## ratio

sd_df.loc[(sd_df.real==3)&((sd_df.pred>=2.5)&(sd_df.pred<3.5))].shape[0] / sd_df.loc[(sd_df.real==3)].shape[0]
sd_df.loc[(sd_df.real==3)&((sd_df.median_pred>=2.5)&(sd_df.median_pred<3.5))].shape[0] /sd_df.loc[(sd_df.real==3)].shape[0]



#%%

se_df.loc[(se_df.real==0)&(se_df.pred<=0.5)]
se_df.loc[(se_df.real==0)&(se_df.median_pred<=0.5)]

## ratio of correctly predicted case
se_df.loc[(se_df.real==0)&(se_df.pred<=0.5)].shape[0] / se_df.loc[(se_df.real==0)].shape[0]
se_df.loc[(se_df.real==0)&(se_df.median_pred<=0.5)].shape[0] / se_df.loc[(se_df.real==0)].shape[0]


#%%

se_df.loc[(se_df.real==1)&((se_df.pred>=0.5)&(se_df.pred<=1.5))]
se_df.loc[(se_df.real==1)&((se_df.median_pred>=0.5)&(se_df.median_pred<=1.5))]

## ratio of correctly predicted case

se_df.loc[(se_df.real==1)&((se_df.pred>=0.5)&(se_df.pred<=1.5))].shape[0] / se_df.loc[(se_df.real==1)].shape[0]
se_df.loc[(se_df.real==1)&((se_df.median_pred>=0.5)&(se_df.median_pred<=1.5))].shape[0] / se_df.loc[(se_df.real==1)].shape[0]



#%%

se_df.loc[(se_df.real==2)&((se_df.pred>=1.5)&(se_df.pred<=2.5))]
se_df.loc[(se_df.real==2)&((se_df.median_pred>=1.5)&(se_df.median_pred<=2.5))]

## ratio of correctly predicted case
se_df.loc[(se_df.real==2)&((se_df.pred>=1.5)&(se_df.pred<=2.5))].shape[0] / se_df.loc[(se_df.real==2)].shape[0]
se_df.loc[(se_df.real==2)&((se_df.median_pred>=1.5)&(se_df.median_pred<=2.5))].shape[0] / se_df.loc[(se_df.real==2)].shape[0]


#%%
se_df.loc[(se_df.real==3)&((se_df.pred>=2.5)&(se_df.pred<=3.5))]
se_df.loc[(se_df.real==3)&((se_df.median_pred>=2.5)&(se_df.median_pred<=3.5))]

## ratio

se_df.loc[(se_df.real==3)&((se_df.pred>=2.5)&(se_df.pred<=3.5))].shape[0] / se_df.loc[(se_df.real==3)].shape[0]
se_df.loc[(se_df.real==3)&((se_df.median_pred>=2.5)&(se_df.median_pred<=3.5))].shape[0] / se_df.loc[(se_df.real==3)].shape[0]



#%%
"""daejeon"""
#%%


dd_df.loc[(dd_df.real==0)&(dd_df.pred<=0.5)]
dd_df.loc[(dd_df.real==0)&(dd_df.median_pred<=0.5)]


## ratio of correctly predicted case
dd_df.loc[(dd_df.real==0)&(dd_df.pred<=0.5)].shape[0] / dd_df.loc[(dd_df.real==0)].shape[0]
dd_df.loc[(dd_df.real==0)&(dd_df.median_pred<=0.5)].shape[0] / dd_df.loc[(dd_df.real==0)].shape[0]

#%%

dd_df.loc[(dd_df.real==1)&((dd_df.pred>=0.5)&(dd_df.pred<=1.5))]
dd_df.loc[(dd_df.real==1)&((dd_df.median_pred>=0.5)&(dd_df.median_pred<=1.5))]

## ratio of correctly predicted case
dd_df.loc[(dd_df.real==1)&((dd_df.pred>=0.5)&(dd_df.pred<=1.5))].shape[0] / dd_df.loc[(dd_df.real==1)].shape[0]
dd_df.loc[(dd_df.real==1)&((dd_df.median_pred>=0.5)&(dd_df.median_pred<=1.5))].shape[0] / dd_df.loc[(dd_df.real==1)].shape[0]


#%%

de_df.loc[(de_df.real==0)&(de_df.pred<=0.5)]
de_df.loc[(de_df.real==0)&(de_df.median_pred<=0.5)]

## ratio of correctly predicted case
de_df.loc[(de_df.real==0)&(de_df.pred<=0.5)].shape[0] / de_df.loc[(de_df.real==0)].shape[0]
de_df.loc[(de_df.real==0)&(de_df.median_pred<=0.5)].shape[0] / de_df.loc[(de_df.real==0)].shape[0]

#%%

de_df.loc[(de_df.real==1)&((de_df.pred>=0.5)&(de_df.pred<=1.5))]
de_df.loc[(de_df.real==1)&((de_df.median_pred>=0.5)&(de_df.median_pred<=1.5))]

## ratio of correctly predicted case
de_df.loc[(de_df.real==1)&((de_df.pred>=0.5)&(de_df.pred<=1.5))].shape[0] / de_df.loc[(de_df.real==1)].shape[0]
de_df.loc[(de_df.real==1)&((de_df.median_pred>=0.5)&(de_df.median_pred<=1.5))].shape[0] / de_df.loc[(de_df.real==1)].shape[0]


#%%

plt.hist(de_df.loc[(de_df.real==0)].pred)
plt.hist(de_df.loc[(de_df.real==0)].median_pred)

plt.hist(de_df.loc[(dd_df.real==0)].pred)
plt.hist(de_df.loc[(dd_df.real==0)].median_pred)

#%%

plt.hist(de_df.loc[(de_df.real==1)].pred)
plt.hist(de_df.loc[(de_df.real==1)].median_pred)

plt.hist(de_df.loc[(dd_df.real==1)].pred)
plt.hist(de_df.loc[(dd_df.real==1)].median_pred)

#%%

sd_df.loc[(sd_df.real==3)].pred.max()
sd_df.loc[(sd_df.real==3)].median_pred.max()

se_df.loc[(se_df.real==3)].pred.max()
se_df.loc[(se_df.real==3)].median_pred.max()


#%% daejeon

dd_df.loc[(dd_df.real==0)].pred.max()
dd_df.loc[(dd_df.real==0)].median_pred.max()

de_df.loc[(de_df.real==0)].pred.max()
de_df.loc[(de_df.real==0)].median_pred.max()



#%%
"""group2와 3에서 성능이 개선된 이유를 찾음"""

sd_df.loc[(sd_df.real>=4)&((sd_df.real<40))]


sd_df.loc[sd_df.real>=40].real - sd_df.loc[sd_df.real>=40].pred




plt.plot(sd_df.loc[sd_df.real>=40].real.values, c='b')
plt.plot(sd_df.loc[sd_df.real>=40].pred.values, c='r')
plt.plot(sd_df.loc[sd_df.real>=40].median_pred.values, c='g')

plt.hist(sd_df.loc[sd_df.real>=40].pred.values)

#%%


plt.plot(sd_df.loc[(sd_df.real>=20)&((sd_df.real<40))].real)
plt.plot(sd_df.loc[(sd_df.real>=20)&((sd_df.real<40))].pred)
plt.plot(sd_df.loc[(sd_df.real>=20)&((sd_df.real<40))].median_pred)

#%%


#%% seoul, low demand

(np.abs(sd_df.loc[(sd_df.real>=4)&((sd_df.real<10))].real - sd_df.loc[(sd_df.real>=4)&((sd_df.real<10))].pred)<1).sum() / sd_df.loc[(sd_df.real>=4)&((sd_df.real<10))].shape[0]

(np.abs(se_df.loc[(se_df.real>=4)&((se_df.real<10))].real - se_df.loc[(se_df.real>=4)&((se_df.real<10))].median_pred)<1).sum() / se_df.loc[(se_df.real>=4)&((se_df.real<10))].shape[0]


#%% seoul, high demand

(np.abs(sd_df.loc[(sd_df.real>=10)&((sd_df.real<40))].real - sd_df.loc[(sd_df.real>=10)&((sd_df.real<40))].pred)<1).sum()

(np.abs(se_df.loc[(se_df.real>=10)&((se_df.real<40))].real - se_df.loc[(se_df.real>=10)&((se_df.real<40))].pred)<1).sum()



#%% daejeon, low demand

(np.abs(dd_df.loc[(dd_df.real>=2)&((dd_df.real<5))].real - dd_df.loc[(dd_df.real>=2)&((dd_df.real<5))].median_pred)<1).sum() / dd_df.loc[(dd_df.real>=2)&((dd_df.real<5))].shape[0]


(np.abs(de_df.loc[(de_df.real>=2)&((de_df.real<5))].real - de_df.loc[(de_df.real>=2)&((de_df.real<5))].pred)<1).sum() / de_df.loc[(de_df.real>=2)&((de_df.real<5))].shape[0]

#%% daejeon, high demand

(np.abs(dd_df.loc[(dd_df.real>=5)&((dd_df.real<10))].real - dd_df.loc[(dd_df.real>=5)&((dd_df.real<10))].median_pred)<1).sum()


(np.abs(de_df.loc[(de_df.real>=5)&((de_df.real<10))].real - de_df.loc[(de_df.real>=5)&((de_df.real<10))].median_pred)<1).sum()




#%%
sd_df.loc[(sd_df.real>=4)&((sd_df.real<10))].shape[0] / sd_df.loc[(sd_df.real>=4)&((sd_df.real<40))].shape[0]
se_df.loc[(se_df.real>=4)&((se_df.real<10))].shape[0] / se_df.loc[(se_df.real>=4)&((se_df.real<40))].shape[0]


sd_df.loc[(sd_df.real>=10)&((sd_df.real<20))].shape[0] / sd_df.loc[(sd_df.real>=4)&((sd_df.real<40))].shape[0]
se_df.loc[(se_df.real>=10)&((se_df.real<20))].shape[0] / se_df.loc[(se_df.real>=4)&((se_df.real<40))].shape[0]


sd_df.loc[(sd_df.real>=20)&((sd_df.real<40))].shape[0] / sd_df.loc[(sd_df.real>=4)&((sd_df.real<40))].shape[0]
se_df.loc[(se_df.real>=20)&((se_df.real<40))].shape[0] / se_df.loc[(se_df.real>=4)&((se_df.real<40))].shape[0]

#%%

dd_df.loc[(dd_df.real>=2)&((dd_df.real<5))].shape[0] / dd_df.loc[(dd_df.real>=2)&((dd_df.real<10))].shape[0]
de_df.loc[(de_df.real>=2)&((de_df.real<5))].shape[0] / de_df.loc[(de_df.real>=2)&((de_df.real<10))].shape[0]


dd_df.loc[(dd_df.real>=5)&((dd_df.real<10))].shape[0] / dd_df.loc[(dd_df.real>=2)&((dd_df.real<10))].shape[0]
de_df.loc[(de_df.real>=5)&((de_df.real<10))].shape[0] / de_df.loc[(de_df.real>=2)&((de_df.real<10))].shape[0]


