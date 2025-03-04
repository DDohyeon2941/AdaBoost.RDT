# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:43:35 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np

"""대전의 경우, 2019년 22번 ID의 정보는 삭제해야함"""


#%%
if __name__ == "__main__":
    """서울 전체"""


    train_pkl, test_pkl = pd.read_pickle(r'preprocess/2018_total_0317.pkl'), pd.read_pickle(r'preprocess/2019_total_0317.pkl')
    train_pkl = train_pkl.reset_index()
    test_pkl = test_pkl.reset_index()


    print(train_pkl[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())
    print(test_pkl[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())

    #%%
    """대전 전체"""

    daejeon_train_pkl, daejeon_test_pkl = pd.read_pickle(r'tashu/2018_total_0317.pkl'), pd.read_pickle(r'tashu/2019_total_0317.pkl')


    daejeon_train_pkl = daejeon_train_pkl.reset_index()
    daejeon_test_pkl = daejeon_test_pkl.reset_index()
    daejeon_test_pkl = daejeon_test_pkl.loc[daejeon_test_pkl.ID != 22].reset_index(drop=True)


    print(daejeon_train_pkl[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())
    print(daejeon_test_pkl[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())



    #%%
    """서울 주중"""

    seoul_week_trn_df = train_pkl.loc[((train_pkl['isholy'] < 1) & (train_pkl['dow'].isin([0,1,2,3,4])))].reset_index(drop=True)
    seoul_week_tst_df = test_pkl.loc[((test_pkl['isholy'] < 1) & (test_pkl['dow'].isin([0,1,2,3,4])))].reset_index(drop=True)


    print(seoul_week_trn_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())
    print(seoul_week_tst_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())


    #%%
    """서울 주말"""

    seoul_holy_trn_df = train_pkl.loc[((train_pkl['isholy'] >= 1) | (train_pkl['dow'].isin([5,6])))].reset_index(drop=True)
    seoul_holy_tst_df = test_pkl.loc[((test_pkl['isholy'] >= 1) | (test_pkl['dow'].isin([5,6])))].reset_index(drop=True)


    print(seoul_holy_trn_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())
    print(seoul_holy_tst_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())

    #%%
    """대전 주중"""

    daejeon_week_trn_df = daejeon_train_pkl.loc[((daejeon_train_pkl['isholy'] < 1) & (daejeon_train_pkl['dow'].isin([0,1,2,3,4])))].reset_index(drop=True)
    daejeon_week_tst_df = daejeon_test_pkl.loc[((daejeon_test_pkl['isholy'] < 1) & (daejeon_test_pkl['dow'].isin([0,1,2,3,4])))].reset_index(drop=True)


    print(daejeon_week_trn_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())
    print(daejeon_week_tst_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())

    #%%
    """대전 주말"""

    daejeon_holy_trn_df = daejeon_train_pkl.loc[((daejeon_train_pkl['isholy'] >= 1) | (daejeon_train_pkl['dow'].isin([5,6])))].reset_index(drop=True)
    daejeon_holy_tst_df = daejeon_test_pkl.loc[((daejeon_test_pkl['isholy'] >= 1) | (daejeon_test_pkl['dow'].isin([5,6])))].reset_index(drop=True)


    print(daejeon_holy_trn_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())
    print(daejeon_holy_tst_df[['ID','r_date','y']].groupby(['ID','r_date']).sum().mean())














