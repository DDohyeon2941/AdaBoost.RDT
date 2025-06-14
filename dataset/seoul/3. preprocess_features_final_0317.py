# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:17:15 2022

@author: dohyeon
"""

from datetime import date
import holidays
import pandas as pd
import numpy as np
import user_utils as uu


def grouping_hours(uni_shour):
    gr1 = [7, 8, 9]
    gr2 = [10, 11, 12, 13, 14, 15, 16]
    gr3 = [17, 18, 19]
    gr4 = [20, 21, 22, 23]
    gr5 = [0, 1, 2, 3, 4, 5, 6]

    output = 0
    if uni_shour in gr1:
        output = 1
    elif uni_shour in gr2:
        output = 2
    elif uni_shour in gr3:
        output = 3
    elif uni_shour in gr4:
        output = 4
    elif uni_shour in gr5:
        output = 5
    return output

def grouping_seasons(uni_month):
    gr1 = [1, 2, 12]
    gr2 = [3, 4, 5]
    gr3 = [6, 7, 8]
    gr4 = [9, 10, 11]

    if uni_month in gr1:
        output = 1
    elif uni_month in gr2:
        output = 2
    elif uni_month in gr3:
        output = 3
    elif uni_month in gr4:
        output = 4
    return output
#%% 데이터 읽어오기
if __name__ == "__main__":

    p1 = pd.read_csv(r'weather_2018.csv')
    p1 = p1.rename(columns={'rain_log2':'log_rain2'})



    p2 = pd.read_csv(r'weather_2019.csv')
    p2 = p2.rename(columns={'rain_log2':'log_rain2', 'huminity':'humidity'})


    #%% 날짜정보 생성, 학습데이터, 테스트 데이터
    trn_holys = [xx.date() for xx in  pd.date_range(start="20180101", end="20190101", freq='d')[:-1] if xx in holidays.KR()]
    p1.loc[:, 'date'] = pd.to_datetime(p1.date)
    p1.loc[:, 'isholy'] = pd.to_datetime(p1.date.dt.date).isin(trn_holys)*1
    p1.loc[:,'season']=[grouping_seasons(xx) for xx in p1.date.dt.month]
    p1.loc[:,'tmask']=[grouping_hours(xx) for xx in p1.date.dt.hour]
    p1.loc[:, 'dow'] = p1.date.dt.dayofweek

    tst_holys = [xx.date() for xx in  pd.date_range(start="20190101", end="20200101", freq='d')[:-1] if xx in holidays.KR()]
    p2.loc[:, 'date'] = pd.to_datetime(p2.date)
    p2.loc[:, 'isholy'] = pd.to_datetime(p2.date.dt.date).isin(tst_holys)*1
    p2.loc[:,'season']=[grouping_seasons(xx) for xx in p2.date.dt.month]
    p2.loc[:,'tmask']=[grouping_hours(xx) for xx in p2.date.dt.hour]
    p2.loc[:, 'dow'] = p2.date.dt.dayofweek

    #%%

    demo_path = r'..\dataset\geographic_demographic\demo_2018_seoul.csv'
    geo_path = r'..\dataset\geographic_demographic\geo_500m.csv'
    geo_path1 = r'..\dataset\geographic_demographic\geo_250m.csv'
    d1, g1 = pd.read_csv(demo_path), pd.read_csv(geo_path)
    g2=pd.read_csv(geo_path1)

    g1.loc[:,'uni_dist'] = g2.uni_dist

    date_df = pd.read_csv(r'dateinfo_0317.csv')
    dg1 = pd.merge(left=d1, right=g1, left_on='ID', right_on='ID')

    new_dg = dg1.loc[dg1.ID.isin(date_df['1'])].reset_index(drop=True)
    new_dg

    #%%
    p1.loc[:, 'date'] = pd.to_datetime(p1.date)
    p1.loc[:, 'r_date'] = pd.to_datetime(p1.date.dt.date)


    aa1=new_dg.iloc[np.tile(np.arange(len(new_dg)), p1.shape[0])].reset_index(drop=True)
    aa2=p1.iloc[np.repeat(np.arange(len(p1)), new_dg.shape[0])].reset_index(drop=True)

    aa3=pd.concat([aa1,aa2],axis=1)
    aa3.columns
    #%% 2018

    date_df_trn = date_df[['0','1_x','2_x']].dropna().astype({'0':int})
    date_df_trn.loc[:,'1_x'] = pd.to_datetime(date_df_trn['1_x'])
    date_aa=date_df_trn.loc[date_df_trn['1_x'] > date_df_trn['1_x'][0]]

    #date_df.loc[:,'1_x'] = pd.to_datetime(date_df['1_x'])
    #date_aa=date_df.loc[date_df['1_x'] > date_df['1_x'][0]]
    #date_aa
    aa3 = aa3.loc[aa3.ID.isin(date_df_trn['0'])].reset_index(drop=True)

    del_list = np.array([])
    for uni_stat, uni_date in zip(date_aa['0'],date_aa['1_x']):
        del_list = np.append(del_list,aa3.loc[(aa3.ID==uni_stat)&(aa3.r_date<uni_date)].index.values)

    #del_list.shape

    (aa3.loc[np.setdiff1d(aa3.index.values,np.sort(del_list))]).reset_index(drop=True).to_pickle(r'2018_features_0317.pkl')

    #%% 2019

    demo_path = r'..\dataset\geographic_demographic\demo_2019_seoul.csv'
    geo_path = r'..\dataset\geographic_demographic\geo_500m.csv'
    geo_path1 = r'..\dataset\geographic_demographic\geo_250m.csv'
    d1, g1 = pd.read_csv(demo_path), pd.read_csv(geo_path)
    g2=pd.read_csv(geo_path1)

    g1.loc[:,'uni_dist'] = g2.uni_dist


    date_df = pd.read_csv(r'dateinfo_0317.csv')
    dg1 = pd.merge(left=d1, right=g1, left_on='ID', right_on='ID')

    new_dg = dg1.loc[dg1.ID.isin(date_df['1'])].reset_index(drop=True)
    new_dg

    p2.loc[:, 'date'] = pd.to_datetime(p2.date)
    p2.loc[:, 'r_date'] = pd.to_datetime(p2.date.dt.date)

    aa1=new_dg.iloc[np.tile(np.arange(len(new_dg)), p2.shape[0])].reset_index(drop=True)
    aa2=p2.iloc[np.repeat(np.arange(len(p2)), new_dg.shape[0])].reset_index(drop=True)
    aa3=pd.concat([aa1,aa2],axis=1)

    date_df_tst = date_df[['1','1_y','2_y']]

    date_df_tst.loc[:,'2_y'] = pd.to_datetime(date_df_tst['2_y'])
    date_aa=date_df_tst.loc[date_df_tst['2_y'] < date_df_tst['2_y'][0]]
    date_aa


    del_list = np.array([])
    for uni_stat, uni_date in zip(date_aa['1'],date_aa['2_y']):
        del_list = np.append(del_list,aa3.loc[(aa3.ID==uni_stat)&(aa3.r_date>uni_date)].index.values)

    (aa3.loc[np.setdiff1d(aa3.index.values,np.sort(del_list))]).reset_index(drop=True).to_pickle(r'2019_features_0317.pkl')
