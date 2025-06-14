# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:17:15 2022

@author: dohyeon
"""


import pandas as pd
import numpy as np
import pickle
import user_utils as uu
from datetime import date
import matplotlib.pyplot as plt
#np.seterr(all='ignore')


def get_stat_date(aaa):
    bin_list = []
    for uni_id, uni_df in aaa.groupby(['rent_id']):
        uni_row = [uni_id, uni_df.r_date.values[0],uni_df.r_date.values[-1]]
        bin_list.append(uni_row)
    return bin_list
#%%
if __name__ == "__main__":

    #데이터 불러옴
    trn_demand_path = r'..\dataset\demands\2018.pkl'
    tst_demand_path = r'..\dataset\demands\2019.pkl'

    t1 = pd.read_pickle(trn_demand_path)
    y1 = pd.read_pickle(tst_demand_path)

    #%% 대여시간(3분 이상, 4시간 이하) >> 학습데이터셋
    t1 = t1.loc[(t1.travel_time >= float(180)) & (t1.travel_time <= float(3600*4))].reset_index(drop=True)
    t1.loc[:, 'r_time'] = t1.rent_time.dt.floor('1h')
    t1.loc[:, 'r_date'] = t1.rent_time.dt.date
    t1=t1.astype({'rent_id':int})

    #%% 대여시간(3분 이상, 4시간 이하) >> 테스트 데이터셋

    y1 = y1.loc[(y1.travel_time >= float(180)) & (y1.travel_time <= float(3600*4))].reset_index(drop=True)
    y1.loc[:, 'r_time'] = y1.rent_time.dt.floor('1h')
    y1.loc[:, 'r_date'] = y1.rent_time.dt.date
    y1=y1.astype({'rent_id':int})
    #%% 1차적으로 유니크한 대여소 인덱스 산출(rent id 기준)

    trn_unique_idx = np.sort(t1.rent_id.unique())
    tst_unique_idx = np.sort(y1.rent_id.unique())
    #%% 대여일수 계산함

    trn_operation_days = 365- (t1.groupby(['r_date','rent_id']).count()['fare'].unstack().isnull()*1).sum(axis=0)

    tst_operation_days = 364-(y1.groupby(['r_date','rent_id']).count()['fare'].unstack().isnull()*1).sum(axis=0)
    #%% 대여일수가 300일 이상인 경우만 인덱싱
    trn_station_id = np.sort(trn_operation_days[trn_operation_days>=300].index.values)
    tst_station_id = np.sort(tst_operation_days[tst_operation_days>=300].index.values)


    #%% 인덱싱한 대여소에서만 return이 발생하는 경우 인덱싱

    t1 = t1.loc[t1.rent_id.isin(trn_station_id)]
    t1 = t1.astype({'return_id':int})

    t1 = t1.loc[t1.return_id.isin(trn_station_id)].reset_index(drop=True)

    y1 = y1.loc[y1.rent_id.isin(tst_station_id)]
    y1 = y1.astype({'return_id':int})

    y1 = y1.loc[y1.return_id.isin(tst_station_id)].reset_index(drop=True)

    #%% date info 생성, 정류소별 연도별 언제 운행시작, 운행 마감했는지 정보

    trn_date_df = pd.DataFrame(data=get_stat_date(t1), columns=['0', '1_x','2_x'])
    tst_date_df = pd.DataFrame(data=get_stat_date(y1), columns=['1', '1_y','2_y'])

    date_df1 = pd.concat([trn_date_df, tst_date_df,], axis=1)
    #%% 이들 중에서 지역 및 인구 정보가 있는 경우만을 실험에서 사용하기로 함
    geo_df = pd.read_csv(r'../dataset/geographic_demographic/geo_500m.csv')


    index_geo_demand = np.intersect1d(date_df1['1'].values,geo_df['ID'].values)

    #%% 재 인덱싱

    trn_station_id_final = np.intersect1d(trn_station_id, index_geo_demand)
    tst_station_id_final = np.intersect1d(tst_station_id, index_geo_demand)

    #%%재 인덱싱한 정류소 정보를 활용해 학습 및 테스트 데이터 재 인덱싱



    t1 = t1.loc[t1.rent_id.isin(trn_station_id_final)]

    t1 = t1.loc[t1.return_id.isin(trn_station_id_final)].reset_index(drop=True)

    y1 = y1.loc[y1.rent_id.isin(tst_station_id_final)]
    y1 = y1.astype({'return_id':int})

    y1 = y1.loc[y1.return_id.isin(tst_station_id_final)].reset_index(drop=True)


    #%%  date info2 생성
    trn_date_df1 = pd.DataFrame(data=get_stat_date(t1), columns=['0', '1_x','2_x'])
    tst_date_df1 = pd.DataFrame(data=get_stat_date(y1), columns=['1', '1_y','2_y'])

    date_df2 = pd.concat([trn_date_df1, tst_date_df1], axis=1)
    #%%

    #%%
    date_df2.to_csv(r'dateinfo_0317.csv', index=False)
    t1.reset_index(drop=True).to_pickle(r'2018_sel_demands_0317.pkl')
    y1.reset_index(drop=True).to_pickle(r'2019_sel_demands_0317.pkl')

    #%% histogram of operation duration





