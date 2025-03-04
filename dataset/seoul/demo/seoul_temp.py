# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:05:12 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np

import requests
import json

import fiona

from shapely.geometry import Point, Polygon

import pyproj
from pyproj import Transformer
from shapely.ops import transform
from functools import partial

"""본 모듈에서 가장 중요한 부분은, 자전거 대여소 및 집계구에 대한 좌표정보를
EPSG: 5179로 바꾼후에 진행해야 된다는 것임

일반적으로 EPSG 4326을 많이 쓰는데, 이렇게 하면 buffer나 intersection.area 처럼, 면적을 구하는 연산에서 좌표적인 특성(위도와 경도가 구분되어야함)을 반영하지 못함
"""

def read_shp(shp_path):
    """single f"""
    f_list = []
    with fiona.open(shp_path) as src:
        for f in src:
            f_list.append(f)
    return f_list

def get_pop_info(idx_dic, census_weight_dic, census_idx_dic, static_df):

    base_dict = {}
    for (uni_key1, uni_val1), (uni_key2, uni_val2), (uni_key3, uni_val3) in zip(idx_dic.items(), census_weight_dic.items(),
                                                                                census_idx_dic.items()):

        newa1= (static_df.loc[static_df.num_idx.isin(uni_val3.values())])[['num_idx','target']]
        newa2 = newa1.groupby('num_idx').sum()

        temp_dict1 = dict(zip(newa2.index.values, newa2['target']))
    
        temp_zero = 0
        for u_uni_key, u_uni_val in uni_val2.items():
            temp_zero+=temp_dict1[uni_val3[u_uni_key]] * u_uni_val
        base_dict[uni_key1] = temp_zero
    return base_dict

def get_pop_info1(idx_dic, census_weight_dic, census_idx_dic, static_df):

    base_dict = {}
    for (uni_key1, uni_val1), (uni_key2, uni_val2), (uni_key3, uni_val3) in zip(idx_dic.items(), census_weight_dic.items(),
                                                                                census_idx_dic.items()):

        newa1= (static_df.loc[static_df.num_idx.isin(uni_val3.values())])[['num_idx','target']]
        newa2 = newa1.groupby('num_idx').sum()

        temp_dict1 = dict(zip(newa2.index.values, newa2['target']))
    
        temp_zero = 0
        for u_uni_key, u_uni_val in uni_val2.items():
            try:
                temp_zero+=temp_dict1[uni_val3[u_uni_key]] * u_uni_val
            except:
                temp_zero +=0
        base_dict[uni_key1] = temp_zero
    return base_dict

#%%

if __name__ == "__main__":
    temp_shp = read_shp(r'_census_data_2021_4_bnd_oa_bnd_oa_11_2021_2021\bnd_oa_11_2021_2021_4Q.shp')
    
    seoul_bike_address = pd.read_csv(r'대여소정보_20210128.csv')

    #%%
    year_num = 2018
    #%%
    aaa2=pd.read_table(r'seoul_census_dataset\11_%s년_산업분류별(10차_대분류)_총괄사업체수.txt'%(year_num), sep='^',  ).fillna(0)
    aaa3=pd.read_table(r'seoul_census_dataset\11_%s년_산업분류별(10차_대분류)_종사자수.txt'%(year_num), sep='^',  ).fillna(0)
    aaa4 = pd.read_table(r'seoul_census_dataset\11_%s년_세대구성별가구.txt'%(year_num), sep='^').fillna(0)
    aaa5 = aaa4.loc[aaa4.cond=='ga_sd_001'].fillna(0).reset_index(drop=True)
    aaa6 = pd.read_table(r'seoul_census_dataset\11_%s년_성연령별인구.txt'%(year_num), sep='^').fillna(0)
    
    tot_2049_codes = ['in_age_035', 'in_age_036', 'in_age_037', 'in_age_038'] + ['in_age_065', 'in_age_066', 'in_age_067', 'in_age_068']
    aaa7 = aaa6.loc[aaa6.cond.isin(tot_2049_codes)].fillna(0).reset_index(drop=True)
    
    #%%
    
    
    shp_list = []
    for xx in temp_shp:
        try:
            shp_list.append(Polygon(xx['geometry']['coordinates'][0]))
        except:
            shp_list.append(Polygon(xx['geometry']['coordinates']))

#%%

    bin_dict = {}
    intersect_dict = {}
    new_bin_bin_dict = {}
    for xidx, (xx, yy) in enumerate(seoul_bike_address[['Yn','Xn']].values):
        """EPSG 5179로 바꿔서 버퍼를 바로 먹일 수 있"""
        #uni_buffered_point = point_buffer1(xx, yy, 500)
        uni_buffered_point = Point((yy,xx)).buffer(500)
        its_idx = [xx for xx,yy in enumerate(shp_list) if yy.intersects(uni_buffered_point)]
        bin_dict[xidx+1] = its_idx
        new_bin_list = {}
        new_bin_dict = {}
        """EPSG 5179로 바꿔서 area를 바로 산출하 수 있음"""

        for uni_its in its_idx:
            new_bin_list[uni_its] = (shp_list[uni_its].buffer(0).intersection(uni_buffered_point)).area / (shp_list[uni_its].area)
            new_bin_dict[uni_its] = int(temp_shp[uni_its]['properties']['TOT_REG_CD'])
        new_bin_bin_dict[xidx+1] = new_bin_dict

        intersect_dict[xidx+1] = new_bin_list
    #%%
    business_dict = get_pop_info(bin_dict, intersect_dict, new_bin_bin_dict, aaa2)
    jong_dict = get_pop_info(bin_dict, intersect_dict, new_bin_bin_dict, aaa3)
    tot_ga_dict = get_pop_info(bin_dict, intersect_dict, new_bin_bin_dict, aaa4)
    
    one_ga_dict =get_pop_info1(bin_dict, intersect_dict, new_bin_bin_dict, aaa5)
    tot_pop_dict = get_pop_info(bin_dict, intersect_dict, new_bin_bin_dict, aaa6)
    tot_2049_dict = get_pop_info1(bin_dict, intersect_dict, new_bin_bin_dict, aaa7)


    #%%
    temp_df11 = pd.DataFrame(data={
    'ID':seoul_bike_address.ID.values,
    'saup':business_dict.values(),
    'jong':jong_dict.values(),
    'tot_ga':tot_ga_dict.values(),
    'single':one_ga_dict.values(),
    'tot_pop':tot_pop_dict.values(),
    '2049':tot_2049_dict.values()})
    
    
    temp_df11.loc[:,'one_ratio'] = temp_df11['single']/temp_df11['tot_ga']
    temp_df11.loc[:,'y_ratio'] = temp_df11['2049']/temp_df11['tot_pop']

    #%%
    #temp_df11.to_csv(r'demo_2018_seoul.csv', index=False)

    #temp_df11.to_csv(r'demo_2019_seoul.csv', index=False)

    #%%




    avg_target_df = aaa4.groupby('num_idx').mean()['target']
    avg_target_df1 = aaa5.groupby('num_idx').mean()['target']
    zero_ga_index = avg_target_df.loc[avg_target_df == 0].index.values
    mm1=(aaa6.loc[aaa6.num_idx.isin(zero_ga_index)]).groupby('num_idx').sum()
    zero_ga_non_pop_index = mm1.loc[mm1.target >0].index.values



