# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:26:18 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np
import ipdb
import requests
import json

import fiona

from shapely.geometry import Point, Polygon, MultiPolygon

import pyproj
from pyproj import Transformer
from shapely.ops import transform
from functools import partial
import matplotlib.pyplot as plt

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

"""point_buffer1 메소드는 안씀"""
def point_buffer1(lat: float, lon: float, radius: int):
    """
    Get the circle or square around a point with a given radius/length in meters.
    """
    standard_crs = "EPSG:4326"
    # Azimuthal equidistant projection
    aeqd_proj = "+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0"


    transformer = Transformer.from_proj(aeqd_proj.format(lat=lat, lon=lon), standard_crs, always_xy=True)

    buffer = Point(0, 0).buffer(radius)

    return transform(transformer.transform, buffer)

##


def get_pop_info(idx_dic, census_weight_dic, census_idx_dic, static_df):

    base_dict = {}

    iii=0
    for (uni_key1, uni_val1), (uni_key2, uni_val2), (uni_key3, uni_val3) in zip(idx_dic.items(), census_weight_dic.items(),
                                                                                census_idx_dic.items()):

        newa1= (static_df.loc[static_df.num_idx.isin(uni_val3.values())])[['num_idx','target']]
        newa2 = newa1.groupby('num_idx').sum()

        temp_dict1 = dict(zip(newa2.index.values, newa2['target']))
        #if iii == 21: ipdb.set_trace()
        temp_zero = 0
        for u_uni_key, u_uni_val in uni_val2.items():
            temp_zero+=temp_dict1[uni_val3[u_uni_key]] * u_uni_val
        base_dict[uni_key1] = temp_zero
        iii+=1
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

def get_one_intersect_li(shp_info, shp_index):

    one_intersect_li = []
    for xx,yy in enumerate(shp_info):
        if yy.intersection(shp_info[shp_index]):
            if not yy.equals(shp_info[shp_index]):
                one_intersect_li.append(xx)
    return one_intersect_li



#%%

if __name__ == "__main__":

    #station_address = pd.read_csv(r'Daejeon_station_info_220601.csv', usecols=['번호','위도','경도'])
    #station_address.columns = ['station_id','lat','lon']

    """아래 대여소 정보에서 Xn과 Yn이 ESPG 5179로 변환한 좌표값"""
    station_address = pd.read_csv(r'타슈 대여소정보_220601v2.csv', usecols=['ID','Xn','Yn'])

    shp_path = r'_census_data_2021_4_bnd_oa_bnd_oa_25_2021_2021/bnd_oa_25_2021_2021_4Q.shp'
    temp_shp = read_shp(shp_path)
    year_num = 2019
    #%%
    aaa2=pd.read_table(r'25_%s년_산업분류별(10차_대분류)_총괄사업체수.txt'%(year_num), sep='^',  ).fillna(0)
    aaa3=pd.read_table(r'25_%s년_산업분류별(10차_대분류)_종사자수.txt'%(year_num), sep='^',  ).fillna(0)
    aaa4 = pd.read_table(r'25_%s년_세대구성별가구.txt'%(year_num), sep='^').fillna(0)
    aaa5 = aaa4.loc[aaa4.cond=='ga_sd_001'].fillna(0).reset_index(drop=True)
    aaa6 = pd.read_table(r'25_%s년_성연령별인구.txt'%(year_num), sep='^').fillna(0)

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
    num_idx_list = []
    for xx in temp_shp:
        num_idx_list.append(int(xx['properties']['TOT_REG_CD']))

    idx_abs_rel_dict = dict(zip(num_idx_list, np.arange(len(num_idx_list))))
    idx_rel_abs_dict = dict(zip(np.arange(len(num_idx_list)),num_idx_list))


    #%% 각 정류소를 나타내는 point에 버퍼를 먹이고, 영역이 겹치는 집계구 인덱스를 산출함

    bin_dict = {}
    intersect_dict = {}
    new_bin_bin_dict = {}
    for xidx, (xx, yy) in enumerate(station_address[['Yn','Xn']].values):
        """EPSG 5179로 바꿨기 때문에, 프로젝션 없이 바로 버퍼를 먹일 수 있음"""
        #uni_buffered_point = point_buffer1(xx, yy, 500)
        uni_buffered_point = Point((yy,xx)).buffer(500)
        its_idx = [xx for xx,yy in enumerate(shp_list) if yy.intersects(uni_buffered_point)]
        bin_dict[xidx+1] = its_idx
        new_bin_list = {}
        new_bin_dict = {}
        """EPSG 5179로 바꿨기 때문에, intersection.area로 면적을 충분히 구할 수 있음"""

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
    'ID':station_address.ID.values,
    'saup':business_dict.values(),
    'jong':jong_dict.values(),
    'tot_ga':tot_ga_dict.values(),
    'single':one_ga_dict.values(),
    'tot_pop':tot_pop_dict.values(),
    '2049':tot_2049_dict.values()})
    
    
    temp_df11.loc[:,'one_ratio'] = temp_df11['single']/temp_df11['tot_ga']
    temp_df11.loc[:,'y_ratio'] = temp_df11['2049']/temp_df11['tot_pop']
    #%%
    
    #temp_df11.to_csv(r'demo_500m_2018.csv', index=False)
    #temp_df11.to_csv(r'demo_500m_2019.csv', index=False)



    #%% 2023-03-23 가구 정보가 없는 집계구에 대한 처리

    avg_target_df = aaa4.groupby('num_idx').sum()['target']
    avg_target_df1 = aaa5.groupby('num_idx').sum()['target']

    zero_ga_index = avg_target_df.loc[avg_target_df == 0].index.values
    aaa6.loc[aaa6.num_idx.isin(zero_ga_index)]
    (aaa6.loc[aaa6.num_idx.isin(zero_ga_index)]).groupby('num_idx').sum()
    mm1=(aaa6.loc[aaa6.num_idx.isin(zero_ga_index)]).groupby('num_idx').sum()
    mm1.loc[mm1.target >0]
    mm1.loc[mm1.target >0].index
    mm1.loc[mm1.target >0].index.values
    zero_ga_non_pop_index = mm1.loc[mm1.target >0].index.values
    #%% ga_df 만들어보기

#%%
    zzzz1 = []
    for xx in zero_ga_non_pop_index:
        one_intersect_li=get_one_intersect_li(shp_list, idx_abs_rel_dict[xx])

        one_sum = 0
        for one_idx in one_intersect_li:
            try:
                one_sum += avg_target_df1.loc[idx_rel_abs_dict[one_idx]]
            except:
                one_sum += 0
        zzzz1.append(one_sum/len(one_intersect_li))
    #%%

    avg_target_df.loc[zero_ga_non_pop_index] = zzzz1

    ###avg_target_df >> avg_target_df1

    bbb1 = pd.DataFrame(index=avg_target_df.index, columns=['tot_ga','one_ga'])
    bbb1.loc[avg_target_df.index, 'tot_ga'] = avg_target_df.values
    bbb1.loc[avg_target_df1.index, 'one_ga'] = avg_target_df1.values
    bbb1.loc[zero_ga_non_pop_index, 'one_ga'] =zzzz1

    bbb1

    #%% 가구수, 1인가구수를 다시 산출함


    tot_ga_dict1 = {}

    for (uni_key1, uni_val1), (uni_key2, uni_val2) in zip(intersect_dict.items(), new_bin_bin_dict.items()):

        tot_ga_dict1[uni_key1] = np.sum(bbb1.loc[uni_val2.values(), 'tot_ga'] *  list(uni_val1.values()))
    #%%

    one_ga_dict1 = {}

    for (uni_key1, uni_val1), (uni_key2, uni_val2) in zip(intersect_dict.items(), new_bin_bin_dict.items()):

        one_ga_dict1[uni_key1] = np.sum(bbb1.loc[uni_val2.values(), 'one_ga'] *  list(uni_val1.values()))

    #%%

    plt.plot(np.array(list(tot_ga_dict1.values())) - np.array(list(tot_ga_dict.values())), label='tot_ga')

    plt.plot(np.array(list(one_ga_dict1.values())) - np.array(list(one_ga_dict.values())), label='one_ga')
    plt.legend()
    plt.title('Dajeon, 2019')
    plt.ylabel('Diffrence')
    plt.xlabel('station')

    #%%
    plt.plot(np.array(list(one_ga_dict.values())) / np.array(list(tot_ga_dict.values())), label='original')

    plt.plot(np.array(list(one_ga_dict1.values())) / np.array(list(tot_ga_dict1.values())), label='adjusted')
    plt.legend()
    plt.title('Dajeon, 2019')
    plt.ylabel('one-ga ratio')
    plt.xlabel('station')
