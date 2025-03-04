# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:26:18 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np

import requests
import json
from haversine import haversine



def load_private_key():
    # 받은 rest_api를 쓰면 됨
    rest_api = '63e2b38e5cd98fb09aeba2634f72d6db' # api 유출하면안되는..?!?!
    headers = ({'Authorization' : "KakaoAK "+rest_api})
    return rest_api, headers



def getLatLng(address):
    result = ""
 
    url = 'https://dapi.kakao.com/v2/local/search/address.json?query=' + address
    rest_api_key = '63e2b38e5cd98fb09aeba2634f72d6db'
    header = {'Authorization': 'KakaoAK ' + rest_api_key}
 
    r = requests.get(url, headers=header)
 
    if r.status_code == 200:
        result_address = r.json()["documents"][0]["address"]
        
        result = result_address["y"], result_address["x"]
    else:
        result = "ERROR[" + str(r.status_code) + "]"
    
    return result


    
def get_cate_size_uni(lon, lat, radius, cate_code, cate_query, url, headers):
    params = {'query' : cate_query, 'x' : lon, 'y' : lat, 'radius' : radius, 'category_group_code' : cate_code}
    total = requests.get(url, params=params, headers=headers).json()
    return total['meta']['total_count']

def get_cate_size_main(station_address_df, radius, cate_code, cate_query, url, header):

    bin_list = []
    for xx, yy in station_address_df[['lon','lat']].values:
        bin_list.append(get_cate_size_uni(xx, yy, radius, cate_code, cate_query, url, headers))
    return bin_list



#%%

if __name__ == "__main__":

    RADIUS = 500
    BIKE_NUM = 262 #타슈1, 이 이후로는 타슈2

    university_address = pd.read_csv(r'Daejeon_University_Address.csv')
    
    university_address.loc[:,'coef'] = [getLatLng(xx) for xx in university_address['주소']]
    station_address = pd.read_csv(r'Daejeon_station_info_220601.csv', usecols=['번호','위도','경도'])
    station_address.columns = ['station_id','lat','lon']
    

    #%% 대여소별 가장 가까운 대학교와의 거리
    
    xx_list = []
    for xx in station_address[['lat','lon']].values:
        yy_list = []
        for yy in university_address['coef']:
            yy_list.append(haversine(tuple(xx),[float(xxx) for xxx in  yy], unit ='m'))
        xx_list.append(yy_list)
    
    
    dist_arr_2d = np.array(xx_list)
    
    
    #uni_dist = np.min(dist_arr_2d, axis=1).tolist()
    station_address.loc[:, 'uni_dist'] = np.min(dist_arr_2d, axis=1)
    
    #%% 대여소별 다른 대여소와의 거리
    
    
    xx_list = []
    for xx in station_address[['lat','lon']].values:
        yy_list = []
        for yy in station_address[['lat','lon']].values[:BIKE_NUM]:
            yy_list.append(haversine(tuple(xx),tuple(yy), unit ='m'))
        xx_list.append(yy_list)
    
    dist_arr_2d1 = np.array(xx_list)

    #stat_num_list = np.sum(dist_arr_2d1<=RADIUS, axis=1)-1
    station_address.loc[:, 'stat_num'] = (np.sum(dist_arr_2d1<=RADIUS, axis=1)-1)
    
    #station_address.to_csv(r'station_address_uni_dist_stat_num.csv', index=False)


    #%%
    
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    headers = {"Authorization": "KakaoAK 63e2b38e5cd98fb09aeba2634f72d6db"}
    #%%
    cafe_list = get_cate_size_main(station_address, RADIUS, 'CE7', '카페', url, headers)
    food_list = get_cate_size_main(station_address, RADIUS, 'FD6', '음식점', url, headers)
    bank_list = get_cate_size_main(station_address, RADIUS, 'BK9', '은행', url, headers)
    subway_list = get_cate_size_main(station_address, RADIUS, 'SW8', '지하철', url, headers)
    tour_list = get_cate_size_main(station_address, RADIUS, 'AT4', '관광', url, headers)


    
    #%%
    #https://www.data.go.kr/data/15081730/fileData.do
    bus_station_df= pd.read_csv(r'대전광역시_시내버스 기반정보_20210527_filtered.csv')
    
    xx_list = [float(str1.split('°')[0]) + float(str1.split('°')[1].split("'E")[0])/60 for str1 in bus_station_df['경도(십진수 도_분) 표기'].values]
    
    yy_list = [float(str2.split('°')[0]) + float(str2.split('°')[1].split("'N")[0])/60 for str2 in bus_station_df['위도(십진수 도_분) 표기'].values]
    
    
    bike_bus_station_dist_arr = np.array([[haversine((yy,xx), (z2,z1), unit='m') for z1, z2 in zip(xx_list, yy_list)] for xx, yy in station_address[['lon','lat']].values])

    bus_list  = np.sum(bike_bus_station_dist_arr<=RADIUS, axis=1)
    #pd.DataFrame(data=np.sum(bike_bus_station_dist_arr<=500, axis=1)).to_csv(r'bus.csv', index=False)
    
    #%%

    station_address.loc[:,'cafe'] = cafe_list
    station_address.loc[:,'food'] = food_list
    station_address.loc[:,'bank'] = bank_list
    station_address.loc[:,'subway'] = subway_list
    station_address.loc[:,'tour'] = tour_list
    station_address.loc[:,'bus'] = bus_list
    #%%
    new_station_df = station_address.loc[:BIKE_NUM-1,['station_id','uni_dist', 'stat_num', 'cafe', 'food', 'bank', 'subway', 'tour', 'bus']]
    new_station_df=new_station_df.rename(columns={'station_id':'ID'})
    #new_station_df.to_csv(r'geo_%sm.csv'%(RADIUS),index=False)





























