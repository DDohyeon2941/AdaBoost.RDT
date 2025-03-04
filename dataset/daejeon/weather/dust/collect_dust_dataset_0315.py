# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:26:18 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np

import requests
import json

from shapely.geometry import Point, Polygon

import pyproj
from pyproj import Transformer
from shapely.ops import transform
from functools import partial


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

#%%

if __name__ == "__main__":
    #https://www.airkorea.or.kr/web/stationInfo?pMENU_NO=93 대전 측정소 정보, 구두 안내(13곳)와는 다르게 11개만 있음
    air_station = pd.read_csv(r'station_list.csv', skiprows=[0,1]).dropna()
    air_station_list = []
    for uni_air_station in air_station['측정소 주소']:
        address_latlng = getLatLng(uni_air_station)
        air_station_list.append(address_latlng)
    
    
    pd.DataFrame(data=[Point([float(yy) for yy in xx]).wkt for xx in air_station_list]).to_csv(r'air_station_location.csv')
    
    """이렇게 측정소 위치를 산출하고나면, QGIS에서 측정소 및 자전거대여소 위치정보를 load해서 비교함 
    집중도가 높은 경우 1번(유성구 구성동) 4번(중구 문창동)이 있음

    구별로 대기정보를 구할 수 없다면, 1번 또는 4번 측정소의 정보만 가지고 모델링 해야함
    구별로 된다고 하면 아래 5곳의 정보를 수집하려고 함

    1. 유성구: 구성동
    2. 동구: 대성동
    3. 서구: 둔산동
    4. 중구: 문창동
    5. 대덕구: 읍내동

    """
