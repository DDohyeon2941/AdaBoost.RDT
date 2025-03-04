# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:12:03 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np

"""대여소"""

seoul_df = pd.read_csv(r'preprocess/dateinfo_0317.csv')


print('seoul')
print((seoul_df['0'].dropna().shape[0], seoul_df['1'].dropna().shape[0]))

daejeon_df_18 = pd.read_csv(r'tashu/station_info_2018.csv')
daejeon_df_19 = pd.read_csv(r'tashu/station_info_2019.csv')

print('daejeon')
print((daejeon_df_18.shape[0], daejeon_df_19.shape[0]))

#%%
"""날씨"""


seoul_weather_2018 = pd.read_csv(r'preprocess/weather_2018.csv')
seoul_weather_2019 = pd.read_csv(r'preprocess/weather_2019.csv')




aa1=seoul_weather_2018.describe()
aa2=seoul_weather_2019.describe()


#%%

daejeon_weather = pd.read_csv(r'tashu/weather/weather.csv', index_col=0)
daejeon_weather_2018 = pd.read_csv(r'tashu/weather/weather_2018.csv')
daejeon_weather_2019 = pd.read_csv(r'tashu/weather/weather_2019.csv')


aa3=daejeon_weather_2018.describe()
aa4=daejeon_weather_2019.describe()


daejeon_weather.loc[:, 'date'] = pd.to_datetime(daejeon_weather['측정시각'])

daejeon_weather.loc[:, 'year'] = [xx.year for xx in daejeon_weather.date]


(daejeon_weather.loc[daejeon_weather.year == 2018]['적설']).fillna(0).sum()

#%%
"""지리정보/인구통계"""

station_idx_2018 = seoul_df['0'].dropna().astype(int).values
station_idx_2019 = seoul_df['1'].dropna().astype(int).values

seoul_demo_2018 = pd.read_csv(r'dataset/geographic_demographic/demo_2018_seoul.csv')
seoul_demo_2019 = pd.read_csv(r'dataset/geographic_demographic/demo_2019_seoul.csv')


bb1 = (seoul_demo_2018.loc[seoul_demo_2018.ID.isin(station_idx_2018)]).describe()
bb2 = (seoul_demo_2019.loc[seoul_demo_2019.ID.isin(station_idx_2019)]).describe()


#%%

daejeon_station_idx_2018 = daejeon_df_18['0'].values
daejeon_station_idx_2019 = daejeon_df_19['0'].values


daejeon_demo_2018 = pd.read_csv(r'tashu/demo/demo_500m_2018.csv')
daejeon_demo_2019 = pd.read_csv(r'tashu/demo/demo_500m_2019.csv')


daejeon_demo_2019 = (daejeon_demo_2019.loc[daejeon_demo_2019.ID != 22]).reset_index(drop=True)


bb3 = (daejeon_demo_2018.loc[daejeon_demo_2018.ID.isin(daejeon_station_idx_2018)]).describe()
bb4 = (daejeon_demo_2019.loc[daejeon_demo_2019.ID.isin(daejeon_station_idx_2019)]).describe()

#%%
"""지리정보"""
"""대전의 경우, 22번 정류소는 제거해야"""

seoul_geo = pd.read_csv(r'dataset/geographic_demographic/geo_500m.csv')
seoul_geo1 = pd.read_csv(r'dataset/geographic_demographic/geo_250m.csv')


seoul_geo1.loc[seoul_geo.ID.isin(station_idx_2018)].describe()

seoul_geo1.loc[seoul_geo.ID.isin(station_idx_2019)].describe()['uni_dist']

cc1=seoul_geo.loc[seoul_geo.ID.isin(station_idx_2018)].describe()
cc2=seoul_geo.loc[seoul_geo.ID.isin(station_idx_2019)].describe()


#%%

daejeon_geo = pd.read_csv(r'tashu/geo/geo_500m.csv')

daejeon_station_idx_2019

cc3=(daejeon_geo.loc[daejeon_geo.ID.isin(daejeon_station_idx_2018)])[['food','cafe','tour','bank','stat_num','uni_dist','subway','bus']].describe()



daejeon_geo_2019 = daejeon_geo.loc[daejeon_geo.ID.isin(daejeon_station_idx_2019)]
daejeon_geo_2019 = (daejeon_geo_2019.loc[daejeon_geo_2019.ID != 22]).reset_index(drop=True)



cc4 = daejeon_geo_2019[['food','cafe','tour','bank','stat_num','uni_dist','subway','bus']].describe()















