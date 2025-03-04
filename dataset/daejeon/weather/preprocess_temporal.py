# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:11:29 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np

temp_df = pd.read_csv(r'weather.csv', index_col=0)

temp_df = temp_df[['측정시각','기온','강수량','풍속','습도','일사','적설']]

temp_df.loc[:, 'time']=pd.to_datetime(temp_df['측정시각'])

temp_df.loc[:, 'year'] = [xx.year for xx in temp_df['time']]


temp_df = temp_df.loc[temp_df.year.isin([2018,2019])].reset_index(drop=True)


#%%

temp_df.loc[:, 'temp'] = temp_df['기온']
temp_df.loc[:,'solar'] = temp_df['일사'].fillna(0)
temp_df.loc[:,'snow'] = temp_df['적설'].fillna(0)
temp_df.loc[:, 'rain_log2'] = (np.log(temp_df['강수량'].rolling(window=2).mean()+1)).fillna(0)
temp_df.loc[:,'wind'] = temp_df['풍속']
temp_df.loc[:, 'humidity'] = temp_df['습도']

temp_df = temp_df[['year','time','temp','solar','snow','rain_log2','wind','humidity']]


### dust_process를 통해 전처리한 미세먼지 데이터를 불러옴

dust_df = pd.read_csv(r'daejeon_dust_2018-2019.csv', index_col=0)
temp_df.loc[:, 'dust'] = dust_df.values


#%%
temp_df.loc[temp_df.year==2018].to_csv(r'weather_2018.csv', index=False)
temp_df.loc[temp_df.year==2019].reset_index(drop=True).to_csv(r'weather_2019.csv', index=False)


#%%
