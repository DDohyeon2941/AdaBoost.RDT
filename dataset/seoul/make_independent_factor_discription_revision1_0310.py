# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:14:29 2023

@author: dohyeon
"""


import pandas as pd
import numpy as np

"""
2023-03-10일에 revision1의 요구대로 데이터셋에 대한 description을 추가하고자 작성, 여기서 데이터 값들은 확인하고, 한글 파일에 일일히 옮겨서 작성하였음
"""


"""
먼저 process_preprocess_trn_tst_0308_weekdays.py에서 trn_df를 산출함
이를 통해 선정된 정류소 번호를 인스턴스로 저장 (832개)
"""

demo_path = r'..\geographic_demographic\demo_250m.csv'
geo_path = r'..\geographic_demographic\geo_250m.csv'
d1, g1 = pd.read_csv(demo_path), pd.read_csv(geo_path)

d1 = d1.set_index('ID')

g1 = g1.set_index('ID')


###
d1.loc[trn_df.reset_index().ID.unique()].describe()
g1.loc[trn_df.reset_index().ID.unique()].describe()



#%%
"""
2019년 데이터에서 himinity를 humidity로 바꿔주고 계산
"""
weather1 = pd.read_csv(r'weather_2018.csv')
weather2 = pd.read_csv(r'weather_2019.csv')


weather2 = weather2.rename(columns={'huminity':'humidity'})
weather3 = pd.concat([weather1, weather2]).reset_index(drop=True)

weather3.describe().round(2)
