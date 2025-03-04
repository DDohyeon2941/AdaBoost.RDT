# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:50:20 2023

@author: dohyeon
"""

import pandas as pd

import os

import os.path, time
import numpy as np


"""날짜 구분이 안되어 있어, 수정일자 순으로 데이터를 읽어오고, 파일에 임의로 인덱스를 부여해서 저장 (0~730)"""
for yy, xx in enumerate(os.listdir(r'dust_raw_files')):

    temp_zz= (pd.read_excel(r"dust_raw_files/"+xx))
    temp_zz.to_csv(r"sorted_dust/%s.csv"%(yy))


#%% 날짜가 임의로 구분된 데이터셋을 불러오고, 대전 중구 문창동 측정소의 데이터셋을 활용해 미세먼지 데이터셋을 만듬

### 그런데 개별 데이터의 경우, 1시 - 24시 형식으로 칼럼이 존재함
### 이는 즉 2018년 1월 1일 0시의 경우, 2017년 12월 31일 24시의 정보로 대체해야 됨을 의미함

"""2018년 01월 01일 00시는 수기로 채워넣는"""
dust_arr = np.zeros(24*365*2)
for xx in range(730):
    temp_df2 = pd.read_csv(r'sorted_dust\%s.csv'%(xx), index_col=0, skiprows=[0,1,2])
    temp_df2 = temp_df2.replace('-', np.nan)
    temp_df3 = temp_df2.loc[temp_df2['측정소명']=='[대전 중구]문창동'].drop(columns=['측정망', '측정소명'])
    temp_df3 = temp_df3.fillna(0).astype(int)
    dust_arr[xx*24:(xx+1)*24]=temp_df3.values[0]


#%%
final_dust = pd.DataFrame(index=pd.date_range(start="20180101", end="20200101", freq='h')[:-1], columns=['dust'], data=np.zeros(24*365*2))


final_dust.iloc[1:]=dust_arr[:-1].reshape(-1,1)

final_dust.to_csv(r'daejeon_dust_2018-2019.csv')










