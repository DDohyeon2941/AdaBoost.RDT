# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:20:50 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 18:35:23 2019

@author: User
"""

# 날씨 변수 전처리.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import configparser
import sys
import pdb


def fill_wind(df_, n_days):
    null_idx = df_[df_[0].isnull()].index
    candi_arr = np.zeros((null_idx.shape[0], n_days*2))

    for uidx, uni_day in enumerate(np.arange(1,n_days+1,1)):
        after_idx = null_idx + pd.DateOffset(days=uni_day)
        before_idx = null_idx - pd.DateOffset(days=uni_day)

        candi_arr[:, uidx*2] = df_.loc[before_idx].values.squeeze()
        candi_arr[:, uidx*2 +1] = df_.loc[after_idx].values.squeeze()
    df_.loc[null_idx, 0] = np.around(np.nanmean(candi_arr, axis=1),1)
    return df_



def fill_cloud(df_, n_hours):
    null_idx = df_[df_[0].isnull()].index
    candi_arr = np.zeros((null_idx.shape[0], int(n_hours)*2))

    for uidx, uni_hour in enumerate(np.arange(1,n_hours+1,1)):
        after_idx = null_idx + pd.DateOffset(hours=uni_hour)
        before_idx = null_idx - pd.DateOffset(hours=uni_hour)

        candi_arr[:, uidx*2] = df_.loc[before_idx].values.squeeze()
        candi_arr[:, uidx*2 +1] = df_.loc[after_idx].values.squeeze()
    df_.loc[null_idx, 0] = np.around(np.nanmean(candi_arr, axis=1),1)
    return df_




#%%

tst_weather = pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\weather_2019_data.csv')
tst_weather.head(3).T

date_range_idx = pd.date_range(start="20190101", end="20200101", freq='h')[:-1]
date_range_df = pd.DataFrame(index=date_range_idx, data=np.ones(date_range_idx.shape[0])*np.nan)

t1=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\temp_2019.csv', encoding='cp949')
t1=t1[t1.columns[2:]]
t1.columns = ['date', 'temp']
t1.set_index('date')
#%%

t2=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\wind_2019.csv', encoding='cp949')
t2=t2[t2.columns[2:]]
t2.columns = ['date', 'wind']

new_t2 = date_range_df.copy(deep=True)
new_t2.loc[t2[t2.columns[0]],0] = t2[t2.columns[-1]].values
new_t2 = fill_wind(new_t2, 3)
new_t2 = new_t2.reset_index()
new_t2.columns = ['date', 'wind']
#%%


t3=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\huminity_2019.csv', encoding='cp949')
t3=t3[t3.columns[2:]]
t3.columns = ['date', 'huminity']
t3.isnull().sum()
#%%

t4=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\sun_2019.csv', encoding='cp949')
t4=t4[t4.columns[2:]]
t4.columns = ['date', 'sun']

new_t4 = date_range_df.copy(deep=True)
new_t4.loc[t4[t4.columns[0]],0] = t4[t4.columns[-1]].values
new_t4 = new_t4.fillna(0.0)
new_t4 = new_t4.reset_index()
new_t4.columns = ['date', 'sun']

#%%

t5=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\snow_2019.csv', encoding='cp949')
t5=t5[t5.columns[2:]]
t5.columns = ['date', 'snow']

new_t5 = date_range_df.copy(deep=True)
new_t5.loc[t5[t5.columns[0]],0] = t5[t5.columns[-1]].values
new_t5 = new_t5.fillna(0.0)
new_t5 = new_t5.reset_index()
new_t5.columns = ['date', 'snow']


#%%
t6=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\dust_2019_1.csv', encoding='cp949', skiprows=[0,1,2,3])

t6=t6[t6.columns[2:]]
t6.columns = ['date', 'dust']

new_t6 = date_range_df.copy(deep=True)
new_t6.loc[t6[t6.columns[0]],0] = t6[t6.columns[-1]].values
new_t6 = new_t6.fillna(0.0)
new_t6 = new_t6.reset_index()
new_t6.columns = ['date', 'dust']

#%%
t7=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\rain_2019_hour.csv', encoding='cp949')
t7=t7[t7.columns[2:-2]]
t7.columns = ['date', 'rain']
t7 = t7.fillna(0.0)
t7.loc[:, 'rain_log2'] = np.log(t7.rain.rolling(window=2).mean()+1)
t7 = t7.fillna(0.0)
t7

#%%
t8=pd.read_csv(r'D:\project_repository\flow_prediction\dock\experiment\dataset\seoul_bike\weather\2019\cloud_2019.csv', encoding='cp949')
t8=t8[t8.columns[2:]]
t8.columns = ['date', 'cloud']
t8.loc[:, 'date'] = pd.to_datetime(t8['date'])
t8 = t8.astype({'cloud':float})

new_t8 = date_range_df.copy(deep=True)
new_t8.loc[t8[t8.columns[0]],0] = t8[t8.columns[-1]].values
new_t8 = fill_cloud(new_t8, 1.0)
new_t8 = new_t8.reset_index()
new_t8.columns = ['date', 'cloud']


#%%




tt1 = pd.merge(pd.merge(t1,t3), t7)
tt1.loc[:, 'date'] = pd.to_datetime(tt1.date)

tt2 = pd.merge(pd.merge(pd.merge(pd.merge(new_t2, new_t4), new_t5), new_t6), new_t8)
tt2.loc[:, 'date'] = pd.to_datetime(tt2.date)

tt3 = pd.merge(tt1,tt2)

tt3.set_index('date').loc[pd.date_range(start="20190101", end="20191201", freq='h')[:-1]]
tst_weather.set_index('date')[['temp','humidity','rain','log_rain2','wind','sun','snow','dust','cloud']]



tt3.to_csv(r'weather_2019.csv',index=False)













#%%
def change_df(df, value_name):
    """
    필요한 컬럼 추출 및 컬럼 이름 변환
    *** input
    df : 변환할 dataframe
    value_name : 측정치가 저장된 컬럼 이름.
    
    *** return
    extradcted_df : 필요한 column만 추출한 dataframe
    """

    # 컬럼길이가 3과 같거나 크면 가공 안된 데이터임.
    if len(df.columns) >=3:
        
        # dust_data는 끝의 데이터 1개를 삭제해야 함
            
        # 날짜, value 값 추출
        extracted_df = df.iloc[:,[-2,-1]]
        # 컬럼 이름 바꿈.
        
        # 3보다 작으면 가공 된 데이터
        extracted_df.columns = ['date', value_name]
        extracted_df['date'] = pd.to_datetime(extracted_df['date']) 
        
        # 마지막 row 값이 year가 다른지 찾음.
       
        last_row_year = extracted_df.loc[len(extracted_df)-1,'date'].year
        
        compared_row_year = extracted_df.loc[0,'date'].year
        
        if last_row_year != compared_row_year:
            last_label = len(extracted_df)-1
            extracted_df.drop(last_label, inplace=True)
            
        return extracted_df


def fill_nan_hum_cloud(df_):
    """
    습도, 운량의 NaN값에 값을 넣음
    앞 뒤 1시간의 값을 기준으로 함.
    """
    
    for col in ['humidity', 'cloud']:
        for ind in range(len(df_)):
            if np.isnan(df_.loc[ind,col]):
                
                # ind가 0인 경우 처리법.
                if ind == 0:
                     df_.loc[ind,col] = df_[~df_[col].isnull()].reset_index().loc[0,col].squeeze()
                     continue
                
                try :
                    if np.isnan(df_.loc[ind+1,col]):
                        df_.loc[ind,col] = df_.loc[ind-1,col]
                    else:
                        df_.loc[ind,col] = np.mean([df_.loc[ind-1,col], df_.loc[ind+1,col]])
                
                except(KeyError):
                    print("끝")
                    break
    
    return df_


def fill_nan_rain_binary(df_):
    
    
    """
    비 데이터를 비가 안오면 0, 오면 1로 처리함
    """
    
    df_['rain_state'] = 1
    one_ind = df_[~df_['rain'].isnull()].index
    
    one_ind = df_[(df_['rain'].isnull())].index.tolist()
    one_ind2 = df_[df_['rain'] == 0].index.tolist()
    one_ind = one_ind + one_ind2
    
    
    df_.loc[one_ind, 'rain_state'] = 0
    df_['rain'] = df_['rain_state']
    df_ = df_.drop('rain_state', axis=1)
    
    
    return df_



def return_train_test_df(df_, window_col_list):
    
    
    train_data = df_[(df_['date']>=train_start) & (df_['date']<=train_end)].reset_index(drop=True)
    test_data = df_[(df_['date']>=test_start) & (df_['date']<=test_end)].reset_index(drop=True)
    
    
    
    return train_data, test_data



def rain_data_processing(year_list):   
    
    #rain_data_name = ['rain_%s'%(train_year), 'rain_%s'%(test_year)]
    #train_test_path = [load_weather_train_path, load_weather_test_path]
    rain_train_test_list = []
    
    rain_hour = pd.DataFrame()
    for year in year_list:
        load_path = load_weather_path+'%s/'%(year)
        try :
            temp_rain_hour = pd.read_csv(load_path+'rain_%s_hour.csv' %year, encoding='euckr', parse_dates=['일시'])
        except(FileNotFoundError):
            return []
        
        rain_hour = pd.concat([rain_hour, temp_rain_hour], axis=0).reset_index(drop=True)
    
    #rain_day = pd.read_csv(t_t_path+'%s_day.csv' %r_data_name, encoding='euckr', parse_dates=['일시'])
    rain_hour.columns = ['sta', 'sta_name','date', 'rain', 'humidity', 'cloud']
    
    #rain_day.columns = ['sta','sta_name','date','hr','d_rain']
    

    rain_hour = fill_nan_hum_cloud(rain_hour)
    #pdb.set_trace()
    # 비 결측값 채우기
    #rain_hour = fill_nan_rain_binary(rain_hour)
    rain_hour['rain'] = rain_hour['rain'].fillna(0)

    rain_hour['rain2'] = rain_hour['rain'].rolling(window=2).mean()
    #pdb.set_trace()
    # 기타 처리
    
    rain_hour['log_rain'] = np.log(rain_hour['rain']+1)
    rain_hour['log_rain2'] = np.log(rain_hour['rain2']+1)
    #pdb.set_trace()
    rain_hour= rain_hour.drop(['rain2'], axis=1)
    rain_hour['rain'] = rain_hour['rain'].apply(lambda x: 1 if x>=0.5 else 0)
    
    rain_col = ['rain','log_rain','log_rain2']
    
    return rain_col, rain_hour


def distinc_train_test(num):
    """
    train, test 데이터 처리에 따라 변수 할당
    """
    
    if num == 0:
        t_t = 'train'
        year = train_year
        start_date = train_start
        end_date = train_end

        save_path = save_weather_train_path
    else :
        t_t = 'test'
        year = test_year
        start_date = test_start
        end_date = test_end

        save_path = save_weather_test_path
        
    return t_t, year, start_date, end_date, save_path


def load_file(path_, file_name, skip_option=False):
    if skip_option == True:
        df = pd.read_csv(path_ + file_name, encoding='euc-kr', skiprows=5)
    else:
        df = pd.read_csv(path_ + file_name, encoding='euc-kr')

    return df


def fill_nan(df_, nan_check_col_, target_date):
    for i in nan_check_col_:
        
        # null인 index 값들 구함.
        null_index = df_[i][df_[i].isnull()].index.values
        
        # 과거 3일, 미래 3일 값을 기준으로 합할 값을 구함.
        for j in null_index:
            target_mean_list = []
            
            for k in range(target_date):
                
                try:
                    before_data = df_.loc[j-24*(k+1),i].squeeze()
                    after_data = df_.loc[j-24*((k+1)*-1),i].squeeze()
                    if np.isnan(before_data) == False:
                        target_mean_list.append(df_.loc[j-24*(k+1),i])
                    if np.isnan(after_data) == False:
                        target_mean_list.append(df_.loc[j-24*((k+1)*-1),i])
                
                # 만약 과거 데이터, 미래 데이터가 없는경우 pass 시킴.
                except(KeyError):
                    pass
                
            df_.loc[j, i] = np.mean(target_mean_list)
            
    df_.fillna(0, inplace=True)
    
    return df_



def check_variable(load_path_):
    
    """
    실제로 있는 변수만 추출하게 만드는 함수
    """
    
    check_var_list = [i.split('_')[0] for i in os.listdir(load_path_)]
    t_weather_list = ['dust', 'snow', 'sun', 'temp', 'wind']
    ind_list = []
    for i in range(len(t_weather_list)):
        
        c_var = t_weather_list[i]
        
        if c_var in check_var_list:
            ind_list.append(i)
            continue
        
    t_weather_list = np.array(t_weather_list)[ind_list].tolist()

    return t_weather_list
            


def main():
    
    
    
    year_list = os.listdir(load_weather_path)
    rain_col, rain_df = rain_data_processing(year_list)
    
    #t_t, year, start_date, end_date, save_path = distinc_train_test(k)
    
    
    
    """
    모든 해당연도에 있는 컬럼값만 추출함
    """
    t_weather_list = []
    for year in year_list:
        load_path = load_weather_path+'%s/'%(year)
        
        temp_t_weather_list = check_variable(load_path)
        if len(t_weather_list) == 0:
            t_weather_list = temp_t_weather_list
        else :
            t_weather_list = np.intersect1d(t_weather_list, temp_t_weather_list)
    
    t_weather_list = t_weather_list.tolist()
    
    
    variable_list = []

    for t_var in t_weather_list:
        t_data = pd.DataFrame()
        for year in year_list:
            load_path = load_weather_path+'%s/'%(year)
            if t_var == 'dust':
                temp_t_data = load_file(load_path, '%s_%s.csv'%(t_var, year), skip_option=True)
                
                
            else:
                temp_t_data = load_file(load_path, '%s_%s.csv'%(t_var, year))
            t_data = pd.concat([t_data, temp_t_data], axis=0)
        
        t_data = t_data.reset_index(drop=True)
        variable_list.append(t_data)
    
    
    

    #rain_data = rain_df[['sta','date']+rain_col]
    #humidity_data = rain_df[['sta','date','humidity']]
    #cloud_data = rain_df[['sta','date','cloud']]
    rain_data = rain_df[['date']+rain_col]
    humidity_data = rain_df[['date','humidity']]
    cloud_data = rain_df[['date','cloud']]

    variable_list.append(rain_data)
    variable_list.append(humidity_data)
    variable_list.append(cloud_data)
    

    #pdb.set_trace()
    for i in range(len(t_weather_list)):
        
        variable_list[i] = change_df(variable_list[i], t_weather_list[i])


    start_date = train_start
    end_date = test_end
    
    data_period = pd.date_range(start=start_date ,freq='H',end = end_date)
    
    weather_data = pd.DataFrame({'date': data_period})


    # 데이터를 merge하여 없는 값은 0 또는 평균값으로 처리함.
    # 왼쪽 데이터를 기준으로 합치는 작업임
    #pdb.set_trace()
    for i in variable_list:
        weather_data = pd.merge(weather_data, i, how='left', on='date')

    #pdb.set_trace()
    weather_data = weather_data.reset_index(drop=True)

    # null 값 처리 : 온도, 풍속, 습도 빼고 전부 0으로 처리함
    # 온도
        # 1. NaN 값 확인
    
    nan_check_col = ['temp','wind']
            # NaN값을 평균 6일치 이전, 이후(각 3일) 데이터의 값을 기준으로 구함

    weather_data = fill_nan(weather_data, nan_check_col, 3)

    
    weather_train, weather_test = return_train_test_df(weather_data, ['temp'])


    weather_train.to_csv(save_weather_train_path+'weather_%s_data.csv'%train_year, index=False)
    weather_test.to_csv(save_weather_test_path+'weather_%s_data.csv'%test_year, index=False)



if __name__ == "__main__":
    main()
    
    
    