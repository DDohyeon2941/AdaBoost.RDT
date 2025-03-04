# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:33:55 2023

@author: dohyeon
"""

from selenium import webdriver
from time import time
# 다운받은 webdriver의 경로를 지정
driver = webdriver.Chrome(executable_path=r'C:\Users\dohyeon\Desktop/chromedriver.exe')
driver.get('https://www.airkorea.or.kr/web/pmRelay?itemCode=10007&pMENU_NO=108')

#처음에 2018년 01월 01일, 대전지역, 검색 및 엑셀파일 다운

driver.find_element_by_xpath(f'/html/body/div[4]/div[2]/div[2]/form/div/select/option[6]').click()

driver.find_element_by_xpath(f'/html/body/div[4]/div[2]/div[2]/form/div/span[1]/img').click()

driver.find_element_by_xpath(f'/html/body/div[6]/div[1]/div/select[1]/option[3]').click()
driver.find_element_by_xpath(f'/html/body/div[6]/div[1]/div/select[2]/option[1]').click()
driver.find_element_by_xpath(f'/html/body/div[6]/table/tbody/tr[1]/td[2]/a').click()
driver.find_element_by_xpath(f'/html/body/div[4]/div[2]/div[2]/form/div/div/a[1]').click()

#엑셀다운 2018년 1월 1일
driver.find_element_by_xpath(f'/html/body/div[4]/div[2]/div[2]/form/div/div/a[3]').click()

for xx in range(730):
    #다음 일자로 넘어가고, 검색하고, 엑셀 다운
    driver.implicitly_wait(10)
    driver.find_element_by_xpath(f'/html/body/div[4]/div[2]/div[2]/form/div/span[1]/span/a[2]').click()
    driver.find_element_by_xpath(f'/html/body/div[4]/div[2]/div[2]/form/div/div/a[1]').click()
    driver.find_element_by_xpath(f'/html/body/div[4]/div[2]/div[2]/form/div/div/a[3]').click()
    driver.implicitly_wait(10)

#%%

#%%

