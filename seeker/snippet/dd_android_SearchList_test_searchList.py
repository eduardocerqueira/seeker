#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-

import allure
from selenium.webdriver.support.ui import WebDriverWait

def test_inputSearch(driver, param):
    allure.dynamic.title("填入搜索条件")
    print(param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/edit_input_search'))
    driver.find_element_by_id('com.asiainfo.android:id/edit_input_search').send_keys(param['text'])

def test_search(driver, param):
    allure.dynamic.title("搜索")
    print(param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/btn_input_search'))
    driver.find_element_by_id('com.asiainfo.android:id/btn_input_search').click()