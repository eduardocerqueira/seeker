#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import allure
from selenium.webdriver.support.ui import WebDriverWait

def test_complete(driver, param):
    allure.dynamic.title("完成")
    print(param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/actionbar_new_btn'))
    driver.find_element_by_id('com.asiainfo.android:id/actionbar_new_btn').click()

def test_back(driver, param):
    allure.dynamic.title("完成")
    print(param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/actionbar_back_btn'))
    driver.find_element_by_id('com.asiainfo.android:id/actionbar_back_btn').click()

