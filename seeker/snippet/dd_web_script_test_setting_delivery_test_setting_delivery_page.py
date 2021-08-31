#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#收发件页面跳转
from web.script.test_sendmail.test_sendmail_editor import *

def test_setting_delivery_page(driver,param):
    allure.dynamic.title("收发信设置-页面跳转")
    #param["setting_page"]="写信设置"
    #dd=["写信设置","回复设置","读信设置","自动转发","假期自动回复","模板信设置"]

    WebDriverWait(driver, 5,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".iconsetupcenter"))).click()
    WebDriverWait(driver, 5,0.1).until(
        EC.element_to_be_clickable((By.LINK_TEXT,"收发信设置"))).click()
    WebDriverWait(driver, 5,0.1).until(
        EC.element_to_be_clickable((By.LINK_TEXT,param["setting_page"]))).click()



if __name__ == '__main__':
    pytest.main(["-s","test_setting_delivery_page.py::test_setting_delivery_page"])