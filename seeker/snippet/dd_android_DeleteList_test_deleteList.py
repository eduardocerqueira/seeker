#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import allure
from selenium.webdriver.support.ui import WebDriverWait
import time

def test_find(driver,param):
    allure.dynamic.title("查找对应邮件:%s"%param['findMail'])
    print (param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/tv_actionbar_lefttitle'))

    i = 5
    while i>0:
        try:
            driver.find_element_by_xpath('//*[contains(@text,"%s")]' % param['findMail'])
            break
        except :
            driver.swipe(0, 923, 0, 163, 3000)
            time.sleep(1)
            i = i-1
            print(i)
    if i == 0:
        print("未查到邮件：%s" % param['findMail'])
        assert False
    else:
        print("已查到邮件：%s" % param['findMail'])

def test_find_click(driver,param):
    allure.dynamic.title("查找对应邮件:%s"%param['findMail'])
    print (param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"%s")]'%param['findMail']))
    assert True
    print("已查到邮件：%s"%param['findMail'])
    driver.find_element_by_xpath('//*[contains(@text,"%s")]' % param['findMail']).click()