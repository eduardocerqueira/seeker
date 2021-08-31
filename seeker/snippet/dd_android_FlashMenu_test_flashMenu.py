#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import allure
from selenium.webdriver.support.ui import WebDriverWait

def test_clickTest(driver,param):
    allure.dynamic.title("点击:%s" % param['text'])
    print(param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[@text="%s"]' % param['text']))
    driver.find_element_by_xpath('//*[@text="%s"]' % param['text']).click()

def test_clickContainsTest(driver,param):
    allure.dynamic.title("点击:%s" % param['text'])
    print(param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"%s")]' % param['text']))
    driver.find_element_by_xpath('//*[contains(@text,"%s")]' % param['text']).click()

    # driver.find_element_by_xpath('//*[contains(@text,"确认")]').click()
    #
    # WebDriverWait(driver, 10, 0.1).until(
    #     lambda driver: driver.find_element_by_xpath('//*[contains(@text,"确认")]' ))
