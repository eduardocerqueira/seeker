#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

#!/usr/bin/env python
# coding=utf-8

import allure
from selenium.webdriver.support.ui import WebDriverWait
import os

def test_clickText(driver,param):
    allure.dynamic.title("点击:%s"%param['text'])
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"%s")]'%param['text']))
    driver.find_element_by_xpath('//*[contains(@text,"%s")]'%param['text']).click()



def test_addBusinessCard(driver,param):
    allure.dynamic.title("新增名片")
    print (param)
    # 等待名片列表加载点击
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"新建签名")]'))
    driver.find_element_by_xpath('//*[contains(@text,"新建签名")]').click()
    # 等待信息加载填写
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"你的名字")]'))
    driver.find_element_by_xpath('//*[contains(@text,"你的名字")]').send_keys(param['name'])
    # 填写常用邮箱
    driver.find_element_by_xpath('//*[contains(@text,"常用邮箱")]').send_keys(param['mailbox'])
    # 填写手机号码
    driver.find_element_by_xpath('//*[contains(@text,"手机号码")]').send_keys(param['phno'])
    # 填写公司
    driver.find_element_by_xpath('//*[contains(@text,"公司/机构")]').send_keys(param['cpy'])
    # 点击更多信息
    driver.find_element_by_xpath('//*[contains(@text,"更多信息")]').click()
    # 填写职衔
    driver.find_element_by_xpath('//*[contains(@text,"职衔名称")]').send_keys(param['titl'])
    # 填写地址
    driver.find_element_by_xpath('//*[contains(@text,"办公地址")]').send_keys(param['adr'])
    # 填写传真
    driver.find_element_by_xpath('//*[contains(@text,"传真号码")]').send_keys(param['fax'])
    # 点击保存
    driver.find_element_by_xpath('//*[contains(@text,"保存")]').click()
    # 点击返回
    driver.find_element_by_id('com.asiainfo.android:id/iv_left_button').click()


def test_back(driver,param):
    allure.dynamic.title("返回")
    print (param)
    # 等待名片列表加载点击
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"设置")]'))
    # 点击返回
    driver.find_element_by_id('com.asiainfo.android:id/iv_left_button').click()


def test_login(driver,param):
    allure.dynamic.title("设置登录")
    print (param)
    os.system('adb shell svc data enable')
    # 等待名片列表加载点击
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"账号设置")]'))
    # 输入密码
    driver.find_element_by_id('com.asiainfo.android:id/ed_pocket_second_page_psw').send_keys(param['pwd'])
    # 填写收件服务器
    driver.find_element_by_id('com.asiainfo.android:id/pocket_second_page_receive_server').send_keys('imap.wo.cn')
    # 填写发件服务器
    driver.find_element_by_id('com.asiainfo.android:id/ed_pocket_second_page_send_server').send_keys('smtp.wo.cn')
    # 点击确认
    driver.find_element_by_id('com.asiainfo.android:id/iv_right_button').click()