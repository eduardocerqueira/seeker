#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import pytest
import allure
import time
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


#全局搜索
def test_mail_search_all(param,driver):
    allure.dynamic.title("登录成功后，点击搜索框输入关键字搜索邮件")
    driver.find_element_by_id('lyfullsearch').click()
    driver.implicitly_wait(5)
    driver.find_element_by_id('lyfullsearch').clear()
    driver.find_element_by_id('lyfullsearch').send_keys(param['key'])
    time.sleep(2)
    #点击 包含测试的邮件
    driver.find_element_by_class_name('u-select-item-hover').click()
    time.sleep(2)
    #关闭搜索出的结果页面
    # driver.find_element_by_class_name('iconfont iconcloseall closeall').click()
    # time.sleep(3)
    try:
        WebDriverWait(driver, 20, 0.1).until(
            lambda driver: driver.find_element_by_class_name("list-body"))
        print("根据包含测试的邮件：搜索出相关邮件列表")
    except:
        print("根据包含测试的邮件：搜索结果为空")
    body = driver.find_element_by_class_name("h3").text
    print(body)
    assert "没有搜索到" in body,"有搜索结果"



#根据发件人搜索
def test_mail_search_sender(param,driver):
    allure.dynamic.title("登录成功后，点击搜索框输入关键字 根据发件人搜索邮件")
    #点击搜索框，输入关键字
    driver.find_element_by_id('lyfullsearch').click()
    driver.implicitly_wait(5)
    driver.find_element_by_id('lyfullsearch').clear()
    driver.find_element_by_id('lyfullsearch').send_keys(param['key'])
    time.sleep(2)
    # 点击发件人包含测试的邮件
    driver.find_element_by_css_selector('.u-select-content > ul > .u-select-item:nth-child(2)').click()
    ele = True
    try:
        WebDriverWait(driver, 20, 0.1).until(
            lambda driver: driver.find_element_by_class_name("list-body"))
        print("根据发件人搜索：搜索出相关邮件列表")
    except:
        print("根据发件人搜索：搜索结果为空")
        ele=False
    assert ele==True , "搜索结果为空"


#根据主题搜索
def test_mail_search_theme(param,driver):
    allure.dynamic.title("登录成功后，点击搜索框输入关键字 根据主题搜索邮件")
    # 点击搜索框，输入关键字
    driver.find_element_by_id('lyfullsearch').click()
    driver.implicitly_wait(5)
    driver.find_element_by_id('lyfullsearch').clear()
    driver.find_element_by_id('lyfullsearch').send_keys(param['key'])
    time.sleep(2)
    # 点击主题包含测试的邮件
    driver.find_element_by_css_selector('.u-select-content > ul > .u-select-item:nth-child(3)').click()
    ele = True
    try:
        WebDriverWait(driver, 20, 0.1).until(
            lambda driver: driver.find_element_by_class_name("list-body"))
        print("根据主题搜索：搜索出相关邮件列表")
    except:
        print("根据主题搜索：搜索结果为空")
        ele = False
    assert ele == True, "搜索结果为空"


#高级搜索
def test_mail_adsearch(param,driver):
    allure.dynamic.title("登录成功后，点击搜索框高级搜索")
    driver.find_element_by_id('lyfullsearch').click()
    driver.implicitly_wait(5)
    driver.find_element_by_id('lyfullsearch').clear()
    driver.find_element_by_id('lyfullsearch').send_keys(param['key'])
    # 点击高级搜索
    driver.find_element_by_css_selector('.u-select-content > ul > .u-select-item:nth-child(5)').click()
    time.sleep(3)
    #展开高级搜索，设置搜索条件，输入关键字
    driver.find_element_by_name('pattern').send_keys(param['key1'])
    #高级搜索选项中，文件夹选择收件箱
    driver.find_elements_by_class_name('u-select-trigger')[1].click()
    driver.find_elements_by_class_name('u-select-item')[2].click()
    #高级搜索选项，输入邮件主题
    driver.find_element_by_name('subject').send_keys(param['subject'])
    #设置完搜索条件后，点击确定按钮
    driver.find_element_by_css_selector('.u-advancedsearchbox-btns > .u-btn:nth-child(1)').click()
    time.sleep(2)
    ele = True
    try:
        WebDriverWait(driver, 20, 0.1).until(
            lambda driver: driver.find_element_by_class_name("list-body"))
        print("高级搜索，搜索出相关邮件列表")
    except:
        print("高级搜索，搜索结果为空")
        ele=False
    assert ele==True,"搜索结果为空"






if __name__ == '__main__':
    pytest.main(['-s','test_mail_search.py::test_mail_search_sender'])
