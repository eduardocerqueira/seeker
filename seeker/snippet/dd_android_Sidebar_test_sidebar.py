#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import allure
from selenium.webdriver.support.ui import WebDriverWait

def test_clickText(driver,param):
    allure.dynamic.title("点击%s"%param['text'])
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[@text="%s"]'%param['text']))
    driver.find_element_by_xpath('//*[@text="%s"]'%param['text']).click()

def test_clickInbox(driver,param):
    allure.dynamic.title("点击收件箱")
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[@text="收件箱"]'))
    driver.find_element_by_xpath('//*[@text="收件箱"]').click()

def test_clickSetting(driver,param):
    allure.dynamic.title("点击设置")
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/drawer_img_setting'))
    driver.find_element_by_id('com.asiainfo.android:id/drawer_img_setting').click()


def test_clickBacking(driver,param):
    allure.dynamic.title("点击返回")
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/mImage_menu_close'))
    driver.find_element_by_id('com.asiainfo.android:id/mImage_menu_close').click()