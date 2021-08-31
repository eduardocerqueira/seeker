#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# coding=utf-8
import allure
from selenium.webdriver.support.ui import WebDriverWait
import os


def test_switchData(driver,param):
    allure.dynamic.title("关闭流量")
    print (param)
    # 通知栏
    driver.open_notifications()
    # 等待并关闭手机流量
    WebDriverWait(driver, 30, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"移动数据")]'))
    driver.find_element_by_xpath('//*[contains(@text,"移动数据")]').click()
    # 返回
    driver.press_keycode(4)


def test_onData(driver,param):
    allure.dynamic.title("打开流量")
    print (param)
    os.system('adb -s 3HX7N17106006538 shell svc data enable')