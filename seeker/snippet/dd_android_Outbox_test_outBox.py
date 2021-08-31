#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import allure
from selenium.webdriver.support.ui import WebDriverWait
from appium.webdriver.common.touch_action import TouchAction
import dill
import pytest


def test_send(driver, param):
    allure.dynamic.title("发送")
    print(param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_right_button'))
    driver.find_element_by_id('com.asiainfo.android:id/iv_right_button').click()


def test_inputSender(driver, param):
    allure.dynamic.title("收件人")
    print(param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/tv_title'))
    # 输入收件人
    driver.find_element_by_xpath(
        '//android.widget.TextView[@text="收件人："]/following-sibling::android.widget.EditText').send_keys(
        '18010096059@wo.cn ')

    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/tv_title'))
    # # 输入收件人
    # driver.find_element_by_xpath(param['xpath']).send_keys(param['value'])


def test_writeMail(driver, param):
    allure.dynamic.title("写邮件内容")
    print(param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/tv_title'))
    # 输入收件人
    driver.find_element_by_xpath(
        '//android.widget.TextView[@text="收件人："]/following-sibling::android.widget.EditText').send_keys(
        '18707142515@wo.cn ')
        # '18010096059@wo.cn ')

    driver.find_element_by_xpath(
        '//android.widget.TextView[@text="主    题："]/following-sibling::android.widget.EditText').send_keys(
        '%s' % param['text'])

    driver.find_element_by_xpath(
        '//android.widget.RelativeLayout[@resource-id="com.asiainfo.android:id/mRelative_webView"]/android.webkit.WebView/android.webkit.WebView/android.view.View').click()

    driver.keyevent(48)
    driver.keyevent(33)
    driver.keyevent(47)
    driver.keyevent(48)

    # driver.start_activity('com.asiainfo.android','com.asiainfo.mail.ui.mainpage.MainActivity')

    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/tv_title'))
    driver.find_element_by_id('com.asiainfo.android:id/tv_title').click()
