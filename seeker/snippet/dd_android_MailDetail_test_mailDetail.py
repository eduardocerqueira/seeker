#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import allure
from selenium.webdriver.support.ui import WebDriverWait
from appium.webdriver.common.touch_action import TouchAction
from autotest.memcachedUtil import *
import dill
from autotest.AppiumExtend import Appium_Extend
import pytest
import time

client = Client(('localhost', 11211))

def test_checkMailDetail(driver,param):
    allure.dynamic.title("检查邮件内容")
    print (param)
    start = time.time()
    WebDriverWait(driver, 40, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//android.view.View[@text="电话:"]'))
    durationTime = time.time() - start
    allure.attach("加载时间:%s" % (durationTime))
    # WebDriverWait(driver, 10, 0.1).until(
    #     lambda driver: driver.find_element_by_id('com.asiainfo.android:id/btn_message_detail_list_item_subject'))
    # 发件邮箱
    sendMail = driver.find_element_by_id('com.asiainfo.android:id/tv_message_detail_list_item_sender_subject_whole_name').text

    # 发件人
    sender = driver.find_element_by_xpath('//android.widget.Image[@text="1"]/parent::android.view.View/following-sibling::android.view.View[1]').get_attribute('name')
    allure.attach("发件人:%s" % sender)
    # 发件邮箱
    fromMail = driver.find_element_by_xpath('//android.view.View[@text="邮箱:"]/following-sibling::android.view.View[1]/android.view.View[2]').get_attribute('name')
    allure.attach("发件邮箱:%s" % fromMail)
    # assert fromMail == param['fromMail']
    # 发件时间
    sendDate = driver.find_element_by_id('com.asiainfo.android:id/tv_message_detail_list_item_date').text
    allure.attach("发件时间:%s" % sendDate)
    assert param['sendDate'] in sendDate
    # title
    mailTitle = driver.find_element_by_id('com.asiainfo.android:id/btn_message_detail_list_item_subject').text
    allure.attach("邮件主题:%s" % mailTitle)
    assert mailTitle == param['mailTitle']
    # 邮件内容
    # mailTitle = driver.find_element_by_xpath('//android.widget.Image[@text="1"]/parent::android.view.View/preceding-sibling::android.view.View[4]').get_attribute('name')
    # driver.find_element_by_xpath('//android.widget.Image[@text="1"]/parent::android.view.View/preceding-sibling::android.view.View').get_attribute('name')
    allure.attach("内容:%s" % mailTitle)
    param['mailTitle'] = mailTitle
    # durationTimes = dill.loads(get('timeLoads'))
    # durationTimes.append(durationTime)
    # set('timeLoads', dill.dumps(durationTimes))
    client.set('param', dill.dumps(param))


def test_backMailDetail(driver,param):
    allure.dynamic.title("返回")
    print (param)
    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_back'))
    driver.find_element_by_id('com.asiainfo.android:id/iv_back').click()

def test_replyMailDetail(driver,param):
    allure.dynamic.title("回复")
    print (param)
    # 内容加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/btnMessageDetailReplyOne'))
    # 点击回复
    driver.find_element_by_id('com.asiainfo.android:id/btnMessageDetailReplyOne').click()


def test_transferMailDetail(driver,param):
    allure.dynamic.title("转发")
    print (param)
    # 内容加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/btnMessageDetailTransfer'))
    # 点击回复
    driver.find_element_by_id('com.asiainfo.android:id/btnMessageDetailTransfer').click()

def test_find(driver,param):
    allure.dynamic.title("查找对应邮件:%s"%param['text'])
    print (param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[@text="%s"]'%param['text']))

    assert True
    print("已查到邮件：%s"%param['findMail'])


def test_star(driver,param):
    allure.dynamic.title("点击星标")
    print (param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_star'))
    driver.find_element_by_id('com.asiainfo.android:id/iv_star').click()
    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"星标")]'))

    assert True
    print("已点击星标")

def test_textChange(driver,param):
    allure.dynamic.title("点击文本大小")
    print (param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/mImage_textChange'))
    driver.find_element_by_id('com.asiainfo.android:id/mImage_textChange').click()

    assert True
    print("已点击文本大小")

def test_slideTextChange(driver,param):
    allure.dynamic.title("滑动调节大小")
    print (param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/mSeekBar'))
    driver.find_element_by_id('com.asiainfo.android:id/mSeekBar').click()
    assert True
    print("滑动调节大小")

def test_replyAll(driver,param):
    allure.dynamic.title("全部回复")
    print (param)
    # 等待并点击全部回复
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/llReplyAllDispacher'))
    driver.find_element_by_id('com.asiainfo.android:id/llReplyAllDispacher').click()
    # 等待填写回复内容
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/etMailReplayAll'))
    driver.find_element_by_id('com.asiainfo.android:id/etMailReplayAll').send_keys('全部回复')
    # 等待发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/btnMessageDetailSend'))
    driver.find_element_by_id('com.asiainfo.android:id/btnMessageDetailSend').click()


def test_check_star(driver,param):
    allure.dynamic.title("检查图标:%s"%param['mailType'])
    print (param)
    # 等待并点击发送
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_star'))
    star = driver.find_element_by_id('com.asiainfo.android:id/iv_star')
    extend = Appium_Extend(driver)
    # 图片路径
    path = ''
    if '星标' == param['mailType']:
        path = "d:\\screen\\detail_star.png"
    elif '非星标' == param['mailType']:
        path = "d:\\screen\\detail_unstar.png"

    # 加载回复图标
    load = extend.load_image(path)
    # 比较
    result = extend.get_screenshot_by_element(star).same_as(load, 0)
    assert result
    print("检查正常")

