#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import allure
from selenium.webdriver.support.ui import WebDriverWait
from appium.webdriver.common.touch_action import TouchAction
from pymemcache.client.base import Client
import dill
from autotest.AppiumExtend import Appium_Extend
import math
import time

client = Client(('localhost', 11211))





def test_clickOk(driver,param):
    allure.dynamic.story("去root警告")
    print (param)
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('android:id/button1'))
    driver.find_element_by_id('android:id/button1').click()

def test_checkMail(driver,param):
    allure.dynamic.title("检查邮件信息")
    print(param)
    start = time.time()
    WebDriverWait(driver, 40, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/message_list_item_layout'))
    # durationTime = time.time()-start
    # allure.attach("加载时间:%s"%(durationTime))
    # 摘要
    abstract = driver.find_elements_by_id('com.asiainfo.android:id/preview')[param['id']].text
    allure.attach("摘要类容:%s" % abstract)
    # assert "摘要和详情，请点击查看" == abstract
    # title
    mailTitle = driver.find_elements_by_id('com.asiainfo.android:id/subject')[param['id']].text
    allure.attach("标题类容:%s" % mailTitle)
    param['mailTitle'] = mailTitle
    # 发件地址
    fromMail = driver.find_elements_by_id('com.asiainfo.android:id/from')[param['id']].text
    allure.attach("发件地址:%s" % fromMail)
    param['fromMail'] = fromMail
    # 发件时间
    sendDate = driver.find_elements_by_id('com.asiainfo.android:id/date')[param['id']].text
    allure.attach("发件时间:%s" % sendDate)
    param['sendDate'] = sendDate
    # durationTimes = dill.loads(get('time'))
    # durationTimes.append(durationTime)
    # set('time', dill.dumps(durationTimes))
    # print(param)
    client.set('param', dill.dumps(param))

def test_checkMailIcon(driver,param):
    print (param)
    allure.dynamic.title("检查邮件图标-%s"%param['mailType'])
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/slide_id_front_view'))

    abstract = driver.find_elements_by_id('com.asiainfo.android:id/slide_id_front_view')[param['id']]

    result = False
    # 图标
    if '已读' == param['mailType'] :
        try:
            abstract = abstract.find_element_by_id('com.asiainfo.android:id/iv_unread')
        except Exception:
            result = True
    elif '星标' == param['mailType'] :
        abstract = abstract.find_element_by_id('com.asiainfo.android:id/btn_star')

    elif '非星标' == param['mailType'] :
        try:
            abstract = abstract.find_element_by_id('com.asiainfo.android:id/btn_star')
        except Exception:
            result = True
    else :
        abstract = abstract.find_element_by_id('com.asiainfo.android:id/iv_unread')
    if '已读' != param['mailType'] and '非星标' != param['mailType']:
        extend = Appium_Extend(driver)
        # 图片路径
        path = ''
        # 区别标签
        if '未读' == param['mailType']:
            path = "/Users/admin/Downloads/image/unread.png"
        elif '回复' == param['mailType']:
            path = "/Users/admin/Downloads/image/reply.png"
        elif '转发' == param['mailType']:
            path = "/Users/admin/Downloads/image/forward.png"
        elif '回复转发' == param['mailType']:
            path = "/Users/admin/Downloads/image/replyForword.png"
        elif '星标' == param['mailType']:
            path = "/Users/admin/Downloads/image/star.png"
        # 加载回复图标
        load = extend.load_image(path)

        # 比较
        result = extend.get_screenshot_by_element(abstract).same_as(load, 0)

    assert result

def test_clickMail(driver,param):
    print (param)
    allure.dynamic.title("点击第%s条"%param['id'])
    # driver.background_app(1)
    # time.sleep(3)
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/slide_id_front_view'))
    id = param['id']
    point = id%4
    for i in range(math.floor(id/4)):
        # 向上滑动4
        driver.swipe(0, 923, 0, 163, 3000)
        time.sleep(1)
    # 读取list
    list = driver.find_elements_by_id('com.asiainfo.android:id/slide_id_front_view')

    list[point].click()

def test_clickSearch(driver,param):
    print (param)
    allure.dynamic.title("点击搜索")
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_title_plus_icon'))
    # 点击第一条邮件
    driver.find_element_by_id('com.asiainfo.android:id/iv_title_plus_icon').click()


def test_longPressMail(driver,param):
    print (param)
    allure.dynamic.title("长按点击点击第%s条"%param['id'])
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/slide_id_front_view'))
    # 点击第一条邮件
    el = driver.find_elements_by_id('com.asiainfo.android:id/slide_id_front_view')[param['id']]

    TouchAction(driver).long_press(el).release().perform()



def test_countList(driver,param):
    allure.dynamic.title("翻页记录列表长度")
    print (param)
    driver.background_app(1)
    time.sleep(3)
    i = 6
    length = 4
    while i > 0:
        # 向上滑动4
        driver.swipe(0, 923, 0, 163, 3000)
        time.sleep(1)
        try:
            driver.find_element_by_xpath('//*[@text="加载更多邮件" and @resource-id = "com.asiainfo.android:id/main_text"]')
            break
        except:
            i = i -1
            length = length + 4

    lastPage = driver.find_elements_by_id('com.asiainfo.android:id/subject')
    length = length+len(lastPage)
    print(length)
    allure.attach("最终长度为%s"%length)

def test_dropDownMenu(driver,param):
    allure.dynamic.title("打开下拉框:%s"%param['mailType'])
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/tv_actionbar_lefttitle'))
    # 点击收件箱
    driver.find_element_by_id('com.asiainfo.android:id/tv_actionbar_lefttitle').click()

def test_screenMail(driver,param):
    print (param)
    allure.dynamic.title("筛选邮件:%s" % param['mailType'])
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/mText_all_edit'))
    if '全部' == param['mailType']:
        driver.find_element_by_id('com.asiainfo.android:id/mRelative_all_email').click()
    elif '未读'== param['mailType']:
        driver.find_element_by_id('com.asiainfo.android:id/mRelative_unread_email').click()
    elif '星标' == param['mailType']:
        driver.find_element_by_id('com.asiainfo.android:id/mRelative_star_email').click()
    elif '批量编辑' == param['mailType']:
        driver.find_element_by_id('com.asiainfo.android:id/mText_all_edit').click()
    elif '关闭' == param['mailType']:
        driver.find_element_by_id('com.asiainfo.android:id/mBottom_bgView').click()

    allure.attach("选择%s"%param['mailType'])

def test_listTypeCheck(driver,param):
    print (param)
    allure.dynamic.title("列表筛选邮件:%s"%param['mailType'])
    # 图片支持
    extend = Appium_Extend(driver)

    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_elements_by_id('com.asiainfo.android:id/subject'))

    # 获取当前列表
    list = driver.find_elements_by_id('com.asiainfo.android:id/subject')
    # 获取当前标签列表
    unreadList = driver.find_elements_by_id('com.asiainfo.android:id/iv_unread')

    # 图片路径
    path = ''
    # 区别标签
    if '未读' == param['mailType']:
        path = "/Users/admin/Downloads/image/unread.png"
    elif '回复'== param['mailType']:
        path = "/Users/admin/Downloads/image/reply.png"
    elif '转发' == param['mailType']:
        path = "/Users/admin/Downloads/image/forward.png"
    elif '回复转发' == param['mailType']:
        path = "/Users/admin/Downloads/image/replyForword.png"

    # 断言列表长度与标签是否一致
    # assert len(list) == len(unreadList)
    allure.attach("列表长%s" %len(list))

    # 对比标签是否正确
    for el in unreadList:
        # 加载回复图标
        load = extend.load_image(path)
        # 比较
        result = extend.get_screenshot_by_element(el).same_as(load, 0)
        allure.attach("比对结果%s" % result)
        assert result

def test_listTypeCheckStart(driver,param):
    allure.dynamic.title("列表筛选邮件:星标")
    print (param)
    # 图片支持
    extend = Appium_Extend(driver)

    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_elements_by_id('com.asiainfo.android:id/subject'))

    # 获取当前列表
    list = driver.find_elements_by_id('com.asiainfo.android:id/subject')
    # 获取当前标签列表
    starList = driver.find_elements_by_id('com.asiainfo.android:id/btn_star')

    # 图片路径
    path = "/Users/admin/Downloads/image/star.png"

    # 断言列表长度与标签是否一致
    assert len(list) == len(starList)
    allure.attach("列表长%s" %len(list))

    # 对比标签是否正确
    for el in starList:
        # 加载回复图标
        load = extend.load_image(path)
        # 比较
        result = extend.get_screenshot_by_element(el).same_as(load, 0)
        allure.attach("比对结果%s" % result)
        assert result


def test_leftSlipDelete(driver,param):
    allure.dynamic.title("左滑删除")
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_left_button'))

    # 获取操作对应的邮件
    testEmail = driver.find_element_by_id('com.asiainfo.android:id/message_list_item_layout')
    # 获取邮件名称
    emailSubject = testEmail.find_element_by_id('com.asiainfo.android:id/subject')
    emailDeleted = emailSubject.text
    # 滑动定位
    xSize = testEmail.size['width'] - 2
    ySize = testEmail.location['y'] + testEmail.size['height'] / 2

    # 左滑动
    driver.swipe(xSize, ySize, xSize / 2, ySize, 3000)
    # TouchAction(driver).press(x=xSize, y=ySize).move_to(x=xSize / 2, y=ySize).move_to(x=xSize / 4, y=ySize).release().perform()
    # 点击删除
    driver.find_element_by_xpath('//android.widget.TextView[@text="删除"]').click()
    # param['findMail'] = emailDeleted
    # set('param', dill.dumps(param))

def test_rightSlip(driver,param):
    allure.dynamic.title("右滑闪操:%s"%param['mailType'])
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_left_button'))
    # 获取操作对应的邮件
    testEmail = driver.find_elements_by_id('com.asiainfo.android:id/message_list_item_layout')[param['id']]
    # 获取邮件名称
    emailSubject = testEmail.find_element_by_id('com.asiainfo.android:id/subject')
    emailSubjectText = emailSubject.text
    # 滑动定位
    xSize = testEmail.size['width'] - 2
    ySize = testEmail.location['y'] + testEmail.size['height'] / 2
    # 右滑动
    driver.swipe(xSize / 2, ySize, xSize, ySize, 3000)
    # TouchAction(driver).press(x=xSize / 4, y=ySize).move_to(x=xSize / 2, y=ySize).move_to(x=xSize, y=ySize).release().perform()
    # 点击闪操
    driver.find_element_by_xpath('//android.widget.TextView[@text="闪操"]').click()

    # param['findMail'] = emailSubjectText
    # set('param', dill.dumps(param))

def test_openSidebar(driver,param):
    allure.dynamic.title("打开侧边栏")
    print (param)
    # 等待收件箱加载
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_left_button'))
    driver.find_element_by_id('com.asiainfo.android:id/iv_left_button').click()


def test_clickWrite(driver,param):
    allure.dynamic.story("点击写邮件")
    print (param)
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_write_mail'))
    driver.find_element_by_id('com.asiainfo.android:id/iv_write_mail').click()


