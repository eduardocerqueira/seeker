#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# coding=utf-8

import allure
from selenium.webdriver.support.ui import WebDriverWait
import os
import time

def test_agreePrivacy(driver,param):
    allure.dynamic.title("同意用户隐私协议")
    print (param)
    # 等待并点击已阅读
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/ibReadedAndAgreed'))
    driver.find_element_by_id('com.asiainfo.android:id/ibReadedAndAgreed').click()
    # 点击确定
    driver.find_element_by_id('com.asiainfo.android:id/mText_popupDetermine').click()


def test_choiceWo(driver,param):
    allure.dynamic.title("选择wo邮箱")
    print (param)
    # 等待并点击沃邮箱
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/iv_womail_icon'))
    driver.find_element_by_id('com.asiainfo.android:id/iv_womail_icon').click()

def test_choice163(driver,param):
    allure.dynamic.title("选择163邮箱")
    print (param)
    # 等待并点击沃邮箱
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[@text="163邮箱"]'))
    driver.find_element_by_xpath('//*[@text="163邮箱"]').click()

def test_choiceQQ(driver,param):
    allure.dynamic.title("选择QQ邮箱")
    print (param)
    # 等待并点击沃邮箱
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[@text="QQ邮箱"]'))
    driver.find_element_by_xpath('//*[@text="QQ邮箱"]').click()

def test_login(driver,param):
    allure.dynamic.title("登录")
    print (param)
    # 等待并填写邮箱/手机号
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/edit_othermail_user_mail'))
    driver.find_element_by_id('com.asiainfo.android:id/edit_othermail_user_mail').send_keys(param['phno'])
    # 填写密码
    driver.find_element_by_id('com.asiainfo.android:id/edit_othermail_user_psw').send_keys(param['pwd'])
    # 点击登录
    driver.find_element_by_id('com.asiainfo.android:id/other_mail_login_bt').click()

def test_loginCode(driver,param):
    allure.dynamic.title("验证码登录")
    print (param)
    # 等待并填写邮箱/手机号
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account'))
    driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account').send_keys(param['phno'])
    # 填写密码
    driver.find_element_by_id('com.asiainfo.android:id/et_dynamic_password').send_keys(param['pwd'])
    # 点击登录
    driver.find_element_by_id('com.asiainfo.android:id/other_mail_login_bt').click()



def test_loginAndSwitchData(driver,param):
    allure.dynamic.title("登录与切换网络")
    print (param)
    # 等待并填写邮箱/手机号
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/edit_othermail_user_mail'))
    driver.find_element_by_id('com.asiainfo.android:id/edit_othermail_user_mail').send_keys(param['phno'])
    # 填写密码
    driver.find_element_by_id('com.asiainfo.android:id/edit_othermail_user_psw').send_keys(param['pwd'])
    # 返回
    driver.hide_keyboard()
    # 点击登录
    driver.find_element_by_id('com.asiainfo.android:id/other_mail_login_bt').click()
    os.system('adb shell svc data disable')


def test_resetPwd(driver,param):
    allure.dynamic.title("选择重置密码")
    print (param)
    # 等待并点击重置密码
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/tv_resetPwdAndLogin'))
    driver.find_element_by_id('com.asiainfo.android:id/tv_resetPwdAndLogin').click()


def test_resetPwdLogin(driver,param):
    allure.dynamic.title("重置密码登录")
    print (param)
    # 等待并填写手机号
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account'))
    driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account').send_keys(param['phno'])
    # 点击获取验证码
    driver.find_element_by_id('com.asiainfo.android:id/btn_get_sms_code').click()
    # 通知栏
    driver.open_notifications()
    # 通知栏
    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_id('com.android.systemui:id/delete'))
    driver.find_element_by_id('com.android.systemui:id/delete').click()

    time.sleep(2)

    driver.open_notifications()

    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"，请勿泄露")]'))
    code = driver.find_element_by_xpath('//*[contains(@text,"，请勿泄露")]').text
    code = code[0:code.index('，')]
    # 返回
    driver.press_keycode(4)

    # 填写密码
    WebDriverWait(driver, 60, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_dynamic_password'))
    driver.find_element_by_id('com.asiainfo.android:id/et_dynamic_password').send_keys(code)
    # 点击登录
    driver.find_element_by_id('com.asiainfo.android:id/other_mail_login_bt').click()


def test_resetPwdLoginAndSwitchWLAN(driver,param):
    allure.dynamic.title("重置登录与切换网络")
    print (param)
    ## 等待并填写手机号
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account'))
    driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account').send_keys(param['phno'])
    # 点击获取验证码
    driver.find_element_by_id('com.asiainfo.android:id/btn_get_sms_code').click()
    driver.open_notifications()
    # 通知栏
    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_id('com.android.systemui:id/delete'))
    driver.find_element_by_id('com.android.systemui:id/delete').click()

    time.sleep(2)

    driver.open_notifications()

    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"，请勿泄露")]'))
    code = driver.find_element_by_xpath('//*[contains(@text,"，请勿泄露")]').text
    code = code[0:code.index('，')]
    # 返回
    driver.press_keycode(4)

    # 填写密码
    WebDriverWait(driver, 60, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_dynamic_password'))
    driver.find_element_by_id('com.asiainfo.android:id/et_dynamic_password').send_keys(code)
    # 点击登录
    driver.find_element_by_id('com.asiainfo.android:id/other_mail_login_bt').click()
    # 关闭网络
    driver.set_network_connection(0)


def test_getCode(driver,param):
    allure.dynamic.title("获取验证码")
    print (param)
    # 等待并填写手机号
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account'))
    driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account').send_keys(param['phno'])
    # 点击获取验证码
    driver.find_element_by_id('com.asiainfo.android:id/btn_get_sms_code').click()
    # 通知栏
    driver.open_notifications()

    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_id('com.android.systemui:id/delete'))
    driver.find_element_by_id('com.android.systemui:id/delete').click()

    driver.open_notifications()

    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"，请勿泄露")]'))
    code = driver.find_element_by_xpath('//*[contains(@text,"，请勿泄露")]').text
    code = code[0:code.index('，')]
    # 返回
    driver.press_keycode(4)

    # 填写密码
    WebDriverWait(driver, 60, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_dynamic_password'))
    driver.find_element_by_id('com.asiainfo.android:id/et_dynamic_password').send_keys(code)



def test_checkUnLoginApp(driver,param):
    allure.dynamic.title("检测app错误弹窗")
    print (param)
    # 等待并填写手机号
    WebDriverWait(driver, 30, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"用户名或密码不正确。")]'))
    driver.find_element_by_xpath('//*[contains(@text,"确定")]').click()
    allure.attach("已弹窗")

def test_checkUnLoginServer(driver,param):
    allure.dynamic.title("检测server错误弹窗")
    print (param)
    # 等待并填写手机号
    WebDriverWait(driver, 30, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"系统繁忙，")]'))
    driver.find_element_by_xpath('//*[@text="手动配置"]').click()
    allure.attach("已弹窗")


def test_checkUnLoginPhone(driver,param):
    allure.dynamic.title("检测手机号错误弹窗")
    print (param)
    # 等待并填写手机号
    WebDriverWait(driver, 30, 0.1).until(
        lambda driver: driver.find_element_by_xpath('//*[contains(@text,"系统繁忙，")]'))
    driver.find_element_by_xpath('//*[@text="手动配置"]').click()
    allure.attach("已弹窗")


def test_fillPhone(driver,param):
    allure.dynamic.title("填写手机号")
    print(param)
    # 等待并填写手机号
    WebDriverWait(driver, 10, 0.1).until(
        lambda driver: driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account'))
    driver.find_element_by_id('com.asiainfo.android:id/et_phone_number_or_email_account').send_keys(param['phno'])


def test_clickLogin(driver,param):
    allure.dynamic.title("点击登录")
    print(param)
    # 点击登录
    driver.find_element_by_id('com.asiainfo.android:id/other_mail_login_bt').click()


def test_clickConfirm(driver,param):
    allure.dynamic.title("点击确定")
    print(param)
    # 点击登录
    driver.find_element_by_xpath('//*[@text="确定"]').click()


def test_clickKeyLogin(driver,param):
    allure.dynamic.title("点击一键登录")
    print(param)
    # 点击登录
    driver.find_element_by_xpath('//*[@text="本机号码一键登录"]').click()


def test_clickKeyLoginOffData(driver,param):
    allure.dynamic.title("点击一键登录")
    print(param)
    # 点击登录
    driver.find_element_by_xpath('//*[@text="本机号码一键登录"]').click()
    os.system('adb -s 3HX7N17106006538 shell svc data disable')

