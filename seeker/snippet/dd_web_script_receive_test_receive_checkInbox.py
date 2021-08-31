#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

import allure
from selenium.webdriver.support.wait import WebDriverWait

# 查看邮箱A发送的邮件邮箱B是否收到(收件箱)
def test_receive_checkinbox(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    cb = driver.find_elements_by_css_selector("span[class='subject']")
    for c in cb:
        if c.text == param['subject']:
        # if c.text == "小兔子":
            print("新邮件已收到！")

# （已设置邮箱a发送的邮件都到垃圾邮件）查看a发送的邮件是否在邮箱b垃圾邮件中
def test_receive_checkspam(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx2 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx2:
        if i.text == "其他文件夹":
            i.click()
    sjx3 = driver.find_elements_by_css_selector("div[class='cnt']")
    for j in sjx3:
        if j.text == "垃圾邮件":
            j.click()
    time.sleep(1)
    cb2 = driver.find_elements_by_css_selector("span[class='subject']")
    for c in cb2:
        if c.text == param['spamsubject']:
        # if c.text == "小兔子":
            print("已加入垃圾邮件！")