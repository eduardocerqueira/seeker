#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

import allure
from selenium.webdriver.support.wait import WebDriverWait

# 所有邮件标为已读
def test_receive_markread(param,driver):
    driver.implicitly_wait(10)
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    driver.find_element_by_css_selector(".icondown:nth-child(3)").click()
    driver.find_element_by_xpath("//a[contains(text(),'所有')]").click()
    driver.find_element_by_css_selector(".u-btns:nth-child(3) > .u-btn:nth-child(2)").click()
    driver.find_element_by_css_selector(".u-menu-show > li:nth-child(1) > a").click()

# 全部标为已读（快捷链接）
def test_receive_markreadlink(param,driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    try:
        WebDriverWait(driver, 5, 0.1).until(
            lambda driver: driver.find_elements_by_link_text("全部设为已读"))
        driver.find_element_by_link_text("全部设为已读").click()
    except:
        return

#  所有邮件标为未读
def test_receive_markunread(param,driver):
    driver.implicitly_wait(10)
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    driver.find_element_by_css_selector(".icondown:nth-child(3)").click()
    driver.find_element_by_xpath("//a[contains(text(),'所有')]").click()
    driver.find_element_by_css_selector(".u-btns:nth-child(3) > .u-btn:nth-child(2)").click()
    driver.find_element_by_css_selector("li:nth-child(2) > a:nth-child(2)").click()