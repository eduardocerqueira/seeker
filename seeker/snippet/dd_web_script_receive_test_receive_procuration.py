#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

import allure
from selenium.webdriver.support.wait import WebDriverWait

# 添加一个代收邮箱账号
def test_receive_procuration(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 进入设置页
    driver.find_element_by_xpath("/html/body/section/aside/div[6]/i").click()
    driver.implicitly_wait(10)
    driver.find_element_by_link_text("高级功能").click()
    driver.find_element_by_link_text("代收邮箱设置").click()
    try:
        WebDriverWait(driver, 3, 0.1).until(
            lambda driver: driver.find_element_by_css_selector("i[class='icontabclose iconfont']"))
        driver.find_element_by_css_selector("i[class='icontabclose iconfont']").click()
        time.sleep(1)
        driver.find_element_by_css_selector(".u-dialog-btns > .u-btn-primary").click()
    except:
        print("无代收邮箱，不需要删除！")
    # 添加代收邮箱信息页面
    driver.find_element_by_css_selector("i[class='iconadd iconfont']").click()
    driver.find_element_by_name("username").send_keys(param['uname'])
    driver.find_element_by_name("password").send_keys(param['upass'])
    # driver.find_element_by_name("server").send_keys("pop.wo.cn")
    # 收取最近一周的邮件
    driver.find_elements_by_css_selector("i[class='radio']")[0].click()
    # 收件箱
    driver.find_elements_by_css_selector("i[class='radio']")[6].click()
    #保存更改
    driver.find_element_by_id("saveBtn").click()
    try:
        WebDriverWait(driver, 3, 0.1).until(
            lambda driver: driver.find_element_by_css_selector("div[class='u-dialog-message']"))
        m = driver.find_element_by_css_selector("div[class='u-dialog-message']").text
        print(m)
        if m == "代收邮箱已存在":
            driver.find_element_by_xpath("/html/body/div[6]/div/div[3]/div/button").click()
            driver.find_element_by_xpath(
            "/html/body/section/article/section/div[2]/div[1]/section/article/div[2]/div/section/div/div/section/div[1]/div[2]/form/div[1]/button[3]").click()
    except:
        return

# 添加两个代收邮箱账号
def test_receive_procuration2(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 进入设置页
    driver.find_element_by_xpath("/html/body/section/aside/div[6]/i").click()
    driver.implicitly_wait(10)
    driver.find_element_by_link_text("高级功能").click()
    driver.find_element_by_link_text("代收邮箱设置").click()
    # 添加代收邮箱信息页面
    driver.find_element_by_css_selector("i[class='iconadd iconfont']").click()
    driver.find_element_by_name("username").send_keys(param['uname2'])
    driver.find_element_by_name("password").send_keys(param['upass2'])
    # driver.find_element_by_name("server").send_keys("pop.wo.cn")
    # 收取最近一周的邮件
    driver.find_elements_by_css_selector("i[class='radio']")[0].click()
    # 收件箱
    driver.find_elements_by_css_selector("i[class='radio']")[6].click()
    # 保存更改
    driver.find_element_by_id("saveBtn").click()
    try:
        WebDriverWait(driver, 3, 0.1).until(
            lambda driver: driver.find_element_by_css_selector("div[class='u-dialog-message']"))
        m = driver.find_element_by_css_selector("div[class='u-dialog-message']").text
        print(m)
        if m == "代收邮箱已存在":
            driver.find_element_by_xpath("/html/body/div[6]/div/div[3]/div/button").click()
            driver.find_element_by_xpath(
                "/html/body/section/article/section/div[2]/div[1]/section/article/div[2]/div/section/div/div/section/div[1]/div[2]/form/div[1]/button[3]").click()
    except:
        return
# 收取所有代收邮箱邮件
def test_receive_procuration_collect(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 进入设置页
    driver.find_element_by_xpath("/html/body/section/aside/div[6]/i").click()
    driver.implicitly_wait(10)
    driver.find_element_by_link_text("高级功能").click()
    driver.find_element_by_link_text("代收邮箱设置").click()
    driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[2]/section/article/div[2]/div[2]/section/div/div/section/div[1]/div[1]/div[2]/div[1]/button[1]").click()
    time.sleep(3)
    y = driver.find_element_by_css_selector("span[class='f-ib f-fl']").text
    if y == "已完成":
        print("代收邮件收取成功！")
    else:
        print("代收邮箱收取失败！请核查！")
    driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[2]/section/article/div[2]/div[2]/section/div/div/section/div[1]/div[3]/div/button").click()

# 查看当前邮件代收情况
def test_receive_procuration_check(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 进入设置页
    driver.find_element_by_xpath("/html/body/section/aside/div[6]/i").click()
    driver.implicitly_wait(10)
    driver.find_element_by_link_text("高级功能").click()
    driver.find_element_by_link_text("代收邮箱设置").click()
    driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[2]/section/article/div[2]/div[2]/section/div/div/section/div[1]/div[1]/div[2]/div[1]/button[2]").click()
    time.sleep(1)
    driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[2]/section/article/div[2]/div[2]/section/div/div/section/div[1]/div[3]/div/button").click()
