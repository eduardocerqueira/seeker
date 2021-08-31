#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

from selenium.webdriver.support.wait import WebDriverWait

# 邮件列表页-拒收邮件（加入黑名单）
def test_receive_blacklist(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #收件列表
    sjx1=driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text=="收件箱":
            i.click()
    driver.implicitly_wait(10)
    # 选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()

    # 点击“更多”
    gd=driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[3].click()
    # 选择拒收邮件（黑名单）
    driver.find_element_by_link_text("拒收邮件").click()
    # 加入黑名单
    qd1 = driver.find_elements_by_css_selector("button[class ='u-btn u-btn-primary']")
    qd1[0].click()

    try:
        WebDriverWait(driver, 20, 0.1).until(
            lambda driver: driver.find_element_by_css_selector("div[class='body']").text)
        print(driver.find_element_by_css_selector("div[class='body']").text)
    except:
        print("加入黑名单失败！")

    # # 已选择的邮件的发信人
    # hmdyh=driver.find_element_by_css_selector("span[class='fromto j-from']")
    #
    # # 查看加入黑名单详情
    # driver.find_element_by_css_selector("span[class='extend-btn']").click()
    #
    # if hmdyh.get_attribute("data-email") == driver.find_element_by_tag_name("dd").text:
    #     print("操作成功")
    # else:
    #     print("加入黑名单的用户名错误")
    #
    # qd4=driver.find_elements_by_css_selector("button[class ='u-btn u-btn-primary']")
    # qd4[len(qd4)-1].click()
# 邮件详情页-拒收邮件（加入黑名单）
def test_receive_detail_blacklist(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #收件列表
    sjx1=driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text=="收件箱":
            i.click()
    time.sleep(1)
    # 选择一封邮件查看详情
    cb2 = driver.find_elements_by_css_selector("span[class='subject']")
    cb2[0].click()
    time.sleep(2)
    # 点击“更多”
    driver.find_element_by_css_selector('.u-btns:nth-child(4) > .u-btn:nth-child(3)').click()
    # 选择来信分类
    time.sleep(1)
    driver.find_element_by_link_text("拒收邮件").click()
    time.sleep(1)
    driver.find_element_by_css_selector('.u-dialog-btns > .u-btn-primary').click()

#邮件设置--添加黑名单
def test_setup_blacklist(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #进入设置页
    driver.find_element_by_xpath("/html/body/section/aside/div[6]/i").click()
    driver.implicitly_wait(10)
    driver.find_element_by_link_text("安全设置").click()

    driver.find_element_by_link_text("黑 名 单").click()
    yxm=driver.find_elements_by_css_selector("input[class='j-new-rule']")
    yxm[0].clear()
    yxm[0].send_keys(param['blackname'])
    tj=driver.find_elements_by_css_selector("button[class='u-btn u-btn-primary']")
    for t in tj:
        if t.text == "添加":
            t.click()
    time.sleep(2)
    hmd = driver.find_elements_by_xpath('//section/div/div/div[2]/table/tbody/tr/td')
    for h in hmd:
        if h.text == param['blackname']:
            print("添加黑名单成功")
            print(h.text)

    time.sleep(2)
    yc = driver.find_elements_by_xpath("//a[contains(text(),'移除')]")
    for y in yc:
        if y.get_attribute("data-name") == param['blackname']:
            y.click()
    driver.find_element_by_css_selector('.u-dialog-btns > .u-btn-primary').click()

# 清空黑名单
def test_setup_clearblacklist(param,driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #进入设置页
    driver.find_element_by_xpath("/html/body/section/aside/div[6]/i").click()
    driver.implicitly_wait(10)
    driver.find_element_by_link_text("安全设置").click()

    driver.find_element_by_link_text("黑 名 单").click()
    # 判断清空黑名单按钮是否禁用
    dis = driver.find_element_by_css_selector("button[data-action='removeAll']").get_attribute("class")
    if dis == "u-btn u-btn-default disabled":
        print("黑名单列表为空，不用清空！")
        return
    else:
        driver.find_element_by_css_selector(".f-ib > .u-btn-default").click()
        driver.find_element_by_css_selector(".u-dialog-btns > .u-btn-primary").click()
