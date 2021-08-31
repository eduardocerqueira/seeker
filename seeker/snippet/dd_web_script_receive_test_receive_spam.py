#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

import allure
from selenium.webdriver.support.wait import WebDriverWait


#收件列表页-举报为垃圾邮件
def test_receive_spam(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #收件列表
    sjx=driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx:
        if i.text=="收件箱":
            i.click()
    driver.implicitly_wait(10)
    #选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()


    #点击“更多”
    gd=driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[3].click()
    #选择“举报”
    driver.find_element_by_link_text("举报").click()
    #举报为垃圾邮件
    driver.find_element_by_css_selector("button[class ='u-btn u-btn-primary']").click()

    #判断举报的邮件是否成功
    # try:
    #      WebDriverWait(driver, 2, 0.1).until(
    #          lambda driver: driver.find_element_by_css_selector("div[class='body']"))
    #     print(driver.find_element_by_css_selector("div['body']").text)
    # except:
    #     print("举报邮件失败！")

    #打开垃圾邮件列表--其他文件夹
    # qt=driver.find_elements_by_css_selector("div[class='cnt']")
    # for j in qt:
    #     if j.text == "其他文件夹":
    #         j.click()
    #
    # #打开垃圾邮件
    # for s in qt:
    #     if s.text == "垃圾邮件":
    #         s.click()

#邮件详情页-举报为垃圾邮件
def test_receive_detail_spam(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx:
        if i.text == "收件箱":
            i.click()
    driver.implicitly_wait(10)
    # 选择一封邮件查看邮件详情
    cb2 = driver.find_elements_by_css_selector("span[class='subject']")
    cb2[0].click()
    # driver.implicitly_wait(10)
    time.sleep(2)
    # 点击“更多”
    driver.find_element_by_css_selector('.u-btns:nth-child(4) > .u-btn:nth-child(3)').click()
    # 选择“举报”
    driver.find_element_by_link_text("举报").click()
    # 举报为垃圾邮件
    time.sleep(2)
    driver.find_element_by_css_selector('.u-dialog-btns > .u-btn-primary').click()

# 收件列表页—将发件人的历史邮件移到垃圾邮件
def test_receive_spam_history(param, driver):
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
    # 选择来信分类
    driver.find_element_by_link_text("来信分类").click()
    fl=driver.find_elements_by_css_selector("a[class='u-select-trigger']")
    for f in fl:
        if f.text == "收件箱":
            f.click()

    # 选择将历史邮件放入垃圾邮件
    xlxz=driver.find_elements_by_css_selector("li[class='u-select-item']")
    for x in xlxz:
        if x.text == "垃圾邮件":
            x.click()
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[1]/section/article/div[2]/div[1]/div/div[5]/div/div[2]/div/div/div[6]/i").click()
    # 确定
    qd1 = driver.find_elements_by_css_selector("button[class ='u-btn u-btn-primary']")
    qd1[0].click()


# 邮件详情页—将发件人的历史邮件移到垃圾邮件
def test_receive_detail_spam_history(param, driver: object):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #收件列表
    sjx1=driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text=="收件箱":
            i.click()
    driver.implicitly_wait(10)
    # 选择一封邮件查看详情
    cb2 = driver.find_elements_by_css_selector("span[class='subject']")
    cb2[0].click()
    time.sleep(1)
    # 点击“更多”
    driver.find_element_by_css_selector('.u-btns:nth-child(4) > .u-btn:nth-child(3)').click()
    # 选择来信分类
    time.sleep(1)
    driver.find_element_by_link_text("来信分类").click()
    fl=driver.find_elements_by_css_selector("a[class='u-select-trigger']")
    for f in fl:
        if f.text == "收件箱":
            f.click()
    time.sleep(1)
    # 选择将历史邮件放入垃圾邮件
    xlxz=driver.find_elements_by_css_selector("li[class='u-select-item']")
    for x in xlxz:
        if x.text == "垃圾邮件":
            x.click()
    time.sleep(1)
    driver.find_element_by_css_selector('.filter-rules > .checkbox').click()
    # 确定
    time.sleep(1)
    driver.find_element_by_css_selector('.u-dialog-btns > .u-btn-primary').click()

# 垃圾邮件列表—这不是垃圾邮件
def test_receive_nospam(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 打开其他文件夹-垃圾邮件
    menu = driver.find_elements_by_css_selector("div[class='cnt']")
    for m in menu:
        if m.text == "其他文件夹":
            m.click()
    ljyj = driver.find_elements_by_css_selector("div[class='cnt']")
    for l in ljyj:
        if l.text == "垃圾邮件":
            l.click()
    time.sleep(1)
    myyj = driver.find_element_by_css_selector("span[class='number']").text
    if myyj == "0":
        print("垃圾邮件为0！")
        return
    else:
        # 选择一封邮件
        cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
        cb1[1].click()
        # 点击“更多”
        gd=driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
        gd[3].click()
        # 选择来信分类
        driver.find_element_by_link_text("这不是垃圾邮件").click()
        driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[1]/section/article/div[2]/div/div/div[5]/div/div[3]/div/button[1]").click()

# 垃圾邮件详情页—这不是垃圾邮件
def test_receive_detail_nospam(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 打开其他文件夹-垃圾邮件
    menu1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for m in menu1:
        if m.text == "其他文件夹":
            m.click()
    ljyj1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for l in ljyj1:
        if l.text == "垃圾邮件":
            l.click()
    time.sleep(1)
    myyj1 = driver.find_element_by_css_selector("span[class='number']").text
    if myyj1 == "0":
        print("垃圾邮件为0！")
        return
    else:
        cb3 = driver.find_elements_by_css_selector("span[class='subject']")
        cb3[0].click()
        time.sleep(1)
        # 点击“更多”
        driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[1]/section/article/div[2]/div[2]/section/div[1]/div[3]/span[3]").click()
        # 选择这不是垃圾邮件
        time.sleep(1)
        driver.find_element_by_link_text("这不是垃圾邮件").click()
        time.sleep(1)
        driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[1]/section/article/div[2]/div[2]/section/div[4]/div/div[3]/div/button[1]").click()