#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time
import allure
from selenium.webdriver.support.wait import WebDriverWait

# 收件箱选择一封邮件删除（到已删除列表）
def test_receive_deletemail(param,driver):
    driver.implicitly_wait(10)
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    # 获取第一封邮件的主题
    cb2 = driver.find_elements_by_css_selector("span[class='subject']")
    xiao = cb2[0].text
    # 选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()
    # 点击删除
    driver.find_element_by_css_selector("span[class='u-btn u-btn-default j-maildelete']").click()
    time.sleep(1)
    # 去已删除列表查看
    sjx2 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx2:
        if i.text == "其他文件夹":
            i.click()
    sjx3 = driver.find_elements_by_css_selector("div[class='cnt']")
    for j in sjx3:
        if j.text == "已删除":
            j.click()
    time.sleep(1)
    #已删除列表邮件封数
    numb1 = driver.find_element_by_css_selector("span[class='number']").text
    number = int(numb1)
    # 已删除列表的邮件主题list
    sc = driver.find_elements_by_css_selector("span[class='subject']")
    for s in sc:
        if s.text == xiao:
            tu = 1
        elif number > 20:
            tu = 2
        elif number <= 20:
            tu = 3
    # 判断刚才删除的邮件是否存在
    if tu == 1:
        print("删除成功！邮件已在已删除列表中！")
    elif tu == 2:
        print("刚才删除的邮件未在已删除列表第一页！")
    elif tu == 3:
        print("未找到刚才删除的邮件！")

# 收件箱选择一封邮件彻底删除
def test_receive_cddeletemail(param,driver):
    driver.implicitly_wait(10)
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    # 获取第一封邮件的主题
    cb2 = driver.find_elements_by_css_selector("span[class='subject']")
    xiao = cb2[0].text
    # 选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()
    # 点击“更多”
    gd=driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[3].click()
    #选择彻底删除
    driver.find_element_by_link_text("彻底删除").click()
    driver.find_element_by_xpath("/html/body/div[5]/div/div[3]/div/button[1]").click()

#     将已删除列表邮件彻底删除
def test_receive_completelydelete(param,driver):
    driver.implicitly_wait(10)
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 已删除列表
    sjx2 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx2:
        if i.text == "其他文件夹":
            i.click()
    sjx3 = driver.find_elements_by_css_selector("div[class='cnt']")
    for j in sjx3:
        if j.text == "已删除":
            j.click()
    # 查看已删除列表是否有邮件
    numb = driver.find_element_by_css_selector("span[class='number']").text
    number = int(numb)
    time.sleep(1)
    if number > 0:
        # 选择所有邮件
        driver.find_element_by_css_selector(".icondown:nth-child(3)").click()
        driver.find_element_by_xpath("//a[contains(text(),'所有')]").click()
        #点击删除
        driver.find_element_by_css_selector("span[class='u-btn u-btn-default j-maildelete']").click()
        time.sleep(1)
        driver.find_element_by_xpath("/html/body/div[5]/div/div[3]/div/button[1]").click()
    else:
        print("已删除列表没有邮件！")
        return

# 快捷按钮，清空已删除列表
def test_receive_cleardelete(param,driver):
    driver.implicitly_wait(10)
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 已删除列表
    sjx2 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx2:
        if i.text == "其他文件夹":
            i.click()
    sjx3 = driver.find_elements_by_css_selector("div[class='cnt']")
    for j in sjx3:
        if j.text == "已删除":
            j.click()
    time.sleep(1)
    # 查看已删除列表是否有邮件
    numb = driver.find_element_by_css_selector("span[class='number']").text
    number = int(numb)
    time.sleep(1)
    if number > 0:
        driver.find_element_by_css_selector("i[class='j-trash iconfont iconrecover']").click()
        driver.find_element_by_xpath("/html/body/div[5]/div/div[3]/div/button[1]").click()
        print("已删除列表已清空！")
    else:
        print("已删除列表没有邮件！")
        return