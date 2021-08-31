#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

from selenium.webdriver.support.wait import WebDriverWait


# 点击收件列表第一封邮件
def test_inbox_fristmail(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #收件列表
    sjx1=driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == param["menu"]:
        # if i.text=="收件箱":
            i.click()
    driver.implicitly_wait(10)
    # 选择一封邮件查看详情
    cb2 = driver.find_elements_by_css_selector("span[class='subject']")
    cb2[0].click()

# 收件箱，选择一封邮件置顶
def test_inbox_topmail(param,driver):
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
    # 点击“标记为”
    gd = driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[2].click()
    # 选择置顶邮件
    driver.find_element_by_link_text("置顶邮件").click()

# 收件箱，选择一封邮件取消置顶
def test_inbox_untopmail(param,driver):
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
    # 点击“标记为”
    gd = driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[2].click()
    # 选择置顶邮件
    driver.find_element_by_link_text("取消置顶邮件").click()

# 收件列表移动到—已发送、已删除、垃圾邮件、病毒文件夹
def test_receive_movemail(param,driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    driver.implicitly_wait(10)
    # 选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()
    # 点击“移动到”
    gd = driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[1].click()
    #移动到已发送
    ydd = driver.find_elements_by_xpath("/html/body/section/article/section/div[2]/div/section/article/div[2]/div/div/div[2]/div/div[5]/div[4]/div/span[1]/ul/li/a")
    for y in ydd:
        if y.text == "已发送":
            y.click()
    time.sleep(1)
    # 选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()
    # 点击“移动到”
    gd = driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[1].click()
    #移动到已删除
    for z in ydd:
        if z.text == "已删除":
            z.click()
    time.sleep(1)
    # 选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()
    # 点击“移动到”
    gd = driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[1].click()
    # 移动到垃圾邮件
    for r in ydd:
        if r.text == "垃圾邮件":
            r.click()
    time.sleep(1)
    # 选择一封邮件
    cb1 = driver.find_elements_by_css_selector("i[class='checkbox']")
    cb1[1].click()
    # 点击“移动到”
    gd = driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")
    gd[1].click()
    # 移动到病毒文件夹
    for s in ydd:
        if s.text == "病毒文件夹":
            s.click()
# 查看包含附件的邮件
def test_receive_enclosure(param,driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    # 点击查看
    driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")[0].click()
    # 选择 包含附件
    driver.find_element_by_link_text("包含附件").click()
    time.sleep(1)
    bh = driver.find_element_by_css_selector("div[class='totals-info']").text

    if "包含附件的邮件" in bh:
        print(bh)
    else:
        print("查询失败")
        return
# 查看不包含附件的邮件
def test_receive_no_enclosure(param,driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    # 点击查看
    driver.find_elements_by_css_selector("span[class='u-btn u-btn-default']")[0].click()
    # 选择 包含附件
    driver.find_element_by_link_text("不包含附件").click()
    time.sleep(1)
    bbh = driver.find_element_by_css_selector("div[class='totals-info']").text

    if "不包含附件的邮件" in bbh:
        print(bbh)
    else:
        print("查询失败")
        return

#首页-详情（自助查询）-收信查询
def test_receive_search(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    driver.implicitly_wait(5)
    #点击进入自助查询
    driver.find_element_by_link_text("详情").click()
    driver.implicitly_wait(5)
    driver.find_element_by_link_text("收信查询").click()
    driver.implicitly_wait(5)
    #如果查不到收信记录，打印提示消息
    try:
        WebDriverWait(driver, 20, 0.1).until(
            lambda driver: driver.find_element_by_xpath("//tr[@class='even']"))
    except:
        print("无收信信息,需核实！")

# 翻页功能
def test_receive_pagedown(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    # 列表邮件数量
    num = driver.find_element_by_css_selector("span[class='number']").text
    if int(num) > 40:
        driver.find_element_by_css_selector("span[class='iconfont iconnext']").click()
        time.sleep(1)
        driver.find_element_by_css_selector("span[class='iconfont iconnext']").click()
        time.sleep(1)
        driver.find_element_by_css_selector("span[class='iconfont iconprevious']").click()
    elif int(num) > 20:
        driver.find_element_by_css_selector("span[class='iconfont iconnext']").click()
        time.sleep(1)
        driver.find_element_by_css_selector("span[class='iconfont iconprevious']").click()
    else:
        print("邮件列表只有1页！")
        return


