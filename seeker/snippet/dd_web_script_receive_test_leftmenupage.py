#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time
from selenium.webdriver.support.wait import WebDriverWait

def test_leftmenupage(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
# 1. 打开收件列表
    menu = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in menu:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    sjx = driver.find_element_by_css_selector("div[class='totals-info']").text
    pd = "收件箱"
    if pd in sjx:
        print(sjx)
    else:
        print("收件列表未打开！")
    time.sleep(1)
# 2. 打开代办邮件列表
    for g in menu:
        if g.text == "待办邮件":
            g.click()
    time.sleep(1)
    dbyj = driver.find_element_by_css_selector("div[class='totals-info']").text
    pd2 = "待办邮件"
    if pd2 in dbyj:
        print(dbyj)
    else:
        print("待办邮件未打开！")
    time.sleep(1)
# 3. 打开草稿箱列表
    for h in menu:
        if h.text == "草稿箱":
            h.click()
    time.sleep(1)
    cgx = driver.find_element_by_css_selector("div[class='totals-info']").text
    pd3 = "草稿箱"
    if pd3 in cgx:
        print(cgx)
    else:
        print("草稿箱未打开！")
    time.sleep(1)
# 4. 打开已发送列表
    for k in menu:
        if k.text == "已发送":
            k.click()
    time.sleep(1)
    yfs = driver.find_element_by_css_selector("div[class='totals-info']").text
    pd4 = "已发送"
    if pd4 in yfs:
        print(yfs)
    else:
        print("已发送未打开！")
    time.sleep(1)
# 5. 打开其他文件夹—已删除
    for j in menu:
        if j.text == "其他文件夹":
            j.click()
    menu3 = driver.find_elements_by_css_selector("div[class='cnt']")
    for m in menu3:
        if m.text == "已删除":
            m.click()
    time.sleep(1)
    ysc = driver.find_element_by_css_selector("div[class='totals-info']").text
    pd5 = "已删除"
    if pd5 in ysc:
        print(ysc)
    else:
        print("已删除未打开！")
    time.sleep(1)
# 6. 打开其他文件夹—垃圾邮件
    for l in menu:
        if l.text == "其他文件夹":
            l.click()
    menu4 = driver.find_elements_by_css_selector("div[class='cnt']")
    for n in menu4:
        if n.text == "垃圾邮件":
            n.click()
    time.sleep(1)
    ljyj = driver.find_element_by_css_selector("div[class='totals-info']").text
    pd6 = "垃圾邮件"
    if pd6 in ljyj:
        print(ljyj)
    else:
        print("垃圾邮件未打开！")
    time.sleep(1)
# 7. 打开其他文件夹—病毒文件夹
    for o in menu:
        if o.text == "其他文件夹":
            o.click()
    menu5 = driver.find_elements_by_css_selector("div[class='cnt']")
    for p in menu5:
        if p.text == "病毒文件夹":
            p.click()
    time.sleep(1)
    bdwj = driver.find_element_by_css_selector("div[class='totals-info']").text
    pd7 = "病毒文件夹"
    if pd7 in bdwj:
        print(bdwj)
    else:
        print("病毒文件夹未打开！")

# 打开写信、收信页面
def test_readwritemail(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 查找是否有打开的写信页面，关闭
    xx = driver.find_elements_by_css_selector("span[class='iconfont icontabclose close']")
    for x in xx:
        x.click()
    time.sleep(1)
    # 点击写信按钮
    driver.find_element_by_css_selector("button[class='u-btn u-btn-default u-btn-large btn-compose j-mlsb']").click()
    time.sleep(1)

    fsan = driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div/section/article/div[2]/div[2]/div/div[3]/span[2]").text

    if fsan == "发 送":
        print("写信页面正常打开！")
    else:
        print("写信页面打开失败！")
        return
    time.sleep(1)
    # 点击收信图标
    driver.find_element_by_css_selector("button[class='u-btn u-btn-default u-btn-large btn-inbox j-mlsb']").click()
    time.sleep(1)
    sjx = driver.find_element_by_css_selector("div[class='totals-info']").text
    if "收件箱" in sjx:
        print("收信列表正常打开！")
    else:
        print("收信列表打开失败！")
        return

# 其他文件夹新增、管理
def test_otherfile(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 其他文件夹-新增
    menu = driver.find_elements_by_css_selector("div[class='cnt']")
    for m in menu:
        if m.text == "其他文件夹":
            m.click()
    driver.find_element_by_css_selector("i[class='j-add iconfont iconadd']").click()
    time.sleep(1)
    driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[2]/section/article/div[2]/div/section/div/div/section/div/section[1]/form/div[1]/button[2]").click()
    # 其他文件夹-管理
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    menu2 = driver.find_elements_by_css_selector("div[class='cnt']")
    for m in menu2:
        if m.text == "其他文件夹":
            m.click()
    time.sleep(1)
    driver.find_element_by_css_selector("i[class='j-setting iconfont iconset']").click()
    wwj = driver.find_element_by_link_text("文件夹管理").text
    if wwj == "文件夹管理":
        print("文件夹管理页面正常打开！")
    else:
        print("页面打开失败")
        return