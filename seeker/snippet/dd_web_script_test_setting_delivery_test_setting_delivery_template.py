#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#收发邮件设置-模板信设置
from selenium.webdriver import ActionChains

from web.script.test_sendmail.test_sendmail_editor import *
from selenium.webdriver.support.ui import Select


def template_text(driver,txt):
    if txt==None:
        pass
    else:
        iframe_body = driver.find_element(By.XPATH, "/html/body/section/article/section/div[2]/div[2]/section/article/div[2]/div/section/div/div/section/div[3]/div[2]/div[2]/div/div/div[2]/iframe[1]")
        driver.switch_to.frame(iframe_body)
        driver.find_element(By.XPATH, "/html/body").clear()
        driver.find_element(By.XPATH, "/html/body").send_keys(txt)
        driver.switch_to.default_content()
        return True



def template_gainname(driver):
    dd=driver.find_elements(By.CSS_SELECTOR,".tpl-item")
    print(dd)
    name=[]
    for i in dd:
        name.append(i.text)
    return name

def template_hoverED(driver,name):
    ele=driver.find_elements(By.CSS_SELECTOR,".tpl-item")
    for i in ele:
        if name ==i.text:
            ActionChains(driver).move_to_element(i).perform()
            time.sleep(0.5)
            return i



def test_setting_delivery_template_a(driver, param):
    allure.dynamic.title("收发邮件设置-模板信设置-点击新增图标")

    """检查是否进入页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".m-setting-info")))

    """测试步骤"""
    #点击新增，进入新增页面
    driver.find_element(By.CSS_SELECTOR,".tpl-item >.iconadd").click()
    #检查是否进入新增编辑页面
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".j-tpl-name")))


def test_setting_delivery_template_add(driver, param):
    allure.dynamic.title("收发邮件设置-模板信设置-新增模版页面编辑")
    # param["template_name"]="模版名称"
    # param["template_body"]="模版内容"

    """检查是否进入页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".m-setting-info")))

    """测试步骤"""
    # #点击新增，进入新增页面，考虑有多个入口，进入页面操作拆分下
    # driver.find_element(By.CSS_SELECTOR,".tpl-item >.iconadd").click()
    #检查是否进入新增编辑页面
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".j-tpl-name")))

    #输入模版名称
    driver.find_element(By.CSS_SELECTOR,".j-tpl-name").clear()
    driver.find_element(By.CSS_SELECTOR,".j-tpl-name").send_keys(param["template_name"])

    #输入模版内容：
    template_text(driver,param["template_body"])

    #保存
    driver.find_element(By.CSS_SELECTOR,".j-edit-template > .u-btns-top > .u-btn-primary").click()

    #获取新增结果，断言
    result=WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    assert "新增写信模板成功" in result ,"新增写信模版失败"

    #成功跳转
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".tpl-item")))
    listname =template_gainname(driver)
    assert param["template_name"] in listname,"模版页面未见新添加模版名称"


def test_setting_delivery_template_update(driver, param):
    allure.dynamic.title("收发邮件设置-模板信设置-修改模版")
    # param["template_name"]="模版名称"
    # param["template_name_new"]="新的模版名称"
    # param["template_body"]="新的模版内容"


    """检查是否进入页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".m-setting-info")))

    """测试步骤"""
    #通过模版名称，选中对象鼠标悬停,点击编辑
    ele=template_hoverED(driver,param["template_name"])
    ele.find_element(By.CSS_SELECTOR,".edit > .iconfont").click()

    #检查是否进入编辑页面
    WebDriverWait(driver, 5, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".j-tpl-name")))

    #输入模版名称
    driver.find_element(By.CSS_SELECTOR,".j-tpl-name").clear()
    driver.find_element(By.CSS_SELECTOR,".j-tpl-name").send_keys(param["template_name_new"])

    #输入模版内容：
    template_text(driver,param["template_body"])

    #保存
    driver.find_element(By.CSS_SELECTOR,".j-edit-template > .u-btns-top > .u-btn-primary").click()

    #获取新增结果，断言
    result=WebDriverWait(driver, 5, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    assert "更新写信模板成功" in result ,"修改写信模版失败"

    #成功跳转
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".tpl-item")))
    listname =template_gainname(driver)
    assert param["template_name_new"] in listname,"模版页面未见新添加模版名称"

def test_setting_delivery_template_write(driver, param):
    allure.dynamic.title("收发邮件设置-模板信设置-选中模版写信")
    # param["template_name"] = "模版名称"
    """检查是否进入页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".m-setting-info")))
    """测试步骤"""
    # 通过模版名称，选中对象鼠标悬停，点击删除
    ele = template_hoverED(driver, param["template_name"])
    ele.find_element(By.CSS_SELECTOR, ".compose > .iconfont").click()

    """检查是否到达写信页面"""
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))

def test_setting_delivery_template_del(driver, param):
    allure.dynamic.title("收发邮件设置-模板信设置-删除指定模版")
    # param["template_name"]="模版名称"
    """检查是否进入页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".m-setting-info")))
    """测试步骤"""
    #通过模版名称，选中对象鼠标悬停，点击删除
    ele=template_hoverED(driver,param["template_name"])
    ele.find_element(By.CSS_SELECTOR,".del > .iconfont").click()

    #确认弹窗，点击确定
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".u-dialog-btns > .u-btn-primary"))).click()

    #获取删除果，断言
    result=WebDriverWait(driver, 5, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    assert "模板信删除成功" in result ,"模板信删除失败"

    assert param["template_name"] not in template_gainname(driver),"模版删除失败"


def test_setting_delivery_template_Alldel(driver, param):
    allure.dynamic.title("收发邮件设置-模板信设置-删除全部模版")
    """检查是否进入页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".m-setting-info")))
    """测试步骤"""
    #遍历所有模版，逐个删除
    ele=template_gainname(driver)
    if len(ele)>1:
        del ele[-1]
        for i in ele:
            ele2 = template_hoverED(driver, i)
            ele2.find_element(By.CSS_SELECTOR, ".del > .iconfont").click()
            # 确认弹窗，点击确定
            WebDriverWait(driver, 10, 0.1).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary"))).click()
            # 获取删除结果，断言
            result = WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "模板信删除成功" in result, "模板信删除失败"
            time.sleep(1)







if __name__ == '__main__':
    pytest.main(['-s','test_setting_delivery_template.py::test_setting_delivery_template_Alldel'])
