#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#写信页面：模版信选中或新增

from web.script.test_sendmail.test_sendmail_editor import *
from web.script.test_setting_delivery.test_setting_delivery_template import template_text


def sendmail_template_ele_state(driver):
    ele_state=True
    try:
        driver.find_element(By.CSS_SELECTOR, ".addToCompose")
    except:
        ele=False
    return ele_state


def test_sendmail_template_a(driver, param):
    allure.dynamic.title("写信页面，右侧模版信-新增模版信-默认回到写信页面")
    newwindow(driver,nb=0)
    param["template_name"]="模版名称"
    param["template_body"]="模版内容"
    #检查是否已经到了写信页面
    WebDriverWait(driver, 5,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))

    driver.find_element(By.LINK_TEXT,"模板信").click()
    #点击模版信新增图标
    WebDriverWait(driver, 5,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".create-mail-tpl > .iconfont"))).click()


    #检查是否进入新增编辑页面
    WebDriverWait(driver, 5, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".j-tpl-name")))
    #输入模版名称
    driver.find_element(By.CSS_SELECTOR,".j-tpl-name").clear()
    driver.find_element(By.CSS_SELECTOR,".j-tpl-name").send_keys(param["template_name"])

    #输入模版内容：
    template_text(driver,param["template_body"])

    #保存
    driver.find_element(By.CSS_SELECTOR,".j-edit-template > .u-btns-top > .u-btn-primary").click()

    #获取新增结果，断言
    result=WebDriverWait(driver, 5, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    assert "新增写信模板成功" in result ,"新增写信模版失败"

    try:
        #如有弹出框直接选中直接切换
        #time.sleep(3)
        WebDriverWait(driver, 5, 0.1).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-default"))).click()

        # # 如有弹出框直接选中切换并存草稿
        # WebDriverWait(driver, 5, 0.1).until(
        #     EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary"))).click()
    except:
        pass
    #验证回到写信页面并且正文与模版内容一致
    read_text=send_Read_text(driver)
    assert read_text==param['template_body'],"与选中模版内容不匹配"

def test_sendmail_template_b(driver, param):
    allure.dynamic.title("写信页面，选中模版")
    newwindow(driver,nb=0)
    #param["template_name"]="ddsad"

    WebDriverWait(driver, 5,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))
    driver.find_element(By.LINK_TEXT,"模板信").click()
    if sendmail_template_ele_state(driver)==True:
        # WebDriverWait(driver, 5,0.1).until(
        #     EC.element_to_be_clickable((By.CSS_SELECTOR,".f-csp:nth-child(3)"))).click()
        template_name=param["template_name"]
        ele="//li[contains(.,'%s')]" % template_name
        WebDriverWait(driver, 5,0.1).until(
            EC.element_to_be_clickable((By.XPATH,ele))).click()
        try:
            #如有弹出框直接选中直接切换
            time.sleep(3)
            WebDriverWait(driver, 5, 0.1).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-default"))).click()

            # # 如有弹出框直接选中切换并存草稿
            # WebDriverWait(driver, 5, 0.1).until(
            #     EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary"))).click()
        except:
            pass

    read_text=send_Read_text(driver)
    assert read_text==param['template_body'],"与选中模版内容不匹配"











if __name__ == '__main__':
    pytest.main(['-s','test_sendmail_template.py::test_sendmail_template_a'])



