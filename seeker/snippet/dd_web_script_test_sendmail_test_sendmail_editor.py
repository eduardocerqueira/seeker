#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
import os

import pytest
import allure

from selenium.webdriver.common.by import By
import time
import random
import string
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select





"""元素操作封装"""

#收件人
def send_to(driver,list_to):
    if list_to==None:
        pass
    else:
        driver.find_element(By.CSS_SELECTOR, ".j-form-item-to > .tag-editor").click()
        for i in list_to:
            driver.find_element(By.CSS_SELECTOR, ".tag-editor-tag > textarea").send_keys(i)


#抄送
def send_cc(driver,list_to):
    if list_to==None:
        pass
    else:
        driver.find_element(By.CSS_SELECTOR, ".j-form-item-cc > .tag-editor").click()
        for i in list_to:
            driver.find_element(By.CSS_SELECTOR, ".tag-editor-tag > textarea").send_keys(i)

#密送
def send_bcc(driver,list_to):
    if list_to==None:
        pass
    else:
        driver.find_element(By.LINK_TEXT, "密送").click()
        WebDriverWait(driver, 3,0.1).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR,".j-form-item-bcc > .tag-editor"))).click()
        #driver.find_element(By.CSS_SELECTOR, ".j-form-item-bcc > .tag-editor").click()
        for i in list_to:
            driver.find_element(By.CSS_SELECTOR, ".tag-editor-tag > textarea").send_keys(i)


#主题
def send_subject(driver,subject=None):
    if subject==None:
        return None
    elif subject==True:
        subject = (''.join(random.choice(string.ascii_letters) for x in range(10))) + str(int(time.time() * 1000))
        driver.find_element(By.CSS_SELECTOR, ".input").clear()
        driver.find_element(By.CSS_SELECTOR, ".input").send_keys(subject)
        return subject
    else:
        driver.find_element(By.CSS_SELECTOR, ".input").clear()
        driver.find_element(By.CSS_SELECTOR, ".input").send_keys(subject)
        return subject

#邮件正文
def send_text(driver,txt):
    if txt==None:
        pass
    else:
        iframe_body = driver.find_element(By.CSS_SELECTOR, ".ke-edit-iframe")
        driver.switch_to.frame(iframe_body)
        driver.find_element(By.XPATH, "/html/body").clear()
        driver.find_element(By.XPATH, "/html/body").send_keys(txt)
        driver.switch_to.default_content()
        return True

#关闭写信页面，处理弹窗
def send_CPM(driver,an="离开"):
    try:
        WebDriverWait(driver, 10, 0.1).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ".u-dialog-wrap")))
        if an == "离开":
            driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn:nth-child(2)").click()
        elif an == "离开并存草稿":
            driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn:nth-child(1)").click()
        elif an == "取消":
            driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-default").click()
        else:
            pass
        return True

    except:
        return False

#检查发送状态，60s内循环查看，如无结果则失败
def send_result(driver):
    num=0
    for i in range(0, 60):
        print(i)
        try:
            WebDriverWait(driver,10,0.1).until(
                     EC.element_to_be_clickable((By.LINK_TEXT, "[显示发送状态]"))).click()

            time.sleep(1)
            driver.find_element(By.CSS_SELECTOR, '.j-status-detail > .u-table')
            listdd = WebDriverWait(driver, 2, 0.1).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.j-status-detail > .u-table')))
            return listdd
        except:

            WebDriverWait(driver,2,0.1).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "[隐藏发送状态]"))).click()


    return "邮件已发送，60s内未查询到发送结果"

"""写信页面下面的复选框"""
# #选中紧急
# def send_urgent(driver,data):
#     if data==None:
#         pass
#     elif data==True:
#         driver.find_element(By.CSS_SELECTOR,"li:nth-child(4) .checkbox").click()
#     else:
#         pass
#
# #选中已读回执
# def send_rrr(driver,data):
#     if data==None:
#         pass
#     elif data==True:
#         driver.find_element(By.CSS_SELECTOR,"li:nth-child(5) .checkbox").click()
#     else:
#         pass
# #选中定时邮件，不进行时间更改，预期结果去草稿箱检查
# def send_time(driver,data):
#     if data in [None,False]:
#         pass
#     elif data==True:
#         driver.find_element(By.CSS_SELECTOR,"li:nth-child(6) .checkbox").click()
#     else:
#         pass
#获取勾选框状态
def send_checkbox_type(ele):
    print(ele)
    if "checkbox-checked" in ele:
        return True
    else:
        return False



#通用复选框函数
def send_checkbox(driver,data,ele_data):
    ele=driver.find_element(By.CSS_SELECTOR,ele_data)
    if data==None:
        return send_checkbox_type(ele.get_attribute("class"))
    elif data==True:
        if "checkbox-checked" in ele.get_attribute("class"):
            pass
        else:
            ele.click()
    elif data == False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele.click()
        else:
            pass
    else:
        pass
    return ele.get_attribute("class")


#选中阅读即焚
def send_snaps(driver,data):
    ele=driver.find_element(By.CSS_SELECTOR,"li:nth-child(7) .checkbox")
    if data == None:
        return send_checkbox_type(ele.get_attribute("class"))
    elif data==False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele.click()
        else:
            pass
    else:
        if "checkbox-checked" in ele.get_attribute("class"):
            driver.find_element(By.NAME, "autoDelReadLimit").clear()
            driver.find_element(By.NAME, "autoDelReadLimit").send_keys(data)
        else:
            ele.click()
            try:
                WebDriverWait(driver, 5, 0.1).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary"))).click()
            except:
                pass
            driver.find_element(By.NAME, "autoDelReadLimit").clear()
            driver.find_element(By.NAME,"autoDelReadLimit").send_keys(data)

    return ele.get_attribute("class")


#选中邮件加密
def send_encryption(driver,data):
    ele=driver.find_element(By.CSS_SELECTOR,"li:nth-child(8) .checkbox")
    if data == None:
        return send_checkbox_type(ele.get_attribute("class"))
    elif data==False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele.click()
        else:
            pass
    else:
        if "checkbox-checked" in ele.get_attribute("class"):
            WebDriverWait(driver, 10, 0.1).until(
                EC.visibility_of_element_located((By.ID, "bt-mailcipher-password")))
            driver.find_element(By.ID, "bt-mailcipher-password").send_keys(data)
        else:
            ele.click()
            WebDriverWait(driver, 10, 0.1).until(
                EC.visibility_of_element_located((By.ID, "bt-mailcipher-password")))
            driver.find_element(By.ID, "bt-mailcipher-password").send_keys(data)







#阅读邮件正文
def send_Read_text(driver):
    iframe_body=driver.find_element(By.CSS_SELECTOR,".ke-edit-iframe")
    driver.switch_to.frame(iframe_body)
    read_text = driver.find_element(By.XPATH, "/html/body").text
    driver.switch_to.default_content()
    return read_text


#查看附件信息
def send_Read_FJ(driver):
    send_Read_FJ_list = []
    try:
        att = driver.find_elements(By.CSS_SELECTOR, ".j-name")
        for i in att:
            send_Read_FJ_list.append(i.text)
        return send_Read_FJ_list
    except:
        return send_Read_FJ_list
# def send_function(driver,data):
#     if data==None:
#         pass
#     elif data[0]== "定时发送":
#         send_time(driver,data[1])
#     elif data[0]=="阅后即焚":
#         send_snaps(driver,data[1])
#     elif data[0]=="邮件加密":
#         send_encryption(driver,data[1])
#     else:
#         pass


"""初始化"""
def initialize(driver,url):
    if "sid" in url: #页面可继续访问
        """回到首页"""
        WebDriverWait(driver, 5, 0.1).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '.j-lylogo > img'))).click()
        """关闭全部子窗口"""
        try:
            WebDriverWait(driver, 5, 0.1).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '.iconcloseall'))).click()
            send_CPM(driver,"离开")
        except:
            pass

    else:
        pass
        # url = "https://cloud.mail.wo.cn/coremail/XT5/index.jsp?sid=%s" % param['sid']
        # driver.get(url)
        #预期调用登录


def newwindow_test(driver):
    handles = driver.window_handles
    handle = driver.current_window_handle
    if len(handles)>1:
        for i in handles:
            print(i)
            if i != handle:
                driver.switch_to_window(handles[0])


def newwindow(driver,nb): #nb为0则为当前停留窗口
    handles = driver.window_handles
    #handle = driver.current_window_handle
    if len(handles)>1:
        handles = driver.window_handles
        driver.switch_to.window(handles[nb])




#{"to":["18611662981@wo.cn;"],"cc":[""],"bcc":[""],"subject":"主题","send_text":"邮件正文","send_savesent":True,"send_urgent":None,"send_rrr":None,"send_time":None,"send_snaps":None,"send_encryption":None}
#{"to":["18611662981@wo.cn;"],"cc":[""],"bcc":[""],"subject":"主题","send_text":"邮件正文"}
def test_sendmail_editor_a(driver,param):
    allure.dynamic.title("写信页面：内容编写")
    newwindow(driver,nb=0)
    #检查是否在写信页面
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))

    send_to(driver,list_to=param["to"])
    driver.find_element(By.LINK_TEXT, "取消抄送")
    # driver.find_element(By.LINK_TEXT, "抄送").click()
    send_cc(driver,list_to=param["cc"])
    send_bcc(driver,list_to=param["bcc"])
    subject=send_subject(driver,subject=param["subject"]) #默认唯一id，可支持自定义send_subject(driver,subject)
    param["subject"]=subject
    send_text(driver,txt=param["send_text"])



def test_sendmail_editor_b(driver,param):
    allure.dynamic.title("写信页面：群发单显-内容编写")
    newwindow(driver,nb=0)

    #检查是否在写信页面
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))

    #群发单显
    driver.find_element(By.LINK_TEXT, "群发单显").click()
    send_to(driver,list_to=param["to"])
    subject=send_subject(driver,subject=param["subject"]) #默认唯一id，可支持自定义send_subject(driver,subject)
    param["subject"]=subject
    send_text(driver,txt=param["send_text"])
    #dd={"send_urgent":None,"send_rrr":None,"send_time":None,"send_snaps":None,"send_encryption":None}




def test_sendmail_editor_d(driver,param):
    allure.dynamic.title("写信页面：按下拉框顺序选择发件人")
    newwindow(driver,nb=0)
    #param["fr"]=1
    #检查是否在写信页面
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))
    driver.find_element(By.CSS_SELECTOR,".u-select-trigger:nth-child(4)").click()

    ele=".u-form-item .u-select-item:nth-child(%s)" %  int(param["fr"])

    driver.find_element(By.CSS_SELECTOR,ele).click()




def test_sendmail_editor_c(driver,param):
    allure.dynamic.title("发送邮件全流程，指定本网邮件+外域邮件，检查发件状态")
    #初始化回到首页
    initialize(driver,driver.current_url)
    #点击写信
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    #检查是否在写信页面
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))

    send_to(driver,list_to=param["to"])
    driver.find_element(By.LINK_TEXT, "取消抄送")
    # driver.find_element(By.LINK_TEXT, "抄送").click()
    send_cc(driver,list_to=param["cc"])
    send_bcc(driver,list_to=param["bcc"])
    subject=send_subject(driver,subject=param["subject"]) #默认唯一id，可支持自定义send_subject(driver,subject)
    param["subject"]=subject
    send_text(driver,txt=param["send_text"])


    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    #检查是否到达写信成功页面
    WebDriverWait(driver,20).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
    dd=driver.find_element(By.XPATH,"//div[2]/div[2]/div/div/div/div").text

    assert "邮件已发送" == dd ,"如未匹配，邮件发送失败"

    listdd = send_result(driver)
    if type(listdd) != list:
        assert listdd =="成功到达对方服务器",listdd
    else:
        for i in listdd:
            str_email=i.text
            li=str_email.split("\n")
            del li[0]
            for j in li:
                l=j.split(" ")
                print(l)
                assert l[1] == "成功到达对方服务器","该邮箱发送失败："+l[0]

    #关闭所有写信窗口
    driver.find_element(By.CSS_SELECTOR, '.iconcloseall').click()


def test_sendmail_editor_checkbox(driver,param):
    allure.dynamic.title("写信页面：置底勾选栏")
    # param["send_savesent"]=True
    # param["send_urgent"] =False
    # param["send_rrr"] =False
    # param["send_time"] =False
    # param["send_snaps"] =False
    # param["send_encryption"] =False
    #initialize(driver,driver.current_url)
    # WebDriverWait(driver, 10,0.1).until(
    #     EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    #newwindow(driver,nb=0)
    #检查是否在写信页面
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))


    try:
        driver.find_element(By.CSS_SELECTOR,".bt-hd-more-show > a").click() #置底栏如果有更多按钮就点击展开
    except:
        pass


    #dd={"send_urgent":None,"send_rrr":None,"send_time":None,"send_snaps":None,"send_encryption":None}
    #是否选中保存到已发送
    send_checkbox(driver, param["send_savesent"], "li:nth-child(3) .checkbox")
    #是否选中紧急
    send_checkbox(driver,param["send_urgent"],"li:nth-child(4) .checkbox")
    #是否已读回执
    send_checkbox(driver,param["send_rrr"],"li:nth-child(5) .checkbox")
    #是否选中定时邮件
    send_checkbox(driver,param["send_time"],"li:nth-child(6) .checkbox")
    #是否选中阅读即焚，如选中仅输入次数
    send_snaps(driver,param["send_snaps"])
    #是否选中邮件加密，如选中仅输入6位数密码
    send_encryption(driver,param["send_encryption"])







if __name__ == '__main__':
    pytest.main(['-s','test_sendmail_editor.py::test_sendmail_editor_d'])





