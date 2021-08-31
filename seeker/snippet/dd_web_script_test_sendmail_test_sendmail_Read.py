#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#阅读已打开的写信页面，把收件人，主题，正文存入缓存
from web.script.test_sendmail.test_sendmail_editor import *
import re

def send_Read_fr_email(driver):
    fr=driver.find_element(By.CSS_SELECTOR, ".u-select-trigger:nth-child(4)").text.strip()
    if "<" in fr:
        send_fr_EMAIL=re.findall(r"<(.*?)>",fr)[0]
        return send_fr_EMAIL
    else:
        return fr

def send_Read_fr_NAME(driver):
    fr=driver.find_element(By.CSS_SELECTOR, ".u-select-trigger:nth-child(4)").text.strip()
    if "<" in fr:
        send_fr_NAME=re.findall(r'"(.*?)"',fr)[0]
        return send_fr_NAME
    else:
        return ""


def test_sendmail_Read_a(driver,param):
    allure.dynamic.title("阅读已打开的写信页面，把发件人、收件人、主题、正文、正文输入格式等存入缓存")
    newwindow(driver,nb=0)
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))


    #获取当前发件人邮箱地址
    param["send_Read_fr_Email"]=send_Read_fr_email(driver)
    #获取当前发件人名称，如没有则为空
    param["send_Read_fr_Name"]=send_Read_fr_NAME(driver)
    #获取收件人
    param["send_Read_to"]=driver.find_element(By.CSS_SELECTOR, ".j-form-item-to > .tag-editor").text
    #获取主题
    param["send_Read_subject"]=driver.find_element(By.CSS_SELECTOR, ".input").get_attribute("value")
    #获取正文内容
    param["send_Read_text"]=send_Read_text(driver)
    #获取附件内容,没有则为[]
    param["send_Read_FJ_list"]=send_Read_FJ(driver)
    #获取勾选框保存"已发送"的状态
    param["send_Read_save_sent"]=send_checkbox(driver,None, "li:nth-child(3) .checkbox")
    #获取正文编辑模式
    param["send_Read_body_type"]='html'
    try:
        driver.find_element(By.LINK_TEXT,"多媒体文本")
        param["send_Read_body_type"] = 'txt'
    except:
        pass







if __name__ == '__main__':
    pytest.main(["-s","test_sendmail_Read.py::test_sendmail_Read_a"])