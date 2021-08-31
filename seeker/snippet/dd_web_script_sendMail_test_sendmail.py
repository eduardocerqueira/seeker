#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
import pytest
import allure
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time
import random
import string
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



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
def send_subject(driver,subject=(''.join(random.choice(string.ascii_letters) for x in range(10)))+str(int(time.time()*1000))):
    if subject==None:
        pass
    else:
        driver.find_element(By.CSS_SELECTOR, ".input").clear()
        driver.find_element(By.CSS_SELECTOR, ".input").send_keys(subject)
        return subject

#邮件正文
def send_text(driver,txt):
    if txt==None:
        pass
    else:
        iframe_body=driver.find_element(By.XPATH,"/html/body/section/article/section/div[2]/div/section/article/div[2]/div[2]/div/div[4]/form/div[2]/div[1]/div[2]/iframe[1]")
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
    for i in range(0, 60):
        print(i)
        try:
            WebDriverWait(driver,10,0.1).until(
                     EC.element_to_be_clickable((By.LINK_TEXT, "[显示发送状态]"))).click()
            driver.find_element(By.CSS_SELECTOR, '.j-status-detail > .u-table')
            listdd = WebDriverWait(driver, 10, 0.1).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.j-status-detail > .u-table')))
            return listdd
        except:
            WebDriverWait(driver,10,0.1).until(
                EC.element_to_be_clickable((By.LINK_TEXT, "[隐藏发送状态]"))).click()
            time.sleep(1)

    return "邮件已发送，60s内未查询到发送结果"
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
#{"to":["18611662981@wo.cn;"],"cc":[""],"bcc":[""],"subject":"主题","send_text":"邮件正文"}
def test_sendmail_a(driver,param):
    allure.dynamic.title("发送邮件")

    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    send_to(driver,list_to=param["to"])
    driver.find_element(By.LINK_TEXT, "取消抄送")
    # driver.find_element(By.LINK_TEXT, "抄送").click()
    send_cc(driver,list_to=param["cc"])
    send_bcc(driver,list_to=param["bcc"])
    send_subject(driver,subject=param["subject"]) #默认唯一id，可支持自定义send_subject(driver,subject)
    send_text(driver,txt=param["send_text"])

    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
    dd=driver.find_element(By.XPATH,"//div[2]/div[2]/div/div/div/div").text
    #关闭所有写信窗口
    driver.find_element(By.CSS_SELECTOR, '.iconcloseall').click()
    assert "邮件已发送" == dd ,"如未匹配，邮件发送失败"




def test_sendmail_b(driver,param):
    allure.dynamic.title("发送邮件，本网邮件+外域邮件，检查发件状态")
    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    send_to(driver,list_to=param["to"])
    driver.find_element(By.LINK_TEXT, "取消抄送")
    # driver.find_element(By.LINK_TEXT, "抄送").click()
    send_cc(driver,list_to=param["cc"])
    send_bcc(driver,list_to=param["bcc"])
    send_subject(driver,subject=param["subject"]) #默认唯一id，可支持自定义send_subject(driver,subject)
    send_text(driver,txt=param["send_text"])


    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    WebDriverWait(driver,200).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
    dd=driver.find_element(By.XPATH,"//div[2]/div[2]/div/div/div/div").text

    assert "邮件已发送" == dd ,"如未匹配，邮件发送失败"
    # for i in range(1, 15):
    #     print(i)
    #     WebDriverWait(driver,10,0.1).until(
    #         EC.element_to_be_clickable((By.LINK_TEXT, "[显示发送状态]"))).click()
    #     dis=driver.find_element(By.PARTIAL_LINK_TEXT,"信件正在处理中").is_displayed()
    #     if dis == True:
    #         WebDriverWait(driver,10,0.1).until(
    #             EC.element_to_be_clickable((By.LINK_TEXT, "[隐藏发送状态]"))).click()
    #         time.sleep(1)
    #     else:
    #         break

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


def test_sendmail_c(driver,param):
    allure.dynamic.title("发送邮件：群发单显")
    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))


    #群发单显
    driver.find_element(By.LINK_TEXT, "群发单显").click()
    send_to(driver,list_to=param["to"])
    send_subject(driver,subject=param["subject"]) #默认唯一id，可支持自定义send_subject(driver,subject)
    send_text(driver,txt=param["send_text"])

    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
    dd=driver.find_element(By.XPATH,"//div[2]/div[2]/div/div/div/div").text
    #关闭所有写信窗口
    driver.find_element(By.CSS_SELECTOR, '.iconcloseall').click()
    assert "邮件已发送" == dd ,"如未匹配，邮件发送失败"





if __name__ == '__main__':
    pytest.main(['-s','test_sendmail.py::test_sendmail_a'])





