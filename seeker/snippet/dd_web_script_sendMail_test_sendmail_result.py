#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#写信结果页面处理
from web.script.test_sendmail.test_sendmail_editor import *
def test_sendmail_result_a(driver, param):
    allure.dynamic.title("写信结果页面,检查是否有邮件发送成功文案")
    #driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
    dd=driver.find_element(By.XPATH,"//div[2]/div[2]/div/div/div/div").text
    #关闭所有写信窗口
    driver.find_element(By.CSS_SELECTOR, '.iconcloseall').click()
    assert "邮件已发送" == dd ,"如未匹配，邮件发送失败"


def test_sendmail_result_b(driver,param):
    allure.dynamic.title("写信结果页面，检查发件状态")
    WebDriverWait(driver,20).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
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
