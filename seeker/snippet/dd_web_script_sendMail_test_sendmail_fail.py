#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao

from web.script.test_sendmail.test_sendmail_editor import *



def test_sendmail_fail_a(driver):
    allure.dynamic.title("缺少收件人，点击发送")
    #driver.implicitly_wait(10)
    #driver.refresh()
    #确定需要元素已经加载
    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    #测试过程
    # send_to(driver,list_to=[""])
    # send_cc(driver,list_to=[""])
    # send_bcc(driver,list_to=[""])
    # subject_str=send_subject(driver) #默认唯一id，可支持自定义send_subject(driver,subject)
    # print(subject_str)
    # send_text(driver,txt="正文")

    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    CPM_text=WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".u-dialog-message"))).text
    assert "请填写收件人地址" in CPM_text, "无收件人发送邮件，提示异常"
    driver.find_element(By.CSS_SELECTOR,".u-dialog-btns > .u-btn").click()
    # 关闭所有写信窗口
    driver.find_element(By.CSS_SELECTOR, '.iconcloseall').click()


def test_sendmail_fail_b(driver):
    allure.dynamic.title("缺少主题与正文邮件，点击发送")
    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    #测试过程
    send_to(driver,list_to=["18611662981@wo.cn;"])
    subject_id=send_subject(driver,"") #默认唯一id，可支持自定义send_subject(driver,subject)

    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".u-dialog-message")))
    driver.find_element(By.CSS_SELECTOR,".u-dialog-btns > .u-btn-default").click() #取消发送

    driver.find_element(By.XPATH, "//span[contains(.,'发 送')]").click()
    CPM_text = WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".u-dialog-message"))).text
    assert "确定不需要写 邮件标题 吗" in CPM_text, "写信，主题为空提示异常"

    driver.find_element(By.CSS_SELECTOR,".u-dialog-btns > .u-btn-primary").click() #确定发送
    CPM_text = WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".u-dialog-message"))).text

    assert "确定不需要写 邮件内容 吗" in CPM_text, "写信，正文为空提示异常"

    """正文为空再次确认"""
    driver.find_element(By.CSS_SELECTOR,".u-dialog-btns > .u-btn-primary").click() #确定发送


    WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
    dd=driver.find_element(By.XPATH,"//div[2]/div[2]/div/div/div/div").text

    assert "邮件已发送" == dd, "如未匹配，邮件发送失败"

def test_sendmail_fail_c(driver):
    allure.dynamic.title("收件人格式异常，点击发送（经验证该功能只要存在一个有效邮箱都可正常发送）")
    #driver.implicitly_wait(10)
    #driver.refresh()
    #确定需要元素已经加载
    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    #测试过程
    send_to(driver,list_to=["xxxxx"])
    # send_cc(driver,list_to=[""])
    # send_bcc(driver,list_to=[""])
    subject_str=send_subject(driver) #默认唯一id，可支持自定义send_subject(driver,subject)
    # print(subject_str)
    send_text(driver,txt="正文")

    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    CPM_text=WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".u-dialog-message"))).text
    assert "请填写收件人地址" in CPM_text, "无收件人发送邮件，提示异常"
    driver.find_element(By.CSS_SELECTOR,".u-dialog-btns > .u-btn").click()




# if __name__ == '__main__':
#     pytest.main(['-s','test_sendmail_fail.py::test_sendmail_fail_c'])





