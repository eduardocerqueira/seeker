#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao

from web.script.test_sendmail.test_sendmail_editor import *



def test_sendmail_draft_a(driver):
    allure.dynamic.title("存草稿箱-关闭未完成的写信页面")
    #driver.implicitly_wait(10)
    #driver.refresh()
    #确定需要元素已经加载
    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    #测试过程
    send_to(driver,list_to=["18611662981@wo.cn;"])
    driver.find_element(By.LINK_TEXT, "取消抄送")
    # driver.find_element(By.LINK_TEXT, "抄送").click()
    send_cc(driver,list_to=["18611662981@wo.cn;"])
    send_bcc(driver,list_to=["18611662981@wo.cn;"])
    subject_str=send_subject(driver) #默认唯一id，可支持自定义send_subject(driver,subject)
    send_text(driver,txt="正文")


    driver.find_element(By.CSS_SELECTOR, '.iconcloseall').click()
    CPM=send_CPM(driver,"离开并存草稿")
    result=WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    # result=driver.find_element(By.CSS_SELECTOR,".body").text

    assert True == CPM,"关闭弹窗，未提示是否存草稿"
    assert "保存草稿成功" in result ,"保存草稿失败"

def test_sendmail_draft_b(driver):
    allure.dynamic.title("存草稿箱-编辑过程中存草稿")
    initialize(driver,driver.current_url)
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    #测试过程
    send_to(driver,list_to=["18611662981@wo.cn;"])
    driver.find_element(By.LINK_TEXT, "取消抄送")
    # driver.find_element(By.LINK_TEXT, "抄送").click()
    send_cc(driver,list_to=["18611662981@wo.cn;"])
    send_bcc(driver,list_to=["18611662981@wo.cn;"])
    subject_id=send_subject(driver) #默认唯一id，可支持自定义send_subject(driver,subject)
    print(subject_id)
    send_text(driver,txt="正文")
    driver.find_element(By.CSS_SELECTOR, ".j-tbl-draft").click()
    result=WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text

    #存完草稿可继续编辑写信页面
    result2=send_text(driver,txt="正文11111")

    assert "保存草稿成功" in result ,"保存草稿失败"
    assert True == result2,"存草稿后，重写正文失败"


    driver.find_element(By.CSS_SELECTOR, '.iconcloseall').click()
    send_CPM(driver, "离开")
    #已预留唯一主题（subject_id），可遍历草稿箱







