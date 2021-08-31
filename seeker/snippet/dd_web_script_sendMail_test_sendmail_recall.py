#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#邮件撤回功能
from web.script.test_sendmail.test_sendmail_editor import *



def test_sendmail_recall_a(driver,param):
    allure.dynamic.title("发送本域名邮箱，点击撤回")

    initialize(driver,driver.current_url)
    driver.refresh()#该用例必须刷新页面，请勿剔除
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor textarea")))

    #测试过程
    send_to(driver,list_to=param["to"])
    driver.find_element(By.LINK_TEXT, "取消抄送")
    # driver.find_element(By.LINK_TEXT, "抄送").click()
    send_cc(driver,list_to=param["cc"])
    send_bcc(driver,list_to=param["bcc"])
    send_subject(driver,subject=param["subject"]) #默认唯一id，可支持自定义send_subject(driver,subject)
    send_text(driver,txt=param["send_text"])


    driver.find_element(By.XPATH,"//span[contains(.,'发 送')]").click()
    WebDriverWait(driver,20).until(
        EC.visibility_of_element_located((By.LINK_TEXT, "[显示发送状态]")))
    dd=driver.find_element(By.XPATH,"//div[2]/div[2]/div/div/div/div").text

    driver.find_element(By.LINK_TEXT, "[召回邮件]").click()
    WebDriverWait(driver, 10,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".u-dialog-btns > .u-btn-primary"))).click()
    #driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary").click()
    for i in range(1,10):
        print(i)
        dis=driver.find_element(By.CSS_SELECTOR,".u-lmask:nth-child(3) .u-lmask-loading").is_displayed()
        print(dis)
        if dis ==True:
            time.sleep(1)
        else:
            break
    title=WebDriverWait(driver,10,0.1).until(
        EC.visibility_of_element_located((By.XPATH,"/html/body/div[6]/div[1]/div[2]"))).text
    print(title)
    driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary").click()
    assert "邮件已发送" == dd ,"如未匹配，邮件发送失败"
    assert "召回成功" in title,"撤回本网邮件功能异常，预期本网邮件可立即撤回"

# def test_sendmail_recall_b(driver):
#     driver.find_element(By.LINK_TEXT, "[召回邮件]").click()
#     WebDriverWait(driver, 10,0.1).until(
#         EC.element_to_be_clickable((By.CSS_SELECTOR,".u-dialog-btns > .u-btn-primary"))).click()
#     #driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary").click()
#     for i in range(1,10):
#         print(i)
#         dis=driver.find_element(By.CSS_SELECTOR,".u-lmask:nth-child(3) .u-lmask-loading").is_displayed()
#         if dis ==True:
#             time.sleep(1)
#         else:
#             break
#     title=WebDriverWait(driver,10,0.1).until(
#         EC.visibility_of_element_located((By.XPATH,"/html/body/div[6]/div[1]/div[2]"))).text
#     print(title)
#     driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary").click()
#     assert "召回成功" in title,"撤回本网邮件功能异常"
#
# #u-lmask-loading
#
#
# if __name__ == '__main__':
#     pytest.main(['-s','test_sendmail_recall.py::test_sendmail_recall_a'])
#
#



