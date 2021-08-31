#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao

from web.script.test_sendmail.test_sendmail_editor import *



def test_sendmail_addresslist_a(driver,param):
    allure.dynamic.title("写信页面：从通讯录弹窗选中到收件人输入框")

    WebDriverWait(driver, 10, 0.1).until(
        EC.element_to_be_clickable((By.LINK_TEXT, "通讯录(点击选择更多联系人)"))).click()
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.XPATH, "//td/div/div/div/form/div/span[2]/input")))

    fail_to=""
    for i in param["addresslist"]:

        driver.find_element(By.XPATH, "//td/div/div/div/form/div/span[2]/input").clear()
        driver.find_element(By.XPATH, "//td/div/div/div/form/div/span[2]/input").send_keys(i)

        #driver.find_element(By.CSS_SELECTOR,"tr:nth-child(1) > td > .checkbox").click()
        time.sleep(1.5)
        #assert driver.find_element(By.CSS_SELECTOR,"td > .checkbox")==True,i
        try:
            driver.find_element(By.CSS_SELECTOR, "td > .checkbox").click()
        except:
            fail_to=i
            break
    driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary").click()

    assert len(fail_to) ==0 ,"该邮箱地址不存在于通讯录："+ fail_to


def test_sendmail_addresslist_b(driver,param):
    allure.dynamic.title("写信页面：从右侧通讯录选中联系人到输入框")


    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".u-input-1")))

    fail_to=""
    for i in param["addresslist"]:

        driver.find_element(By.CSS_SELECTOR, ".u-input-1").clear()
        driver.find_element(By.CSS_SELECTOR, ".u-input-1").send_keys(i)
        try:
            WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".expand:nth-child(1) a")))
            driver.find_element(By.CSS_SELECTOR, ".expand:nth-child(1) a").click()
        except:
            fail_to=i
            break
    #driver.find_element(By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary").click()

    assert len(fail_to) ==0 ,"该邮箱地址不存在于通讯录："+ fail_to





if __name__ == '__main__':
    pytest.main(['-s','test_sendmail_addresslist.py::test_sendmail_addresslist_b'])





