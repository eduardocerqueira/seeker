#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#回复邮件与全部回复
from web.script.test_sendmail.test_sendmail_editor import *

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
def send_subject(driver,subject= (''.join(random.choice(string.ascii_letters) for x in range(10))) + str(int(time.time() * 1000))):
    if subject==None:
        pass
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
        iframe_body=driver.find_element(By.XPATH,"/html/body/section/article/section/div[2]/div/section/article/div[2]/div[2]/div/div[4]/form/div[2]/div[1]/div[2]/iframe[1]")
        driver.switch_to.frame(iframe_body)
        driver.find_element(By.XPATH, "/html/body").clear()
        driver.find_element(By.XPATH, "/html/body").send_keys(txt)
        driver.switch_to.default_content()
        return True

#阅读邮件正文
def Read_text(driver):
    iframe_body=driver.find_element(By.CSS_SELECTOR,".ke-edit-iframe")
    driver.switch_to.frame(iframe_body)
    read_text=driver.find_element(By.XPATH, "/html/body").text
    driver.switch_to.default_content()
    return read_text

def test_sendmail_Read_a(driver,param):
    allure.dynamic.title("阅读已打开的写信页面，把收件人，主题，正文存入缓存")
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))

    item_to=driver.find_element(By.CSS_SELECTOR, ".j-form-item-to > .tag-editor").text
    item_subject=driver.find_element(By.CSS_SELECTOR, ".input").get_attribute("value")
    param["send_Read_to_list"]=item_to
    param["send_Read_subject"]=item_subject
    param["send_Read_text"]=Read_text(driver)



if __name__ == '__main__':
    pytest.main(["-s","test_sendmail_Read.py::test_sendmail_Read_a"])