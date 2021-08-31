#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao

#阅读邮件内容,并把发件人、收件人、主题、正文内容，提交到缓存
from web.script.test_sendmail.test_sendmail_editor import *

def rep(str_email):
    to_str = str_email.replace("<", "").replace(">", "")
    return to_str

#阅读邮件正文
def Read_text(driver):
    iframe_body=driver.find_element(By.CSS_SELECTOR,".j-mail-content")
    driver.switch_to.frame(iframe_body)
    read_text=driver.find_element(By.XPATH, "/html/body").text
    driver.switch_to.default_content()
    return read_text

def test_email_Read_a(driver,param):
    allure.dynamic.title("阅读已打开的邮件内容,并把发件人、收件人、主题、正文内容，提交到缓存")
    WebDriverWait(driver, 10,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".mail-subject")))
    text_subject=driver.find_element(By.CSS_SELECTOR, ".mail-subject").text
    try:
        driver.find_element(By.CSS_SELECTOR,".u-btn-round > .icondown").click()
    except:
        pass
    Read_addressee_fr=driver.find_element(By.CSS_SELECTOR,".u-email > .address").text
    Read_to_list=[]
    text_addressee_to=driver.find_elements(By.CSS_SELECTOR,".j-contacts .address")
    for i in text_addressee_to:
        dd=i.text
        Read_to_list.append(rep(dd))

    read_text=Read_text(driver)

    param["Read_fr"]=(rep(Read_addressee_fr))
    param["Read_to_list"]=Read_to_list
    param["Read_subject"]=text_subject
    param["Read_text"]=(read_text)






if __name__ == '__main__':
    pytest.main(["-s","test_email_Read.py::test_email_Read_a"])