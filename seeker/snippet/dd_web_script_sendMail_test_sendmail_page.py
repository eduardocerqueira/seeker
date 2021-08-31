#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#页面转换按钮点击
from web.script.test_sendmail.test_sendmail_editor import *

def test_sendmail_page(driver,param):
    page_to=param["sendmail_page"]
    if page_to=="写信":
        WebDriverWait(driver, 10,0.1).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    elif page_to=="发送":
        WebDriverWait(driver, 10,0.1).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(.,'发 送')]"))).click()
        #driver.find_element(By.XPATH, "//span[contains(.,'发 送')]").click()


