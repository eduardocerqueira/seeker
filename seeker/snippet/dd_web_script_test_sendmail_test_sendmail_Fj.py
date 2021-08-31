#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#处理附件上传
import sys

from web.script.test_sendmail.test_sendmail_editor import *
def test_sendmail_Fj_a(driver, param):
    allure.dynamic.title("写信页面，附件上传")
    newwindow(driver,nb=0)
    print(sys.platform)#linux结果为linux*，windows为win32/64
    send_to(driver,list_to=param["to"])
    driver.find_element(By.CSS_SELECTOR,".u-upload").click()




if __name__ == '__main__':
    pytest.main(['-s','test_sendmail_Fj.py::test_sendmail_Fj_a'])



