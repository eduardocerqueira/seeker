#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#写信页面，选择信纸


#阅读邮件正文
def writingpaper_url(driver):
    iframe_body=driver.find_element(By.CSS_SELECTOR,".ke-edit-iframe")
    driver.switch_to.frame(iframe_body)
    read_ure=driver.find_element(By.XPATH, "/html/body/table").get_attribute('background')
    driver.switch_to.default_content()
    return str(read_ure)


from web.script.test_sendmail.test_sendmail_editor import *

def test_sendmail_writingpaper_a(driver, param):
    allure.dynamic.title("写信页面，选择信纸")
    newwindow(driver,nb=0)
    param["writingpaper"]="5" #可输入1-6 ，按顺序排列选中1为无信纸

    WebDriverWait(driver, 5,0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".j-form-item-to > .tag-editor")))
    ele = ".stationery-item:nth-child(%s) img" % param["writingpaper"]
    driver.find_element(By.LINK_TEXT,"信纸").click()
    WebDriverWait(driver, 5,0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,ele))).click()
    if int(param["writingpaper"])>1:
        url = writingpaper_url(driver)
        param["writingpaper_url"] = url

        assert r"https://mail.wo.cn/coremail/common/stationery/" in url,'正文内未检测到信纸'





if __name__ == '__main__':
    pytest.main(['-s','test_sendmail_writingpaper.py::test_sendmail_writingpaper_a'])



