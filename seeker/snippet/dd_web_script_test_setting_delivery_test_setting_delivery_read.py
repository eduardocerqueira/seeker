#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#收发邮件设置-读信设置
from web.script.test_sendmail.test_sendmail_editor import *
from selenium.webdriver.support.ui import Select





def read_newwindowtoreadletter(driver,datatype=None):
    ele1=driver.find_element(By.CSS_SELECTOR,".u-form-item:nth-child(1) div:nth-child(1) .radio")
    ele2=driver.find_element(By.CSS_SELECTOR,".u-form-item:nth-child(1) > .u-form-value > div:nth-child(2) .radio")

    assert datatype in ["在原窗口打开", "在新窗口打开"], "参数错误"
    if datatype=="在原窗口打开":
        if 'radio-checked' in ele1.get_attribute("class"):
            pass
        else:
            ele1.click()
    elif datatype=="在新窗口打开":
        if 'radio-checked' in ele2.get_attribute("class"):
            pass
        else:
            ele2.click()
    else:
        pass
    return datatype

def read_afterdel(driver,datatype=None):
    ele1=driver.find_element(By.CSS_SELECTOR,".j-read-form > .u-form-item:nth-child(2) div:nth-child(1) .radio")
    ele2=driver.find_element(By.CSS_SELECTOR,".j-read-form > .u-form-item:nth-child(2) > .u-form-value > div:nth-child(2) .radio")

    assert datatype in ['回到"文件夹"页面', '继续阅读下一封邮件(推荐选择)'], "参数错误"
    if datatype=='回到"文件夹"页面':
        if 'radio-checked' in ele1.get_attribute("class"):
            pass
        else:
            ele1.click()
    elif datatype=='继续阅读下一封邮件(推荐选择)':
        if 'radio-checked' in ele2.get_attribute("class"):
            pass
        else:
            ele2.click()
    else:
        pass
    return datatype

def read_op_readreceipt(driver,datatype=None):
    ele1=driver.find_element(By.CSS_SELECTOR,".j-read-form > .u-form-item:nth-child(3) div:nth-child(1) .radio")
    ele2=driver.find_element(By.CSS_SELECTOR,".j-read-form > .u-form-item:nth-child(3) > .u-form-value > div:nth-child(2) .radio")
    ele3=driver.find_element(By.CSS_SELECTOR,"div:nth-child(3) > label > .radio")

    assert datatype in ['提示我是否发送回执(推荐选择)', '忽略所有回执请求','自动发送回执'], "参数错误"
    if datatype=='提示我是否发送回执(推荐选择)':
        if 'radio-checked' in ele1.get_attribute("class"):
            pass
        else:
            ele1.click()
    elif datatype=='忽略所有回执请求':
        if 'radio-checked' in ele2.get_attribute("class"):
            pass
        else:
            ele2.click()
    elif datatype=='自动发送回执':
        if 'radio-checked' in ele3.get_attribute("class"):
            pass
        else:
            ele3.click()
    else:
        pass
    return datatype

def read_subject_refw_prefix_expand(driver,datatype=None):
    ele1=driver.find_element(By.CSS_SELECTOR,".u-form-item:nth-child(4) div:nth-child(1) .radio")
    ele2=driver.find_element(By.CSS_SELECTOR,".u-form-item:nth-child(4) > .u-form-value > div:nth-child(2) .radio")

    assert datatype in ['展开', '收起'], "参数错误"
    if datatype=='展开':
        if 'radio-checked' in ele1.get_attribute("class"):
            pass
        else:
            ele1.click()
    elif datatype=='收起':
        if 'radio-checked' in ele2.get_attribute("class"):
            pass
        else:
            ele2.click()
    else:
        pass
    return datatype

def test_setting_delivery_read_a(driver, param):
    allure.dynamic.title("收发邮件设置-读信设置")
    # param["setting_delivery_read"]={"read_newwindowtoreadletter":"在新窗口打开","read_afterdel":'继续阅读下一封邮件(推荐选择)',
    #                                 "read_op_readreceipt":'忽略所有回执请求',
    #                                 "read_subject_refw_prefix_expand":"收起"
    #                         }
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".u-form-item:nth-child(1) div:nth-child(1) .radio")))

    #读信：["在原窗口打开", "在新窗口打开"]
    read_newwindowtoreadletter(driver,datatype=param["setting_delivery_read"]["read_newwindowtoreadletter"])
    #删除邮件后：['回到"文件夹"页面', '继续阅读下一封邮件(推荐选择)']
    read_afterdel(driver,datatype=param["setting_delivery_read"]["read_afterdel"])
    #收到"已读回执"请求时：['提示我是否发送回执(推荐选择)', '忽略所有回执请求','自动发送回执']
    read_op_readreceipt(driver,datatype=param["setting_delivery_read"]["read_op_readreceipt"])
    #主题“回复/转发”前缀：['展开', '收起']
    read_subject_refw_prefix_expand(driver,datatype=param["setting_delivery_read"]["read_subject_refw_prefix_expand"])

    driver.find_element(By.CSS_SELECTOR,".m-read-set .u-btn-primary").click()
    result=WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    assert "设置更改成功" in result ,"设置更改失败"
    driver.refresh()








if __name__ == '__main__':
    pytest.main(['-s','test_setting_delivery_read.py::test_setting_delivery_read_a'])
